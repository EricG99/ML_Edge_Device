# ML_Helpfunctions/base_inference.py
import logging
import sys
import threading
import time
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import os

from ML_Helpfunctions import pipeline_utils, Load_Prepare_Data
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient

# Hinzugefügter Import für die Typ-Annotation und Initialisierung
from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor

class BaseInferenceProcessor(ABC):
    """
    Abstrakte Basisklasse für die Inferenz. Definiert die Schnittstelle für alle Modelle.
    Kapselt das Laden von Artefakten und die Orchestrierung eines einzelnen Inferenzschritts.
    """
    def __init__(self, config: dict, folder_flag: str):
        self.config = config
        self.folder_flag = folder_flag
        self.model = None
        self.scaler = None
        self.y_scaler = None
        self.feature_list = None
        self.training_config = None
        self.target_feature = self.config.get('base_features', [None])[0]
        self._mqtt_client = None
        self._lock = threading.Lock()
        self.latest_payload = None
        self.step_counter = 0
        self._pending_entry = None

        # --- ANGEPASSTER/NEUER TEIL START ---
        # Gemeinsame Attribute werden zentral in der Basisklasse initialisiert,
        # um Codeduplizierung in den Subklassen (CNN1D, LSTM, etc.) zu vermeiden.
        self.data_processor = RealTimeDataProcessor(config)
        self.lags = int(self.config.get("lags", 1))
        # --- ANGEPASSTER/NEUER TEIL ENDE ---

        # FINALER FIX: Daten für den Batch-Iterator einmalig vorladen, um den Zustand zu halten
        self._batch_data_df = None
        self._batch_data_position = 0
        if self.config.get("loading_strategy") == "split":
            logging.info("Lade Batch-Daten einmalig für zustandsbehafteten Iterator...")
            df = Load_Prepare_Data.load_test_data_by_fraction(
                config=self.config,
                train_fraction=self.config.get("train_fraction", 0.7),
                make_date_as_index=False
            )
            df.columns = df.columns.str.lower()
            self._batch_data_df = df.sort_values("datetime").reset_index(drop=True)
            logging.info(f"{len(self._batch_data_df)} Zeilen für die Inferenz vorgeladen.")

    def _model_tag(self) -> str | None:
        """
        Erzeugt einen robusten, kurzen Tag für den Dateinamen: 
        'keras', 'tflite_<stem>' oder 'sklearn'.
        """
        tag = "unknown"
        try:
            import os
            model_instance = getattr(self, "model", None)
            if model_instance is None:
                return "unloaded"

            # Hole den Klassennamen für eine robuste Prüfung
            class_name = model_instance.__class__.__name__.lower()

            # --- KORRIGIERTE, ROBUSTE PRÜFUNG ---
            # 1. Prüfe auf TFLite Interpreter
            if class_name == 'interpreter':
                stem = "model"
                # Hole den spezifischen Dateinamen aus der Konfig (z.B. 'model_quant_float16.tflite')
                model_filename = (self.config or {}).get("model_filename")
                if model_filename:
                    stem = os.path.splitext(os.path.basename(model_filename))[0]
                tag = f"tflite_{stem}"
            
            # 2. Prüfe auf Keras Modell
            elif 'keras' in str(type(model_instance)):
                tag = "keras"
            
            # 3. Prüfe auf Scikit-Learn Modelle (z.B. RandomForest)
            elif hasattr(model_instance, '_estimator_type'):
                tag = "sklearn"
            
        except Exception:
            tag = "error" # Fallback-Tag im Fehlerfall
        
        return tag

    def save_step_result(
        self,
        prediction_entry: dict,
        total_time_s: float | None = None,
        cpu_percent: float | None = None,
        ram_mb: float | None = None,
        ram_percent: float | None = None,
        output_path: str | None = None
    ) -> str | None:
        """
        Persistiert einen Inferenz-Schritt in eine eindeutige CSV-Datei pro Modellvariante.
        """
        import os
        import logging

        if not prediction_entry:
            return None

        if not hasattr(self, "_predictions_file_path"):
            self._predictions_file_path = None

        # --- NEU: Eindeutigen Dateinamen pro Modellvariante erstellen (nur beim ersten Aufruf) ---
        path_to_use = output_path
        if path_to_use is None and self._predictions_file_path is None:
            try:
                paths = self.config.get("paths", {})
                pred_dir = paths.get("Prediction_Data", ".")
                os.makedirs(pred_dir, exist_ok=True) # Sicherstellen, dass der Ordner existiert

                run_id = self.config.get("run_id", "unknown_run")
                algo = getattr(self, "folder_flag", "algo")
                
                # Verwende einen stabilen Namen für die Datenquelle
                data_name = os.path.splitext(os.path.basename(self.config.get("dataset", "live_data")))[0]
                
                model_tag = self._model_tag()
                
                # Dateinamen zusammensetzen
                filename = f"StepPredictions_{run_id}_{algo}_{data_name}"
                if model_tag:
                    filename += f"__{model_tag}"
                filename += ".csv"
                
                generated_path = os.path.join(pred_dir, filename)
                self._predictions_file_path = generated_path
                path_to_use = self._predictions_file_path
                logging.info(f"Ergebnis-CSV für diesen Lauf: {generated_path}")

            except Exception as e:
                logging.error(f"Fehler bei der Erstellung des eindeutigen Dateipfads: {e}")
                # Fallback, um einen Absturz zu verhindern
                path_to_use = self._predictions_file_path or None

        elif path_to_use is None:
            path_to_use = self._predictions_file_path
        # --- ENDE NEU ---

        date = prediction_entry.get("datetime")
        true_value = prediction_entry.get("true_value")

        # RF-spezifisches Alignment (Logik bleibt unverändert)
        future_list = (
            prediction_entry.get("future_forecast")
            or prediction_entry.get("rolling_forecast")
            or []
        )
        pred_t = prediction_entry.get("prediction", None)
        horizon = int(self.config.get("horizon", 1))
        folder_lower = str(getattr(self, "folder_flag", "")).lower()
        is_rf = folder_lower in ("random_forest", "random forest", "rf")

        if is_rf:
            aligned_forecast = []
            if horizon >= 1:
                if pred_t is not None:
                    aligned_forecast.append(pred_t)
                else:
                    aligned_forecast.append(future_list[0] if future_list else None)
            if horizon > 1:
                aligned_forecast.extend(list(future_list[:max(0, horizon - 1)]))
            forecast = aligned_forecast
        else:
            forecast = list(future_list[:horizon]) if horizon > 0 else list(future_list)

        inference_time_s = prediction_entry.get("inference_time_s", None)
        breakdown = prediction_entry.get("time_breakdown", None)

        # Der Aufruf der Hilfsfunktion schreibt nun in den eindeutigen Pfad
        final_path = pipeline_utils.append_prediction_step(
            config=self.config,
            date=date,
            true_value=true_value,
            forecast=forecast,
            inference_time_s=inference_time_s,
            total_time_s=total_time_s,
            cpu_percent=cpu_percent,
            ram_mb=ram_mb,
            ram_percent=ram_percent,
            breakdown=breakdown,
            output_path=path_to_use
        )

        return final_path


    def flush_pending_entry(self) -> dict | None:
        """Gibt den letzten, noch nicht abgeschlossenen Vorhersage-Eintrag zurück."""
        return self._pending_entry

    def _update_latest_payload(self, data: dict):
        with self._lock:
            self.latest_payload = data

    def load_artifacts(self):
        try:
            logging.info(f"Lade Artefakte für '{self.folder_flag}'...")
            (
                self.scaler, self.feature_list, self.model,
                self.training_config, self.y_scaler
            ) = pipeline_utils.load_model_artifacts_for_inference(self.config, self.folder_flag)
            self._post_load_artifacts()
            logging.info("✅ Artefakte erfolgreich geladen.")
        except Exception as e:
            logging.error(f"Artefakte konnten nicht geladen werden: {e}", exc_info=True)
            sys.exit(1)

    def set_artifacts_from_memory(self, artifacts: dict):
        """Übernimmt neue Artefakte (Hot-Swap) ohne vorhandene Werte zu verlieren."""
        prev_config = getattr(self, "config", None)
        prev_features = getattr(self, "feature_list", None)

        # Modell & Scaler – nur überschreiben, wenn vorhanden
        if artifacts.get("model") is not None:
            self.model = artifacts["model"]
        if artifacts.get("scaler") is not None:
            self.scaler = artifacts["scaler"]
        if artifacts.get("y_scaler") is not None:
            self.y_scaler = artifacts["y_scaler"]

        # Config & Feature-Liste – niemals auf None setzen
        self.config = artifacts.get("config") or prev_config
        self.feature_list = artifacts.get("features") or prev_features

        # Hook nach Hot-Swap (z. B. Puffer übernehmen)
        try:
            self._on_artifacts_swapped()
        except Exception as e:
            logging.error("Fehler im _on_artifacts_swapped-Hook: %s", e)

    
    def process_step(self, payload: dict) -> dict | None:
        import time
        import numpy as np
        import pandas as pd
        import logging

        if not payload:
            return None

        start_processing_time = time.perf_counter()

        # 1) modell-spezifische Aufbereitung
        input_data, timestamp_t, true_value_t = self._prepare_input_data(payload)

        # 2) noch kein ausreichend gefülltes Fenster
        if input_data is None:
            if true_value_t is not None:
                return {
                    "datetime": timestamp_t,
                    "prediction": None,
                    "true_value": float(true_value_t),
                    "future_forecast": [],
                    # für UI/CSV: trotzdem mit erwarteten Keys schreiben
                    "cpu_percent": None,
                    "ram_mb": None,
                }
            return None

        # Für Hot-Swap-Reseed merken
        self._last_input_data = input_data

        # 3) Inferenz (skaliert) + Rückskalierung
        future_pred_scaled, t_inf_ms = self._run_inference_unified(input_data)
        future_pred_unscaled = self._inverse_transform_prediction(future_pred_scaled)

        # 4) Horizon hart auf H bringen (ohne künstliche Wiederholung)
        H = int(self.config.get("horizon", 1))
        future_pred_unscaled = np.asarray(future_pred_unscaled, dtype=float).reshape(-1)
        if future_pred_unscaled.size < H:
            pad = np.full(H - future_pred_unscaled.size, np.nan, dtype=float)
            future_pred_unscaled = np.concatenate([future_pred_unscaled, pad], axis=0)
        elif future_pred_unscaled.size > H:
            future_pred_unscaled = future_pred_unscaled[:H]

        # 5) Eintrag für Zeitpunkt t aus Pending von t-1 übernehmen
        completed_entry_for_t = None
        if self._pending_entry is not None:
            completed_entry_for_t = self._pending_entry.copy()
            completed_entry_for_t["true_value"] = None if true_value_t is None else float(true_value_t)
            # inference_time_s aus pending übernehmen
            it_ms = self._pending_entry.get("model_inference_time_ms")
            if it_ms is not None:
                completed_entry_for_t["inference_time_s"] = float(it_ms) / 1000.0
            # **NEU/Backcompat:** System-Metriken in erwarteten Keys
            completed_entry_for_t["cpu_percent"] = self._pending_entry.get("cpu_percent")
            completed_entry_for_t["ram_mb"] = self._pending_entry.get("ram_mb")

        # 6) System-Metriken abgreifen (in erwarteten Keys!)
        cpu_percent = None
        ram_mb = None
        try:
            from ML_Helpfunctions import pipeline_utils as PU
            cpu_percent = float(PU.get_cpu_usage())
            mem = PU.get_memory_usage()  # dict: {"total_gb","used_gb","percent"} oder "N/A"
            ram_mb = float(mem["used_gb"]) * 1024.0 if isinstance(mem, dict) and mem.get("used_gb") != "N/A" else None
            ram_percent = float(mem["percent"]) if isinstance(mem, dict) and mem.get("used_gb") != "N/A" else None
        except Exception:
            pass

        # 7) neuen Pending für t+1 setzen (mit erwarteten Keys)
        import pandas as pd
        self._pending_entry = {
            "datetime": timestamp_t + pd.Timedelta(seconds=self.config.get("inference_interval_sec", 1.0)),
            "prediction": float(future_pred_unscaled[0]) if np.isfinite(future_pred_unscaled[0]) else None,
            "true_value": None,
            "rolling_forecast": future_pred_unscaled.tolist(),
            "cpu_percent": cpu_percent,
            "ram_percent": ram_percent,
            "ram_usage": mem,  
            "model_inference_time_ms": float(t_inf_ms),
            "total_processing_time_ms": float((time.perf_counter() - start_processing_time) * 1000.0),
        }

        # 8) Housekeeping + Fallback für ersten Schritt
        self.step_counter += 1
        if completed_entry_for_t is None:
            completed_entry_for_t = {
                "datetime": timestamp_t,
                "prediction": None,
                "true_value": float(true_value_t) if true_value_t is not None else None,
                "cpu_percent": cpu_percent,
                "ram_mb": ram_mb,
                "ram_percent": ram_percent,
            }

        # kompletten Horizon am t-Eintrag hinterlegen
        completed_entry_for_t["future_forecast"] = future_pred_unscaled.tolist()

        # 9) Logging
        true_str = (
            f"{completed_entry_for_t['true_value']:.4f}"
            if completed_entry_for_t.get("true_value") is not None else "N/A"
        )
        pred_str_t = (
            f"{completed_entry_for_t['prediction']:.4f}"
            if completed_entry_for_t.get("prediction") is not None else "Warte..."
        )
        pred_str_t_plus_1 = (
            f"{future_pred_unscaled[0]:.4f}" if np.isfinite(future_pred_unscaled[0]) else "N/A"
        )
        ts_str = timestamp_t.strftime("%H:%M:%S.%f")[:-3] if hasattr(timestamp_t, "strftime") else str(timestamp_t)

        logging.info(
            f"Step [{self.step_counter}] | Zeit: {ts_str} | "
            f"Vergleich für t: ECHT={true_str}, PRED={pred_str_t} | "
            f"NEUER FORECAST für t+1 -> {pred_str_t_plus_1}"
        )

        return completed_entry_for_t



    def get_data_source_iterator(self):
        strategy = self.config.get("loading_strategy", "split")
        if strategy == "split":
            return self._batch_iterator
        elif strategy == "live_mqtt":
            self._start_mqtt_client()
            return self._mqtt_iterator
        else:
            raise ValueError(f"Unbekannte Ladestrategie: {strategy}")

    def stop(self):
        if self._mqtt_client:
            self._mqtt_client.stop()
            logging.info("MQTT-Client gestoppt.")
            
            
    def save_final_results(self, all_predictions: list):
        if not all_predictions:
            logging.warning("Keine Vorhersagen zum Speichern vorhanden.")
            return

        logging.info(f"Speichere {len(all_predictions)} Vorhersagen...")
        try:
            import numpy as np
            import pandas as pd

            df = pd.DataFrame(all_predictions)

            # gültige Zeilen: true_value vorhanden UND mindestens eine Forecast-Spalte vorhanden
            has_fc = df["rolling_forecast"].notna() | df["future_forecast"].notna()
            valid_rows_mask = df["true_value"].notna() & has_fc
            df_valid = df[valid_rows_mask]

            if df_valid.empty:
                logging.warning("Keine gültigen Paare aus wahren Werten und Vorhersagen gefunden. Metriken können nicht berechnet werden.")
                pipeline_utils.save_metrics_summary(
                    metrics={"error": "No valid data to calculate metrics"},
                    run_config=self.config,
                    training_config=self.training_config or {},
                    paths=self.config.get("paths", {})
                )
                return

            # Forecast-Spalte wählen
            h = int(self.config.get("horizon", 1))
            if "rolling_forecast" in df_valid.columns and df_valid["rolling_forecast"].iloc[0] is not None:
                fc_col = "rolling_forecast"
            elif "future_forecast" in df_valid.columns and df_valid["future_forecast"].iloc[0] is not None:
                fc_col = "future_forecast"
            else:
                logging.error("Finale Auswertung: Keine Forecast-Spalte gefunden.")
                return

            # Hilfs-Funktion: Vektoren auf Länge H bringen (ohne künstliches Wiederholen)
            def _to_fixed(vec, H):
                arr = np.asarray(vec, dtype=float).reshape(-1)
                if arr.size >= H:
                    return arr[:H]
                out = np.full(H, np.nan, dtype=float)
                out[:arr.size] = arr
                return out

            # PRED (N x H)
            y_pred_list = [v for v in df_valid[fc_col] if v is not None]
            if not y_pred_list:
                logging.warning("Forecast-Spalte enthält keine gültigen Listen. Metriken können nicht berechnet werden.")
                return
            y_pred = np.vstack([_to_fixed(v, h) for v in y_pred_list])  # (N, H)

            # TRUE korrekt zuordnen: pred(t+h) ↔ true(t+h)
            y_true_1d = df_valid["true_value"].to_numpy()               # Länge N
            N = y_pred.shape[0]
            H = y_pred.shape[1]
            M = max(N - H, 0)                                           # gemeinsame Länge für alle Horizonte

            if M <= 0:
                logging.warning("Zu wenig Punkte für horizon-gerechte Auswertung.")
                pipeline_utils.save_metrics_summary(
                    metrics={"error": "Not enough aligned samples for metrics"},
                    run_config=self.config,
                    training_config=self.training_config or {},
                    paths=self.config.get("paths", {})
                )
                return

            y_pred_aligned = y_pred[:M, :]                              # (M, H)
            y_true_cols = [y_true_1d[h_ : h_ + M] for h_ in range(1, H + 1)]
            y_true_aligned = np.column_stack(y_true_cols)               # (M, H)

            # Metriken berechnen
            logging.info(f"Berechne Metriken für {len(y_true_aligned)} konsistente Datenpunkte.")
            metrics = pipeline_utils.evaluate_all_metrics(y_true_aligned, y_pred_aligned, horizon=int(self.config.get("horizon", 1)))

            # Zusatzinfos für JSON
            extra = {}
            if hasattr(self, "_predictions_file_path") and self._predictions_file_path:
                extra["predictions_file_path"] = self._predictions_file_path
            model_tag = self._model_tag()
            if model_tag:
                extra["model_tag"] = model_tag

            metrics_json_path = pipeline_utils.save_metrics_summary(
                metrics=metrics,
                run_config=self.config,
                training_config=self.training_config or {},
                paths=self.config.get("paths", {}),
                extra_info=extra if extra else None
            )
            try:
                import os
                if getattr(self, "_predictions_file_path", None):
                    logging.info(f"📄 StepPredictions CSV: {os.path.abspath(self._predictions_file_path)}")
                if metrics_json_path:
                    logging.info(f"📁 ErrorMetrics JSON: {os.path.abspath(metrics_json_path)}")
                agg_csv = os.path.join(self.config.get("paths", {}).get("Error_Metrics", "."), "ErrorMetrics_all_runs.csv")
                logging.info(f"📊 ErrorMetrics (aggregiert, CSV): {os.path.abspath(agg_csv)}")
            except Exception:
                pass

        except Exception as e:
            logging.error(f"Fehler beim finalen Speichern der Ergebnisse: {e}", exc_info=True)



    def _run_inference_unified(self, input_data: np.ndarray):
        start = time.perf_counter()
        if hasattr(self.model, "get_input_details"):
            interpreter = self.model
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # --- KORREKTUR: Datentyp-Prüfung und manuelle Quantisierung der Eingabe ---
            input_tensor_index = input_details[0]["index"]
            expected_dtype = input_details[0]["dtype"]
            
            input_tensor = np.asarray(input_data)

            if expected_dtype == np.int8:
                # Quantisierungsparameter aus dem Modell extrahieren
                quantization_params = input_details[0].get('quantization', (1.0, 0))
                scale, zero_point = quantization_params
                
                # Manuelle Quantisierung der Float32-Eingabedaten nach INT8
                if scale != 0.0:
                    input_tensor = (input_tensor / scale + zero_point).astype(np.int8)
                else:
                    input_tensor = input_tensor.astype(np.int8) # Fallback
                
                interpreter.set_tensor(input_tensor_index, input_tensor)
            else:
                # Standardfall für FLOAT32 oder andere Typen
                interpreter.set_tensor(input_tensor_index, input_tensor.astype(expected_dtype))
            # --- ENDE EINGABE-KORREKTUR ---
            
            interpreter.invoke()
            
            pred = interpreter.get_tensor(output_details[0]["index"])

            # --- KORREKTUR: De-Quantisierung der Ausgabe ---
            # Wenn der Output ebenfalls quantisiert ist, muss er zurück in float konvertiert werden
            if output_details[0]['dtype'] == np.int8:
                quantization_params = output_details[0].get('quantization', (1.0, 0))
                scale, zero_point = quantization_params
                if scale != 0.0:
                    pred = (pred.astype(np.float32) - zero_point) * scale
            # --- ENDE AUSGABE-KORREKTUR ---

        elif hasattr(self.model, "predict"):
            try:
                pred = self.model.predict(input_data, verbose=0)
            except TypeError:
                pred = self.model.predict(input_data)
        else:
            raise TypeError(f"Das Modell vom Typ {type(self.model)} hat keine bekannte Inferenzmethode.")
        
        dur_ms = (time.perf_counter() - start) * 1000.0
        return np.asarray(pred), dur_ms

    def _start_mqtt_client(self):
        if not self._mqtt_client:
            self._mqtt_client = MqttInferenceClient(
                broker_ip=self.config['MQTT_BROKER_IP'],
                port=self.config['MQTT_PORT'],
                topic=self.config['MQTT_TOPIC'],
                on_message_callback=self._update_latest_payload
            )
            self._mqtt_client.run()

    def _mqtt_iterator(self, max_steps):
        count = 0
        while max_steps == 'infinite' or count < max_steps:
            with self._lock:
                payload = self.latest_payload
                self.latest_payload = None
            if payload:
                yield payload
                count += 1
            else:
                time.sleep(0.05)

    def _batch_iterator(self, max_steps):
        # FINALER FIX: Dieser Iterator ist jetzt zustandsbehaftet und setzt sich nicht mehr zurück.
        if self._batch_data_df is None or self._batch_data_position >= len(self._batch_data_df):
            logging.info("Batch-Datenquelle ist erschöpft.")
            return

        start_pos = self._batch_data_position
        end_pos = min(start_pos + max_steps, len(self._batch_data_df))
        
        subset_df = self._batch_data_df.iloc[start_pos:end_pos]

        for _, row in subset_df.iterrows():
            payload = row.to_dict()
            payload['datetime'] = pd.to_datetime(payload['datetime'])
            yield payload
        
        # Position für den nächsten Aufruf aktualisieren
        self._batch_data_position = end_pos

            
    def _post_load_artifacts(self):
        # Trainings-Config in die Laufzeit-Config übernehmen
        if getattr(self, "training_config", None):
            self.config["training_config"] = self.training_config
            self.config["lags"] = int(self.training_config.get("lags", self.config.get("lags", 1)))
            self.config["horizon"] = int(self.training_config.get("horizon", self.config.get("horizon", 1)))

        # bisherigen Buffer sichern, dann Prozessor mit korrekten lags neu aufsetzen
        old_buf = getattr(self.data_processor, "_buffer", None)
        self.data_processor = RealTimeDataProcessor(self.config)
        if old_buf is not None and not old_buf.empty:
            # Buffer wieder einhängen (prime)
            df = old_buf.reset_index().rename(columns={"index": "datetime"})
            self.data_processor.prime_buffer(df)

    @abstractmethod
    def _prepare_input_data(self, payload: dict) -> tuple[np.ndarray | None, any, float | None]:
        raise NotImplementedError

    @abstractmethod
    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        raise NotImplementedError