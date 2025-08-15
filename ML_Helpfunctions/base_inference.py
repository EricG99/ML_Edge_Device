# ML_Helpfunctions/base_inference.py
import logging
import sys
import threading
import time
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import os

from ML_Helpfunctions import Pipeline_Utils, Load_Prepare_Data
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient

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

    def save_step_result(
        self,
        prediction_entry: dict,
        total_time_s: float | None = None,
        cpu_percent: float | None = None,
        ram_mb: float | None = None,
        # --- NEUE ZEILE START ---
        ram_percent: float | None = None,
        # --- NEUE ZEILE ENDE ---
        output_path: str | None = None
    ) -> str | None:
        """
        Persistiert einen Inferenz-Schritt (True, n-Step-Forecast, Zeiten, CPU/RAM).
        Erwartete Keys in prediction_entry:
        - 'datetime', 'true_value',
        - 'future_forecast' ODER 'rolling_forecast' (Liste mit H Werten),
        - optional: 'inference_time_s', 'time_breakdown'
        """
        import os
        import logging

        if not prediction_entry:
            return None

        # 🔒 Robust gegen fehlendes Attribut
        if not hasattr(self, "_predictions_file_path"):
            self._predictions_file_path = None

        date = prediction_entry.get("datetime")
        true_value = prediction_entry.get("true_value")

        # --- RF-spezifisches Alignment: pred_h1 = Vorhersage FÜR t; danach t+1..t+(H-1)
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
                    aligned_forecast.append(future_list[0] if len(future_list) > 0 else None)
            if horizon > 1:
                aligned_forecast.extend(list(future_list[:max(0, horizon - 1)]))
            forecast = aligned_forecast
        else:
            # LSTM (und andere): unverändert – future_list entspricht bereits der Logik
            forecast = list(future_list[:horizon]) if horizon > 0 else list(future_list)

        inference_time_s = prediction_entry.get("inference_time_s", None)
        breakdown = prediction_entry.get("time_breakdown", None)

        # --- Schreiben / Anhängen der Step-CSV
        path = Pipeline_Utils.append_prediction_step(
            config=self.config,
            date=date,
            true_value=true_value,
            forecast=forecast,
            inference_time_s=inference_time_s,
            total_time_s=total_time_s,
            cpu_percent=cpu_percent,
            ram_mb=ram_mb,
            # --- NEUE ZEILE START ---
            ram_percent=ram_percent,
            # --- NEUE ZEILE ENDE ---
            breakdown=breakdown,
            output_path=output_path or self._predictions_file_path  # darf None sein
        )

        # Merken (ab hier existiert der Pfad garantiert)
        if self._predictions_file_path is None:
            self._predictions_file_path = path

        # ✅ GEWÜNSCHTES LOGGING DER PFADE
        try:
            abs_step_path = os.path.abspath(path) if path else None
            if abs_step_path:
                logging.info(f"📝 append_prediction_step -> Datei: {abs_step_path}")
        except Exception:
            pass

        try:
            metrics_dir = (self.config.get("paths") or {}).get("Error_Metrics")
            if metrics_dir:
                metrics_summary_path = os.path.abspath(os.path.join(metrics_dir, "metrics_summary.csv"))
                if os.path.exists(metrics_summary_path):
                    logging.info(f"📈 Error-Metrics (Summary) vorhanden: {metrics_summary_path}")
                else:
                    logging.info(f"📈 Error-Metrics (Summary, geplant): {metrics_summary_path}")
        except Exception:
            pass

        # Zusätzlich einmalig (beim ersten Schreiben) einen klaren Hinweis loggen
        try:
            if prediction_entry.get("_first_write_log_once") and abs_step_path:
                logging.info(f"📄 StepPredictions gestartet: {abs_step_path}")
        except Exception:
            pass

        return path


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
            ) = Pipeline_Utils.load_model_artifacts_for_inference(self.config, self.folder_flag)
            self._post_load_artifacts()
            logging.info("✅ Artefakte erfolgreich geladen.")
        except Exception as e:
            logging.error(f"Artefakte konnten nicht geladen werden: {e}", exc_info=True)
            sys.exit(1)

    def set_artifacts_from_memory(self, shared_model_dict: dict):
        """
        Übernimmt Modell/Scaler/Featureliste/Konfig aus dem gemeinsamen Speicher.
        Ruft danach _post_load_artifacts() (für Klassenspezifika) und optional
        den Hook _on_artifacts_swapped(), damit z. B. DataProcessor mit neuer
        Config/Features neu aufgebaut werden kann.
        """
        self.model = shared_model_dict["model"]
        self.scaler = shared_model_dict["scaler"]
        self.y_scaler = shared_model_dict.get("y_scaler")
        self.feature_list = shared_model_dict["features"]
        self.config = shared_model_dict["config"]

        # Klassenspezifische Folgearbeiten (z. B. TFLite/Keras-Umschaltung etc.)
        self._post_load_artifacts()

        # NEU: Optionaler Hook in Kindklassen (RF/LSTM), um z. B. DataProcessor neu zu initialisieren
        if hasattr(self, "_on_artifacts_swapped"):
            try:
                self._on_artifacts_swapped()
            except Exception as hook_err:
                logging.error(f"Fehler im _on_artifacts_swapped-Hook: {hook_err}", exc_info=True)

        logging.info("✅ Artefakte aus dem Speicher übernommen (inkl. Hook-Aufruf, falls vorhanden).")

    
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
            from ML_Helpfunctions.Pipeline_Utils import PipelineUtils
            cpu_percent = float(PipelineUtils.get_cpu_usage())
            ram_mb = float(PipelineUtils.get_memory_usage())
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
            "ram_mb": ram_mb,
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
            df = pd.DataFrame(all_predictions)
            valid_rows_mask = df["true_value"].notna() & df["rolling_forecast"].notna()
            df_valid = df[valid_rows_mask]

            if df_valid.empty:
                logging.warning("Keine gültigen Paare aus wahren Werten und Vorhersagen gefunden. Metriken können nicht berechnet werden.")
                return

            horizon = int(self.config.get("horizon", 1))

            import numpy as np
            h = int(self.config.get("horizon", 1))

            # Forecast-Spalte erkennen
            if "rolling_forecast" in df_valid.columns:
                fc_col = "rolling_forecast"
            elif "future_forecast" in df_valid.columns:
                fc_col = "future_forecast"
            else:
                logging.error("Finale Auswertung: Keine Forecast-Spalte ('rolling_forecast' oder 'future_forecast') gefunden.")
                return None

            def _to_fixed(vec, H):
                arr = np.asarray(vec, dtype=float).reshape(-1)
                if arr.size >= H:
                    return arr[:H]
                out = np.empty(H, dtype=float)
                out[:] = np.nan
                out[:arr.size] = arr
                return out

            y_pred = np.vstack([_to_fixed(v, h) for v in df_valid[fc_col].tolist()])
            y_true_1d = df_valid["true_value"].to_numpy()
            y_true = np.tile(y_true_1d.reshape(-1, 1), reps=(1, y_pred.shape[1]))

            logging.info(f"Berechne Metriken für {len(y_true)} konsistente Datenpunkte.")
            metrics = Pipeline_Utils.evaluate_all_metrics(y_true, y_pred, horizon=horizon)

            extra = {}
            if hasattr(self, "_predictions_file_path") and self._predictions_file_path:
                extra["predictions_file_path"] = self._predictions_file_path

            metrics_json_path = Pipeline_Utils.save_metrics_summary(
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
                agg_csv = os.path.join(self.config.get("paths", {}).get("Error_Metrics", self.config.get("paths", {}).get("Prediction_Data", ".")), "ErrorMetrics_all_runs.csv")
                logging.info(f"📊 ErrorMetrics (aggregiert, CSV): {os.path.abspath(agg_csv)}")
            except Exception:
                pass
            logging.info("✅ Finale Ergebnisse erfolgreich gespeichert.")
        except Exception as e:
            logging.error(f"Fehler beim Speichern der finalen Ergebnisse: {e}", exc_info=True)


    def _run_inference_unified(self, input_data: np.ndarray):
        start = time.perf_counter()
        if hasattr(self.model, "get_input_details"):
            interpreter = self.model
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            if tuple(input_details[0]["shape"]) != tuple(input_data.shape):
                interpreter.resize_tensor_input(input_details[0]["index"], input_data.shape, strict=False)
                interpreter.allocate_tensors()
            interpreter.set_tensor(input_details[0]["index"], input_data.astype(np.float32))
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]["index"])
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
        pass

    @abstractmethod
    def _prepare_input_data(self, payload: dict) -> tuple[np.ndarray | None, any, float | None]:
        raise NotImplementedError

    @abstractmethod
    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        raise NotImplementedError