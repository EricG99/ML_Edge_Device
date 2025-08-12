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
        
        # --- KORREKTUR: Schrittzähler für die Log-Ausgabe ---
        self.step_counter = 0

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
        self.model = shared_model_dict["model"]
        self.scaler = shared_model_dict["scaler"]
        self.y_scaler = shared_model_dict.get("y_scaler")
        self.feature_list = shared_model_dict["features"]
        self.config = shared_model_dict["config"]
        self._post_load_artifacts()
        logging.info("✅ Artefakte aus dem Speicher übernommen.")

    def process_step(self, payload: dict) -> dict | None:
        """
        Führt einen kompletten Inferenzschritt für ein gegebenes Payload aus.
        Gibt ein Dictionary mit den Ergebnissen oder None zurück.
        """
        if not payload:
            return None

        input_data, timestamp, true_value = self._prepare_input_data(payload)
        if input_data is None:
            return None

        prediction_scaled, model_inference_time_ms = self._run_inference_unified(input_data)
        predictions_unscaled = self._inverse_transform_prediction(prediction_scaled)

        total_processing_time_ms = (time.perf_counter() - payload.get('start_time', time.perf_counter())) * 1000.0
        
        prediction_entry = {
            "datetime": timestamp,
            "prediction": float(predictions_unscaled[0]) if predictions_unscaled.size > 0 else None,
            "true_value": None if true_value is None else float(true_value),
            "rolling_forecast": predictions_unscaled.tolist(),
            "cpu_load": Pipeline_Utils.get_cpu_usage(),
            "ram_usage": Pipeline_Utils.get_memory_usage(),
            "model_inference_time_ms": float(model_inference_time_ms),
            "total_processing_time_ms": float(total_processing_time_ms)
        }

        # --- KORREKTUR: Gewünschte Log-Ausgabe für die Konsole ---
        self.step_counter += 1
        true_str = f"{prediction_entry['true_value']:.4f}" if prediction_entry['true_value'] is not None else "N/A"
        pred_str = f"{prediction_entry['prediction']:.4f}" if prediction_entry['prediction'] is not None else "N/A"
        cpu_str = f"{prediction_entry['cpu_load']:.1f}%" if prediction_entry['cpu_load'] is not None else "N/A"

        logging.info(f"Step [{self.step_counter}] -> True: {true_str} | Pred: {pred_str} | CPU: {cpu_str}")
        # --- Ende der Log-Ausgabe ---

        return prediction_entry

    # ... (Rest der Datei bleibt unverändert) ...

    def get_data_source_iterator(self):
        """Gibt einen Iterator zurück, der die Payloads für die Inferenz liefert."""
        strategy = self.config.get("loading_strategy", "split")
        if strategy == "split":
            return self._batch_iterator
        elif strategy == "live_mqtt":
            self._start_mqtt_client()
            return self._mqtt_iterator
        else:
            raise ValueError(f"Unbekannte Ladestrategie: {strategy}")

    def stop(self):
        """Beendet Hintergrundprozesse wie den MQTT-Client."""
        if self._mqtt_client:
            self._mqtt_client.stop()
            logging.info("MQTT-Client gestoppt.")
            
    def save_final_results(self, all_predictions: list):
        """Speichert die gesammelten Vorhersagen und Metriken in CSV-Dateien."""
        if not all_predictions:
            logging.warning("Keine Vorhersagen zum Speichern vorhanden.")
            return

        logging.info(f"Speichere {len(all_predictions)} Vorhersagen...")
        try:
            df = pd.DataFrame(all_predictions)
            horizon = self.config.get("horizon", 1)
            y_pred = np.stack(df["rolling_forecast"].dropna().to_numpy())
            
            y_true_1d = df["true_value"].ffill().bfill().to_numpy()
            y_true = np.tile(y_true_1d.reshape(-1, 1), reps=(1, y_pred.shape[1]))

            metrics = Pipeline_Utils.evaluate_all_metrics(y_true, y_pred, horizon=horizon)
            Pipeline_Utils.save_metrics_summary(metrics, self.config, self.training_config or {}, self.config.get("paths", {}))
            
            logging.info("✅ Finale Ergebnisse erfolgreich gespeichert.")
        except Exception as e:
            logging.error(f"Fehler beim Speichern der finalen Ergebnisse: {e}", exc_info=True)

    def _run_inference_unified(self, input_data: np.ndarray):
        start = time.perf_counter()
        # TFLite-Interpreter?
        if hasattr(self.model, "get_input_details"):
            interpreter = self.model
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # ==================== WICHTIGE KORREKTUR HIER ====================
            # Prüfe, ob die Eingabeform angepasst werden muss (z.B. von None auf 1)
            if tuple(input_details[0]["shape"]) != tuple(input_data.shape):
                logging.info(f"Passe TFLite-Eingabe an: von {input_details[0]['shape']} zu {input_data.shape}")
                interpreter.resize_tensor_input(input_details[0]["index"], input_data.shape, strict=False)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details() # Details neu abrufen
                output_details = interpreter.get_output_details()
            # ==================== ENDE DER KORREKTUR =======================

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
                if self.latest_payload:
                    payload = self.latest_payload
                    self.latest_payload = None
                else:
                    payload = None
            if payload:
                payload['start_time'] = time.perf_counter()
                yield payload
                count += 1
            else:
                time.sleep(0.05)

    def _batch_iterator(self, max_steps):
        df = Load_Prepare_Data.load_test_data_by_fraction(
            config=self.config,
            train_fraction=self.config.get("train_fraction", 0.7),
            make_date_as_index=False
        )
        df.columns = df.columns.str.lower()
        df = df.sort_values("datetime")
        count = 0
        for _, row in df.iterrows():
            if count >= max_steps:
                break
            payload = row.to_dict()
            payload['datetime'] = pd.to_datetime(payload['datetime'])
            payload['start_time'] = time.perf_counter()
            yield payload
            count += 1
            
    def _post_load_artifacts(self):
        pass

    @abstractmethod
    def _prepare_input_data(self, payload: dict) -> tuple[np.ndarray | None, any, float | None]:
        raise NotImplementedError

    @abstractmethod
    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        raise NotImplementedError