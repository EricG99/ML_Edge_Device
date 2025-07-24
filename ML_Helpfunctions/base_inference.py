# ML_Helpfunctions/base_inference.py
import logging
import sys
import os
import threading
import time
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

# GEÄNDERT: Zusätzliche Importe für Batch-Verarbeitung und saubere Struktur
from ML_Helpfunctions import Pipeline_Utils
from ML_Helpfunctions import Load_Prepare_Data 
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient


class BaseInferenceProcessor(ABC):
    """
    Abstrakte Basisklasse für Inferenz-Pipelines.
    GEÄNDERT: Unterstützt jetzt sowohl Live-Inferenz (MQTT) als auch Batch-Inferenz (CSV).
    """
    def __init__(self, config: dict, broker_ip: str, port: int, topic: str, folder_flag:str):
        self.config = config
        self.broker_ip = broker_ip
        self.port = port
        self.topic = topic
        self.folder_flag = folder_flag
        
        self.training_config = None
        self.model, self.scaler, self.feature_list = None, None, None
        self.is_running = False
        self.lock = threading.Lock()
        
        self._live_data_buffer = pd.DataFrame()
        self.latest_payload = None
        
        self.results_buffer = []
        self.inference_steps = self.config.get("inference_steps", "infinite")
        self.step_counter = 0
        
        if isinstance(self.inference_steps, int):
            logging.info(f"Inference will run for a maximum of {self.inference_steps} steps.")
        else:
            logging.info("Inference will run indefinitely until stopped manually.")

    def load_artifacts(self):
        """Lädt die für die Inferenz benötigten Artefakte."""
        try:
            logging.info("Loading inference artifacts...")
            self.scaler, self.feature_list, self.model, self.training_config = \
                Pipeline_Utils.load_model_artifacts_for_inference(self.config, self.folder_flag )
            # Die Ziel-Variable wird aus der Konfiguration geholt
            self.target_feature = self.config.get('base_features', [None])[0]
            logging.info("✅ Artifacts loaded successfully.")
        except (FileNotFoundError, ValueError) as e:
            logging.error(f"Fatal error loading artifacts: {e}")
            sys.exit(1)

    @abstractmethod
    def _prepare_input_data(self):
        """Muss von der Subklasse implementiert werden."""
        pass

    def update_latest_data(self, data: dict):
        """Callback, um die neueste Nachricht zu erhalten."""
        with self.lock:
            self.latest_payload = data

    # NEU: Diese Methode kapselt die Logik für eine einzelne Vorhersage
    def _run_single_prediction(self):
        """
        Führt eine einzelne Vorhersage aus, transformiert das Ergebnis zurück und speichert es.
        Gibt True zurück, wenn die Inferenz gestoppt werden soll.
        """
        if self.latest_payload is None:
            return False

        input_data, timestamp, true_value = self._prepare_input_data()

        if input_data is not None:
            prediction_scaled, inference_time_ms = Pipeline_Utils.run_timed_inference(
                model=self.model, input_data=input_data
            )
            cpu_load = Pipeline_Utils.get_cpu_usage()
            
            try:
                target_index = self.feature_list.index(self.target_feature)
            except (ValueError, AttributeError):
                target_index = 0

            prediction_unscaled = Pipeline_Utils.safe_inverse_transform(
                scaler=self.scaler, array=prediction_scaled.reshape(1, -1), target_index=target_index
            )
            prediction_list = prediction_unscaled.flatten().tolist()

            self.step_counter += 1
            logging.info(f"Step {self.step_counter}: Prediction for {timestamp.isoformat()}: {[f'{p:.2f}' for p in prediction_list]}")

            result_entry = { "datetime": timestamp, "true_value": true_value, "cpu_load_percent": cpu_load, "inference_time_ms": inference_time_ms }
            for i, pred_val in enumerate(prediction_list):
                result_entry[f"prediction_step_{i+1}"] = pred_val
            self.results_buffer.append(result_entry)
            
            if isinstance(self.inference_steps, int) and self.step_counter >= self.inference_steps:
                logging.info(f"--- ✅ Reached configured inference limit of {self.inference_steps} steps. ---")
                return True # Signal zum Stoppen
        return False # Signal zum Weitermachen

    # GEÄNDERT: Die Logik der alten `_run_inference_loop` wurde hierher verschoben und leicht angepasst.
    def _live_inference_loop(self):
        """Die Kernschleife, die periodisch die Live-Inferenz ausführt."""
        self.is_running = True
        logging.info("🚀 Starting live inference loop...")
        while self.is_running:
            start_time = time.time()
            with self.lock:
                should_stop = self._run_single_prediction()
            if should_stop:
                self.is_running = False
            elapsed = time.time() - start_time
            sleep_for = max(0, self.config.get("inference_interval_sec", 1.0) - elapsed)
            time.sleep(sleep_for)
        logging.info("Live inference loop has stopped.")

    def _run_batch_inference(self):
        """Führt eine Batch-Inferenz auf dem Test-Split der CSV-Datei aus."""
        logging.info("--- 🚀 Running BATCH Inference on Test-Split from CSV ---")
        
        # KORREKTUR 1: Lade die Daten, aber lasse 'datetime' als normale Spalte.
        test_df = Load_Prepare_Data.load_test_data_by_fraction(
            config=self.config,
            train_fraction=self.config.get("train_fraction", 0.75),
            make_date_as_index=False 
        )

        if test_df.empty:
            logging.warning("Test DataFrame is empty. Nothing to process.")
            return

        # KORREKTUR 2: Iteriere durch die Zeilen und erstelle ein sauberes Dictionary.
        # 'to_dict("records")' ist perfekt dafür geeignet.
        for payload_dict in test_df.to_dict("records"):
            # Der Payload enthält jetzt garantiert einen 'datetime'-Schlüssel.
            self.latest_payload = payload_dict
            
            should_stop = self._run_single_prediction()
            if should_stop:
                break
        logging.info("✅ Batch inference complete.")

    # NEU: Eigene Methode, die den MQTT-Client und den Thread startet
    def _run_live_mode(self):
        """Startet den gesamten Inferenzprozess für den Live-Modus."""
        inference_thread = threading.Thread(target=self._live_inference_loop, daemon=True)
        mqtt_client = MqttInferenceClient(
            broker_ip=self.broker_ip, port=self.port, topic=self.topic,
            on_message_callback=self.update_latest_data
        )
        try:
            logging.info("🚀 Starting MQTT client for LIVE inference...")
            mqtt_client.run()
            inference_thread.start()
            while inference_thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logging.info("\n🚨 Shutdown signal (KeyboardInterrupt) received.")
        finally:
            self.is_running = False
            if inference_thread.is_alive():
                inference_thread.join()
            mqtt_client.client.loop_stop()
            mqtt_client.client.disconnect()
            logging.info("MQTT client stopped.")

    def _save_results(self):
        """Speichert die gesammelten Ergebnisse am Ende des Laufs."""
        if not self.results_buffer:
            logging.warning("No results were collected, nothing to save.")
            return

        logging.info(f"\n💾 Saving {len(self.results_buffer)} collected inference results...")
        self.config, paths = Pipeline_Utils.setup_experiment(self.config, self.folder_flag, run_type='inference')
        results_df = pd.DataFrame(self.results_buffer)

        pred_filename = f"inference_results_{self.config.get('loading_strategy')}_{self.config['run_id']}.csv"
        pred_output_path = os.path.join(paths.get("Prediction_Data"), pred_filename)
        results_df.to_csv(pred_output_path, index=False, date_format='%Y-%m-%dT%H:%M:%S.%f')
        logging.info(f"✅ Detailed prediction results saved to: {pred_output_path}")

        if "true_value" in results_df.columns and "prediction_step_1" in results_df.columns:
            y_true = results_df["true_value"].to_numpy(); y_pred = results_df["prediction_step_1"].to_numpy()
            valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
            metrics = Pipeline_Utils.evaluate_all_metrics(y_true=y_true[valid_indices], y_pred=y_pred[valid_indices]) if np.sum(valid_indices) > 0 else {}
            Pipeline_Utils.save_metrics_summary(metrics=metrics, infer_config=self.config, train_config=self.training_config, paths=paths)
        else:
            logging.warning("Could not compute metrics: 'true_value' or 'prediction_step_1' not in results.")

    def run(self):
        """
        Startet den gesamten Inferenzprozess basierend auf der Lade-Strategie
        aus der Konfiguration.
        """

        # Die ursprüngliche `run` Methode wird durch diesen Dispatcher ersetzt.
        # Der Code wird nicht gelöscht, sondern in die neuen Methoden refaktorisiert.
        self.load_artifacts()
        strategy = self.config.get("loading_strategy")
        logging.info(f"Executing inference with strategy: '{strategy}'")

        if strategy == "split":
            self._run_batch_inference()
        elif strategy == "live_mqtt":
            self._run_live_mode()
        else:
            logging.error(f"Unknown loading_strategy: '{strategy}'. Aborting.")
            return
        
        # Das Speichern der Ergebnisse erfolgt nach Abschluss des jeweiligen Modus.
        self._save_results()
        logging.info("✅ Inference task complete.")