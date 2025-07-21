# ML_Helpfunctions/base_inference.py
import logging
import sys
import os
import threading
import time
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from ML_Helpfunctions import Pipeline_Utils
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient



class BaseInferenceProcessor(ABC):
    """
    Abstrakte Basisklasse für Inferenz-Pipelines.
    Kapselt die Logik zum Laden von Artefakten, MQTT-Kommunikation,
    Inferenzschleife und das Speichern der Ergebnisse.
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
        
        # Puffer für Ergebnisse
        self.results_buffer = []

        # NEU: Zähler und Limit für Inferenzschritte
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
            # ERWEITERT: Unpackt jetzt vier Werte
            self.scaler, self.feature_list, self.model, self.training_config = \
                Pipeline_Utils.load_model_artifacts_for_inference(self.config, self.folder_flag )
            logging.info("✅ Artifacts loaded successfully.")
        except (FileNotFoundError, ValueError) as e:
            logging.error(f"Fatal error loading artifacts: {e}")
            sys.exit(1)

    @abstractmethod
    def _prepare_input_data(self):
        """
        Muss von der Subklasse implementiert werden.
        Bereitet die Eingabedaten (Vektor/Tensor) für das spezifische Modell vor.
        Sollte (input_data, timestamp, current_true_value) zurückgeben.
        """
        pass

    def update_latest_data(self, data: dict):
        """Callback, um die neueste MQTT-Nachricht zu erhalten."""
        with self.lock:
            self.latest_payload = data
            
    def _run_inference_loop(self):
        """Die Kernschleife, die periodisch die Inferenz ausführt."""
        self.is_running = True
        logging.info("🚀 Starting inference loop...")
        
        while self.is_running:
            start_time = time.time()
            
            with self.lock:
                if self.latest_payload is None:
                    time.sleep(self.config.get("inference_interval_sec", 1.0))
                    continue
                
                input_data, timestamp, true_value = self._prepare_input_data()

            if input_data is not None:
                prediction, inference_time_ms = Pipeline_Utils.run_timed_inference(
                    model=self.model, input_data=input_data
                )
                cpu_load = Pipeline_Utils.get_cpu_usage()
                
                prediction_value = prediction[0] if isinstance(prediction, (list, np.ndarray)) and prediction.ndim > 1 else prediction
                prediction_list = prediction_value.tolist() if hasattr(prediction_value, 'tolist') else [prediction_value]

                self.step_counter += 1
                logging.info(f"Step {self.step_counter}: Prediction at {timestamp.isoformat()}: {prediction_list} | Time: {inference_time_ms:.2f}ms | CPU: {cpu_load:.1f}%")

                # KORREKTUR: Alle Vorhersageschritte dynamisch zum Ergebnis hinzufügen
                result_entry = {
                    "datetime": timestamp,
                    "true_value": true_value,
                    "cpu_load_percent": cpu_load,
                    "inference_time_ms": inference_time_ms
                }
                for i, pred_val in enumerate(prediction_list):
                    result_entry[f"prediction_step_{i+1}"] = pred_val
                
                self.results_buffer.append(result_entry)
                
                if isinstance(self.inference_steps, int) and self.step_counter >= self.inference_steps:
                    logging.info(f"--- ✅ Reached configured inference limit of {self.inference_steps} steps. Stopping loop. ---")
                    self.is_running = False

            elapsed = time.time() - start_time
            sleep_for = max(0, self.config.get("inference_interval_sec", 1.0) - elapsed)
            time.sleep(sleep_for)
            
        logging.info("Inference loop has stopped.")

    def _save_results(self):
        """
        Speichert die detaillierten Vorhersagen und separat eine Zusammenfassung
        mit Metriken und Konfigurationen.
        """
        if not self.results_buffer:
            logging.warning("No results were collected, nothing to save.")
            return

        logging.info(f"\nSaving {len(self.results_buffer)} collected inference results...")
        self.config, paths = Pipeline_Utils.setup_experiment(self.config, self.folder_flag, run_type='inference')
        
        results_df = pd.DataFrame(self.results_buffer)

        # --- Schritt 1: Detaillierte Vorhersage-Datei mit allen Spalten speichern ---
        try:
            pred_filename = f"inference_results_{self.config['run_id']}.csv"
            pred_output_path = os.path.join(paths.get("Prediction_Data"), pred_filename)
            
            # KORREKTUR: Speichere den gesamten DataFrame ohne Spaltenfilter
            results_df.to_csv(pred_output_path, index=False, date_format='%Y-%m-%dT%H:%M:%S.%f')
            
            logging.info(f"✅ Detailed prediction results saved to: {pred_output_path}")

        except Exception as e:
            logging.error(f"Failed to save detailed prediction results: {e}", exc_info=True)

        # --- Schritt 2: Gesamtmetriken für die Zusammenfassung berechnen ---
        # Die Metriken werden standardmäßig auf den ersten Schritt berechnet
        if "true_value" in results_df.columns and "prediction_step_1" in results_df.columns:
            y_true = results_df["true_value"].to_numpy()
            y_pred = results_df["prediction_step_1"].to_numpy()
            
            valid_indices = ~np.isnan(y_true) & ~np.isnan(y_pred)
            if np.sum(valid_indices) > 0:
                metrics = Pipeline_Utils.evaluate_all_metrics(
                    y_true=y_true[valid_indices],
                    y_pred=y_pred[valid_indices]
                )
                logging.info(f"Overall Inference Metrics for summary: {metrics}")
            else:
                metrics = {}
                logging.warning("Could not compute metrics for summary file (not enough valid data).")
        else:
            metrics = {}
            logging.warning("Skipping metrics calculation: 'true_value' or 'prediction_step_1' not in results.")


        # --- Schritt 3: Zentrale Zusammenfassungs-Datei speichern/aktualisieren ---
        try:
            Pipeline_Utils.save_metrics_summary(
                metrics=metrics,
                infer_config=self.config,
                train_config=self.training_config,
                paths=paths
            )
        except Exception as e:
            logging.error(f"Failed to save metrics summary file: {e}", exc_info=True)


    def run(self):
        """Startet den gesamten Inferenzprozess."""
        self.load_artifacts()
        
        inference_thread = threading.Thread(target=self._run_inference_loop, daemon=True)
        
        mqtt_client = MqttInferenceClient(
            broker_ip=self.broker_ip, port=self.port, topic=self.topic,
            on_message_callback=self.update_latest_data
        )
        
        try:
            logging.info("Starting MQTT client...")
            mqtt_client.run()
            inference_thread.start()
            
            # Die Schleife wartet entweder auf den Interrupt oder darauf,
            # dass der Inferenz-Thread self.is_running auf False setzt.
            while inference_thread.is_alive() and self.is_running:
                time.sleep(0.5)

        except KeyboardInterrupt:
            logging.info("\n🚨 Shutdown signal (KeyboardInterrupt) received.")
            self.is_running = False
        finally:
            logging.info("Stopping services...")
            if inference_thread.is_alive():
                inference_thread.join()
            
            mqtt_client.client.loop_stop()
            mqtt_client.client.disconnect()
            
            if not self.folder_flag:
                logging.warning("Kein 'folder_flag' übergeben, Ergebnisse können nicht gespeichert werden.")
            else:
                self._save_results()

            logging.info("✅ Shutdown complete.")