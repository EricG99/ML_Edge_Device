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
    def __init__(self, config: dict, broker_ip: str, port: int, topic: str):
        self.config = config
        self.broker_ip = broker_ip
        self.port = port
        self.topic = topic
        
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
            self.scaler, self.feature_list, self.model = Pipeline_Utils.load_model_artifacts_for_inference(self.config)
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
                
                prediction_value = prediction[0] if isinstance(prediction, (list, np.ndarray)) else prediction
                prediction_list = prediction_value.tolist() if hasattr(prediction_value, 'tolist') else [prediction_value]

                # NEU: Schrittzähler erhöhen
                self.step_counter += 1
                
                logging.info(f"Step {self.step_counter}: Prediction at {timestamp.isoformat()}: {prediction_list} | Time: {inference_time_ms:.2f}ms | CPU: {cpu_load:.1f}%")

                self.results_buffer.append({
                    "datetime": timestamp,
                    "true_value": true_value,
                    "prediction_step_1": prediction_list[0],
                    "cpu_load_percent": cpu_load,
                    "inference_time_ms": inference_time_ms
                })
                
                # NEU: Prüfen, ob das Limit erreicht ist
                if isinstance(self.inference_steps, int) and self.step_counter >= self.inference_steps:
                    logging.info(f"--- ✅ Reached configured inference limit of {self.inference_steps} steps. Stopping loop. ---")
                    self.is_running = False # Dies beendet die Schleife

            elapsed = time.time() - start_time
            sleep_for = max(0, self.config.get("inference_interval_sec", 1.0) - elapsed)
            time.sleep(sleep_for)
            
        logging.info("Inference loop has stopped.")

    def _save_results(self):
        """Speichert die gesammelten Inferenz-Ergebnisse in einer CSV-Datei."""
        if not self.results_buffer:
            logging.warning("No results were collected, nothing to save.")
            return

        logging.info(f"\nSaving {len(self.results_buffer)} collected inference results...")
        
        self.config, paths = Pipeline_Utils.setup_experiment(self.config)
        output_dir = paths.get("Prediction_Data", "Output/Prediction_Data")
        
        try:
            results_df = pd.DataFrame(self.results_buffer)
            filename = f"inference_results_{self.config['run_id']}.csv"
            output_path = os.path.join(output_dir, filename)
            
            results_df.to_csv(output_path, index=False, date_format='%Y-%m-%dT%H:%M:%S.%f')
            logging.info(f"✅ Results successfully saved to: {output_path}")

        except Exception as e:
            logging.error(f"Failed to save inference results: {e}", exc_info=True)


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
            
            self._save_results()
            logging.info("✅ Shutdown complete.")