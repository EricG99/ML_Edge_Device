# rf_inference.py
import time
import pandas as pd
import threading
import logging
import argparse
import sys
import os

# --- Suppress scikit-learn feature name warnings ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
from ML_Helpfunctions import Feature_Engeneering as fe
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient
from ML_Helpfunctions.base_inference import BaseInferenceProcessor


from config.config_ml_rf import param_rf_test
from config.config_general import CONFIG_PATH, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RFInference(BaseInferenceProcessor):
    """Spezialisierte Inferenzklasse für Random Forest."""

    def __init__(self, config: dict, broker_ip: str, port: int, topic: str):
        super().__init__(config, broker_ip, port, topic)
        self.target_feature = config['base_features'][0]

    def _prepare_input_data(self):
        """Bereitet einen 2D-Feature-Vektor für das RF-Modell vor."""
        # Neuen Datenpunkt zum Puffer hinzufügen
        new_row = pd.DataFrame([self.latest_payload])
        new_row['datetime'] = pd.to_datetime(new_row['datetime'])
        new_row = new_row.set_index('datetime')
        
        self._live_data_buffer = pd.concat([self._live_data_buffer, new_row]).sort_index()
        
        # Buffer-Größe begrenzen
        max_history = self.config.get('max_fe_window', 50)
        if len(self._live_data_buffer) > max_history:
            self._live_data_buffer = self._live_data_buffer.iloc[-max_history:]

        # Prüfen, ob genügend Daten für Feature Engineering vorhanden sind
        min_buffer_size = max(self.config.get('lags', 1), self.config.get('rolling_window_size', 1)) + 1
        if len(self._live_data_buffer) < min_buffer_size:
            return None, None, None

        # Feature Engineering
        featured_buffer, _ = fe.add_all_features(self._live_data_buffer.copy(), self.config)
        
        if featured_buffer.empty or featured_buffer[self.feature_list].iloc[-1].isnull().values.any():
            logging.warning("NaNs detected in the latest feature vector. Skipping inference.")
            return None, None, None

        # Letzten Vektor extrahieren und skalieren
        last_vector_full = featured_buffer[self.feature_list].iloc[-1:]
        X_live_scaled = self.scaler.transform(last_vector_full.values)
        
        timestamp = last_vector_full.index[-1]
        true_value = self.latest_payload.get(self.target_feature)
        
        return X_live_scaled, timestamp, true_value

# class LiveInferenceProcessor:
#     """
#     Handles live data processing and timed inference.
#     It uses an internal buffer and runs predictions at a fixed interval.
#     """
#     def __init__(self, model, scaler, feature_list, config, on_prediction_callback=None):
#         self.model = model
#         self.scaler = scaler
#         self.feature_list = feature_list
#         self.config = config
#         self.on_prediction_callback = on_prediction_callback
#         self._live_data_buffer = pd.DataFrame()
#         self.latest_payload = None
#         self.lock = threading.Lock()
#         self.is_running = False

#         self.target_feature = config['base_features'][0]
#         self.inference_interval_sec = config.get("inference_interval_sec", 1.0)

#     def update_latest_data(self, data: dict):
#         """Callback to receive the latest MQTT message."""
#         with self.lock:
#             self.latest_payload = data

#     def _prepare_live_vector(self):
#         """Prepares a feature vector from the internal data buffer."""
#         if self.latest_payload is None:
#             return None, None

#         new_row = pd.DataFrame([self.latest_payload])
#         new_row['datetime'] = pd.to_datetime(new_row['datetime'])
#         new_row = new_row.set_index('datetime')
        
#         self._live_data_buffer = pd.concat([self._live_data_buffer, new_row]).sort_index()
#         max_history = self.config.get('max_fe_window', 50) + self.config.get('lags', 5)
#         if len(self._live_data_buffer) > max_history:
#             self._live_data_buffer = self._live_data_buffer.iloc[-max_history:]

#         # --- HIER IST DIE KORREKTUR ---
#         # Prüfen, ob genügend Daten für Lags und rollierende Fenster vorhanden sind
#         # Wir nehmen die größte aus der Konfig bekannte Fenstergröße als Minimum
#         min_buffer_size = max(self.config.get('lags', 1), self.config.get('rolling_window_size', 1)) + 1
#         if len(self._live_data_buffer) < min_buffer_size:
#             logging.warning(
#                 f"Buffer füllt sich... {len(self._live_data_buffer)}/{min_buffer_size} Zeilen vorhanden. Überspringe Inferenz."
#             )
#             return None, None # Wichtig: Inferenz für diesen Zyklus überspringen

#         # Erst jetzt das Feature Engineering aufrufen
#         featured_buffer, _ = fe.add_all_features(self._live_data_buffer.copy(), self.config)
#         # --- ENDE DER KORREKTUR ---

#         if featured_buffer.empty or featured_buffer[self.feature_list].isnull().values.any():
#             return None, None

#         last_vector_full = featured_buffer[self.feature_list].iloc[-1:]
#         X_live_2D = self.scaler.transform(last_vector_full.values)
        
#         return X_live_2D, last_vector_full.index[-1]

#     def run_inference_loop(self):
#         """The core timed loop for running inference."""
#         self.is_running = True
#         while self.is_running:
#             start_time = time.time()
            
#             with self.lock:
#                 if self.latest_payload is None:
#                     time.sleep(self.inference_interval_sec)
#                     continue
                
#                 inference_vector, timestamp_obj = self._prepare_live_vector()

#             if inference_vector is not None:
#                 prediction, inference_duration_ms = PipelineUtils.run_timed_inference(
#                     model=self.model, input_data=inference_vector
#                 )
                
#                 # If a callback is provided (for the web app), use it.
#                 if self.on_prediction_callback:
#                     output_data = self._create_output_data(prediction, timestamp_obj, inference_duration_ms)
#                     self.on_prediction_callback(output_data)
#                 else:
#                     # Otherwise, just log to console.
#                     logging.info(f"Prediction at {timestamp_obj.isoformat()}: {prediction[0].tolist()}")

#             elapsed_time = time.time() - start_time
#             sleep_time = max(0, self.inference_interval_sec - elapsed_time)
#             time.sleep(sleep_time)
#         logging.info("Inference loop stopped.")
        
#     def _create_output_data(self, prediction, timestamp_obj, inference_duration_ms):
#         """Helper to structure output data for the web app."""
#         cpu_load = PipelineUtils.get_cpu_usage()
#         current_value = self.latest_payload.get(self.target_feature)
#         predictions_list = prediction[0].tolist()

#         future_dates = [
#             (timestamp_obj + pd.Timedelta(seconds=(i + 1) * self.inference_interval_sec)).isoformat()
#             for i in range(len(predictions_list))
#         ]
        
#         return {
#             "date": timestamp_obj.isoformat(),
#             "true_value": current_value,
#             "predicted_value_step_1": predictions_list[0],
#             "predicted_value_step_n": predictions_list[-1],
#             "future_forecast": {"dates": future_dates, "values": predictions_list},
#             "cpu_load": cpu_load,
#             "inference_time_ms": inference_duration_ms
#         }
    
#     def stop(self):
#       self.is_running = False

# def run_standalone_inference(config, broker_ip, port, topic):
#     """
#     Manages the MQTT client and the inference loop for standalone execution.
#     """
#     try:
#         scaler, features, model = PipelineUtils.load_model_artifacts_for_inference(config)
#     except FileNotFoundError:
#         sys.exit(1)
        
#     processor = LiveInferenceProcessor(model, scaler, features, config)
    
#     # Start the timed inference loop in a separate thread
#     inference_loop_thread = threading.Thread(target=processor.run_inference_loop, daemon=True)
#     inference_loop_thread.start()

#     # Setup and run MQTT client to receive data
#     mqtt_client = MqttInferenceClient(
#         broker_ip=broker_ip, port=port, topic=topic,
#         on_message_callback=processor.update_latest_data
#     )
#     logging.info("Starting MQTT client...")
#     mqtt_client.run() # This will block until the client is stopped

#     try:
#         while True:
#             time.sleep(1)
#     except KeyboardInterrupt:
#         logging.info("Shutting down standalone inference...")
#         processor.stop()
#         mqtt_client.client.loop_stop()
#         mqtt_client.client.disconnect()
#         inference_loop_thread.join()
#         logging.info("Shutdown complete.")

if __name__ == "__main__":
    logging.info("--- MODE: Standalone Inference (Console Output) ---")
    
    # --- Base Configuration ---
    infer_config = {**CONFIG_PATH, **CONFIG_LOAD_ARTIFACTS, **param_rf_test}

    # Beispiel-Pfade für den 'fast' mode (müssen durch die Trainingsartefakte ersetzt werden)
    infer_config['model_path_static'] = "trained_rf_model.joblib" 
    infer_config['scaler_path_static'] = "trained_rf_scaler.joblib"
    infer_config['features_path_static'] = "trained_rf_features.joblib"
    
    # MQTT-Konfiguration
    mqtt_broker_ip = MQTT_CONFIG['MQTT_BROKER_IP']
    mqtt_port = MQTT_CONFIG['MQTT_PORT']
    mqtt_topic = MQTT_CONFIG['MQTT_TOPIC']
    
    #run_standalone_inference(infer_config, mqtt_broker_ip, mqtt_port, mqtt_topic)

    processor = RFInference(infer_config, mqtt_broker_ip, mqtt_port, mqtt_topic)
    processor.run()