# rf_inference.py
import time
import pandas as pd
import threading
import logging
import argparse
import sys
import os
import argparse 

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

FOLDER_FLAG = "RandomForest"



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RFInference(BaseInferenceProcessor):
    """Spezialisierte Inferenzklasse für Random Forest."""

    def __init__(self, config: dict, broker_ip: str, port: int, topic: str, folder_flag:str):
        super().__init__(config, broker_ip, port, topic, folder_flag)
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


if __name__ == "__main__":
    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(description="Standalone Random Forest Inference")
    parser.add_argument("--load_id", type=str, help="Optional: The specific run ID to load artifacts from (e.g., 2025-07-21_103000_1234).")
    parser.add_argument("--model_filename", type=str, help="Optional: The specific model filename within the run folder (e.g., model.joblib).")
    args = parser.parse_args()
    
    logging.info("--- MODE: Standalone Inference (Console Output) ---")
    
    # --- Base Configuration ---
    infer_config = {**CONFIG_PATH, **CONFIG_LOAD_ARTIFACTS, **param_rf_test}

    # --- Overwrite config with command-line arguments if provided ---
    if args.load_id:
        infer_config['load_id'] = args.load_id
        infer_config['inference_mode'] = 'load_artifacts_path'  # Force path mode
        logging.info(f"Using command-line argument --load_id: {args.load_id}")
    
    if args.model_filename:
        infer_config['model_filename'] = args.model_filename
        logging.info(f"Using command-line argument --model_filename: {args.model_filename}")

    # Beispiel-Pfade für den 'fast' mode (müssen durch die Trainingsartefakte ersetzt werden)
    infer_config['model_path_static'] = "trained_rf_model.joblib" 
    infer_config['scaler_path_static'] = "trained_rf_scaler.joblib"
    infer_config['features_path_static'] = "trained_rf_features.joblib"
    
    # MQTT-Konfiguration
    mqtt_broker_ip = MQTT_CONFIG['MQTT_BROKER_IP']
    mqtt_port = MQTT_CONFIG['MQTT_PORT']
    mqtt_topic = MQTT_CONFIG['MQTT_TOPIC']
    
    processor = RFInference(infer_config, mqtt_broker_ip, mqtt_port, mqtt_topic, FOLDER_FLAG)
    processor.run()