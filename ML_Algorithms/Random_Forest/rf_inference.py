# rf_inference.py

import pandas as pd
import logging
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
from ML_Helpfunctions.base_inference import BaseInferenceProcessor
# *** NEU: Import des optimierten Datenprozessors ***
from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor
from config.config_ml_random_forest import param_rf_test
from config.config_general import CONFIG_PATH, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS

FOLDER_FLAG = "RandomForest"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RFInference(BaseInferenceProcessor):
    """
    Spezialisierte Inferenzklasse für Random Forest, die den optimierten RealTimeDataProcessor verwendet.
    """

    def __init__(self, config: dict, broker_ip: str, port: int, topic: str, folder_flag:str):
        super().__init__(config, broker_ip, port, topic, folder_flag)
        self.target_feature = config['base_features'][0]
        
        # *** NEU: Instanz des Datenprozessors erstellen ***
        # Die gesamte Puffer- und Feature-Logik wird an diese Klasse delegiert.
        self.data_processor = RealTimeDataProcessor(config)

    def _prepare_input_data(self):
        """
        Bereitet einen 2D-Feature-Vektor für das RF-Modell vor, indem der RealTimeDataProcessor genutzt wird.
        """
        if self.latest_payload is None:
            return None, None, None

        featured_buffer = self.data_processor.update_and_process(self.latest_payload)
        
        if featured_buffer is None:
            return None, None, None

        if featured_buffer.empty or featured_buffer[self.feature_list].iloc[-1].isnull().values.any():
            logging.warning("NaNs im finalen Inferenz-Vektor entdeckt. Überspringe Schritt.")
            return None, None, None

        last_vector_full = featured_buffer[self.feature_list].iloc[-1:]
        X_live_scaled = self.scaler.transform(last_vector_full.values)
        
        timestamp = last_vector_full.index[-1]
        
        payload_lower = {k.lower(): v for k, v in self.latest_payload.items()}
        true_value = payload_lower.get(self.target_feature)

        # =================================================================
        # === BULLETPROOF DEBUG-LOGGER ===
        # Dieser Block wird nur ausgeführt, wenn der Wert nicht gefunden wurde.
        if true_value is None:
            logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.warning("FEHLER: 'true_value' konnte nicht gefunden werden!")
            logging.warning(f"--> Gesuchter Schlüssel (aus config): '{self.target_feature}'")
            logging.warning(f"--> Verfügbare Schlüssel (aus MQTT): {list(payload_lower.keys())}")
            logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # =================================================================
        
        return X_live_scaled, timestamp, true_value


if __name__ == "__main__":
    # --- Argument Parser Setup ---
    parser = argparse.ArgumentParser(description="Standalone Random Forest Inference")
    parser.add_argument("--load_id", type=str, help="Optional: The specific run ID to load artifacts from.")
    parser.add_argument("--model_filename", type=str, help="Optional: The specific model filename.")
    args = parser.parse_args()
    
    logging.info("--- MODE: Standalone RF Inference (Console Output) ---")
    
    infer_config = param_rf_test.copy()
    infer_config.update(CONFIG_LOAD_ARTIFACTS)
    infer_config['paths'] = CONFIG_PATH['paths']
    
    if args.load_id:
        infer_config['load_id'] = args.load_id
        infer_config['inference_mode'] = 'load_artifacts_path'
    if args.model_filename:
        infer_config['model_filename'] = args.model_filename

    mqtt_broker_ip = MQTT_CONFIG['MQTT_BROKER_IP']
    mqtt_port = MQTT_CONFIG['MQTT_PORT']
    mqtt_topic = MQTT_CONFIG['MQTT_TOPIC']
    
    processor = RFInference(infer_config, mqtt_broker_ip, mqtt_port, mqtt_topic, FOLDER_FLAG)
    processor.run()