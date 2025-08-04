import pandas as pd
import numpy as np
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
# *** NEU: Import des neuen, optimierten Datenprozessors ***
from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor
from config.config_ml_lstm import param_lstm_test
from config.config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FOLDER_FLAG = "LSTM"

class LSTMInference(BaseInferenceProcessor):
    """
    Spezialisierte Inferenzklasse für LSTM, die den optimierten RealTimeDataProcessor verwendet.
    """
    
    def __init__(self, config: dict, broker_ip: str, port: int, topic: str, folder_flag:str):
        super().__init__(config, broker_ip, port, topic, folder_flag)
        self.lags = config.get('lags', 10)
        self.target_feature = config['base_features'][0]
        
        # *** NEU: Instanz des Datenprozessors erstellen ***
        # Die gesamte Puffer-Logik wird an diese Klasse delegiert.
        self.data_processor = RealTimeDataProcessor(config)

    def _prepare_input_data(self):
        """
        Bereitet ein 3D-Fenster [1, lags, features] für das LSTM-Modell vor.
        Diese Methode ist jetzt deutlich schlanker und schneller.
        """
        if self.latest_payload is None:
            return None, None, None

        # 1. Daten an den Prozessor übergeben und Features berechnen lassen
        featured_buffer = self.data_processor.update_and_process(self.latest_payload)
        
        # Wenn der Prozessor nicht genügend Daten hat, gibt er None zurück
        if featured_buffer is None:
            return None, None, None

        # 2. Letztes Fenster für die Inferenz extrahieren
        # Sicherstellen, dass genügend Zeilen nach dem Feature Engineering übrig sind
        if len(featured_buffer) < self.lags:
            return None, None, None
            
        window_df = featured_buffer[self.feature_list].iloc[-self.lags:]
        
        # 3. Auf NaN-Werte prüfen (wichtig nach rollierenden Berechnungen)
        if window_df.isnull().values.any():
            logging.warning("NaNs im finalen Inferenz-Fenster entdeckt. Überspringe Schritt.")
            return None, None, None

        # 4. Daten skalieren und in das 3D-Format für das LSTM bringen
        window_scaled = self.scaler.transform(window_df.values)
        inference_window = np.expand_dims(window_scaled, axis=0)
        
        # Metadaten für das Logging und die Speicherung extrahieren
        timestamp = window_df.index[-1]
        # Standardisiere die Schlüssel des rohen Payloads, um den true_value sicher zu finden
        payload_lower = {k.lower(): v for k, v in self.latest_payload.items()}
        true_value = payload_lower.get(self.target_feature)
        
        return inference_window, timestamp, true_value

# Der __main__-Teil bleibt unverändert, da er nur zum Testen dient
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone LSTM Inference")
    parser.add_argument("--load_id", type=str, help="Optional: The specific run ID to load artifacts from.")
    parser.add_argument("--model_filename", type=str, help="Optional: The specific model filename.")
    args = parser.parse_args()

    logging.info("--- MODE: Standalone LSTM Inference (Console Output) ---")
    
    infer_config = param_lstm_test.copy()
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
    
    processor = LSTMInference(infer_config, mqtt_broker_ip, mqtt_port, mqtt_topic, FOLDER_FLAG)
    processor.run()
