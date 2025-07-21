# cnn1d_inference.py
import logging
import sys
import os
import pandas as pd
import numpy as np

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
from ML_Helpfunctions import Feature_Engeneering as fe
from ML_Helpfunctions.base_inference import BaseInferenceProcessor

# --- Configuration Imports ---
from config.config_general import CONFIG_PATH, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS

# Eindeutiger Ordner-Name, um die Artefakte zu finden
FOLDER_FLAG = "CNN1D"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CNN1DInference(BaseInferenceProcessor):
    """Spezialisierte Inferenzklasse für 1D-CNN."""
    
    def __init__(self, config: dict, broker_ip: str, port: int, topic: str, folder_flag:str):
        super().__init__(config, broker_ip, port, topic, folder_flag)
        self.lags = config.get('lags', 10)
        self.target_feature = config['base_features'][0]

    def _prepare_input_data(self):
        """
        Bereitet ein 3D-Fenster [1, lags, features] für das 1D-CNN-Modell vor.
        Die Logik ist identisch mit der für LSTM.
        """
        new_row = pd.DataFrame([self.latest_payload])
        new_row['datetime'] = pd.to_datetime(new_row['datetime'])
        new_row = new_row.set_index('datetime')
        
        self._live_data_buffer = pd.concat([self._live_data_buffer, new_row]).sort_index()

        max_history = self.config.get('max_fe_window', 50) + self.lags
        if len(self._live_data_buffer) > max_history:
            self._live_data_buffer = self._live_data_buffer.iloc[-max_history:]

        if len(self._live_data_buffer) < self.lags:
            logging.warning(f"Buffer füllt sich für CNN-Fenster... {len(self._live_data_buffer)}/{self.lags}. Überspringe.")
            return None, None, None

        featured_buffer, _ = fe.add_all_features(self._live_data_buffer.copy(), self.config)
        
        if len(featured_buffer) < self.lags:
            return None, None, None

        window_df = featured_buffer[self.feature_list].iloc[-self.lags:]
        
        if window_df.isnull().values.any():
            logging.warning("NaNs im finalen Inferenz-Fenster. Überspringe Inferenz.")
            return None, None, None

        window_scaled = self.scaler.transform(window_df.values)
        inference_window = np.expand_dims(window_scaled, axis=0) # Shape: (1, lags, features)
        
        timestamp = window_df.index[-1]
        true_value = self.latest_payload.get(self.target_feature)
        
        return inference_window, timestamp, true_value


if __name__ == "__main__":
    logging.info("--- MODE: Standalone 1D-CNN Inference (Console Output) ---")
    
    param_cnn1d_test = {
        'model_name': 'cnn1d_test',
        'lags': 10,
        'base_features': ['Group4-2_S6_MassFlowRate'],
    }

    # --- Basiskonfiguration für die Inferenz ---
    infer_config = {**CONFIG_PATH, **CONFIG_LOAD_ARTIFACTS, **param_cnn1d_test}

    # Statische Pfade für 'fast' mode
    infer_config['model_path_static'] = "trained_cnn_model.keras" 
    infer_config['scaler_path_static'] = "trained_cnn_scaler.joblib"
    infer_config['features_path_static'] = "trained_cnn_features.joblib"
    
    # Für 'path' mode, eine ID von einem Trainingslauf eintragen
    # infer_config['load_id'] = "IHRE_RUN_ID" 
    infer_config["inference_mode"] = "load_artifacts_path"
    
    # MQTT-Konfiguration
    mqtt_broker_ip = MQTT_CONFIG['MQTT_BROKER_IP']
    mqtt_port = MQTT_CONFIG['MQTT_PORT']
    mqtt_topic = MQTT_CONFIG['MQTT_TOPIC']
    
    # --- Inferenz starten ---
    processor = CNN1DInference(infer_config, mqtt_broker_ip, mqtt_port, mqtt_topic, FOLDER_FLAG)
    processor.run()