# LSTM/lstm_inference.py
import time
import pandas as pd
import numpy as np
import threading
import logging
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
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline3D # Für die Typ-Hinweise
from ML_Helpfunctions.base_inference import BaseInferenceProcessor


from config.config_ml_lstm import param_lstm_test
from config.config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FOLDER_FLAG = "LSTM"


class LSTMInference(BaseInferenceProcessor):
    """Spezialisierte Inferenzklasse für LSTM."""
    
    def __init__(self, config: dict, broker_ip: str, port: int, topic: str, folder_flag:str):
        super().__init__(config, broker_ip, port, topic, folder_flag)
        self.lags = config.get('lags', 10)
        self.target_feature = config['base_features'][0]

    def _prepare_input_data(self):
        """Bereitet ein 3D-Fenster [1, lags, features] für das LSTM-Modell vor."""
        new_row = pd.DataFrame([self.latest_payload])
        new_row['datetime'] = pd.to_datetime(new_row['datetime'])
        new_row = new_row.set_index('datetime')
        
        self._live_data_buffer = pd.concat([self._live_data_buffer, new_row]).sort_index()

        max_history = self.config.get('max_fe_window', 50) + self.lags
        if len(self._live_data_buffer) > max_history:
            self._live_data_buffer = self._live_data_buffer.iloc[-max_history:]

        if len(self._live_data_buffer) < self.lags:
            logging.warning(f"Buffer filling for LSTM window... {len(self._live_data_buffer)}/{self.lags}. Skipping.")
            return None, None, None

        featured_buffer, _ = fe.add_all_features(self._live_data_buffer.copy(), self.config)
        
        if len(featured_buffer) < self.lags:
            return None, None, None

        window_df = featured_buffer[self.feature_list].iloc[-self.lags:]
        
        if window_df.isnull().values.any():
            logging.warning("NaNs in final inference window. Skipping.")
            return None, None, None

        window_scaled = self.scaler.transform(window_df.values)
        inference_window = np.expand_dims(window_scaled, axis=0)
        
        timestamp = window_df.index[-1]
        true_value = self.latest_payload.get(self.target_feature)
        
        return inference_window, timestamp, true_value


if __name__ == "__main__":
    logging.info("--- MODE: Standalone LSTM Inference (Console Output) ---")
    
    # --- Basiskonfiguration für den Test ---
    # Nutzt die Logik aus dem Haupt-Inferenzskript
    infer_config = param_lstm_test.copy()
    infer_config.update(CONFIG_LOAD_ARTIFACTS)
    infer_config['paths'] = CONFIG_PATH['paths']
    
    # Beispiel-Pfade für den 'fast' mode (müssen durch die Trainingsartefakte ersetzt werden)
    infer_config['model_path_static'] = "trained_lstm_model.keras" 
    infer_config['scaler_path_static'] = "trained_lstm_scaler.joblib"
    infer_config['features_path_static'] = "trained_lstm_features.joblib"


    infer_config["inference_mode"] = "load_artifacts_path" # "load_artifacts_fast" oder "load_artifacts_path"
    # infer_config["load_id"] = "2025-07-20_002945_8844" 

    # MQTT-Konfiguration
    mqtt_broker_ip = MQTT_CONFIG['MQTT_BROKER_IP']
    mqtt_port = MQTT_CONFIG['MQTT_PORT']
    mqtt_topic = MQTT_CONFIG['MQTT_TOPIC']
    
    # run_standalone_inference(infer_config, mqtt_broker_ip, mqtt_port, mqtt_topic)
    # --- Inferenz starten ---
    processor = LSTMInference(infer_config, mqtt_broker_ip, mqtt_port, mqtt_topic, FOLDER_FLAG)
    processor.run()