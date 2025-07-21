# xgboost_inference.py
import logging
import sys
import os
import pandas as pd

# --- Suppress scikit-learn feature name warnings ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
from ML_Helpfunctions import Feature_Engeneering as fe
from ML_Helpfunctions.base_inference import BaseInferenceProcessor

# --- Configuration Imports ---
# from config.config_ml_xgboost import param_xgb_test
from config.config_general import CONFIG_PATH, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS
from config.config_ml_xgboost import param_xgb_test

# Eindeutiger Ordner-Name, um die Artefakte zu finden
FOLDER_FLAG = "XGBoost"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class XGBoostInference(BaseInferenceProcessor):
    """Spezialisierte Inferenzklasse für XGBoost."""

    def __init__(self, config: dict, broker_ip: str, port: int, topic: str, folder_flag:str):
        super().__init__(config, broker_ip, port, topic, folder_flag)
        self.target_feature = config['base_features'][0]

    def _prepare_input_data(self):
        """
        Bereitet einen 2D-Feature-Vektor für das XGBoost-Modell vor.
        Diese Logik ist identisch mit der für RandomForest.
        """
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
            logging.warning(f"Buffer füllt sich... {len(self._live_data_buffer)}/{min_buffer_size}. Überspringe.")
            return None, None, None

        # Feature Engineering auf dem Live-Buffer
        featured_buffer, _ = fe.add_all_features(self._live_data_buffer.copy(), self.config)
        
        # Prüfen auf NaNs nach dem Feature Engineering
        if featured_buffer.empty or featured_buffer[self.feature_list].iloc[-1].isnull().values.any():
            logging.warning("NaNs im finalen Feature-Vektor entdeckt. Überspringe Inferenz.")
            return None, None, None

        # Letzten Vektor extrahieren und mit geladenem Scaler transformieren
        last_vector_full = featured_buffer[self.feature_list].iloc[-1:]
        X_live_scaled = self.scaler.transform(last_vector_full.values)
        
        timestamp = last_vector_full.index[-1]
        true_value = self.latest_payload.get(self.target_feature)
        
        return X_live_scaled, timestamp, true_value

if __name__ == "__main__":
    logging.info("--- MODE: Standalone XGBoost Inference (Console Output) ---")
    


    # --- Basiskonfiguration für die Inferenz ---
    infer_config = {**CONFIG_PATH, **CONFIG_LOAD_ARTIFACTS, **param_xgb_test}

    # Statische Pfade für 'fast' mode
    infer_config['model_path_static'] = "trained_xgb_model.joblib" 
    infer_config['scaler_path_static'] = "trained_xgb_scaler.joblib"
    infer_config['features_path_static'] = "trained_xgb_features.joblib"
    
    # Wichtige Schlüssel für Inferenz
    # 'load_id' wird benötigt, wenn inference_mode = 'load_artifacts_path'
    # infer_config['load_id'] = "2025-07-21_..." 
    infer_config["inference_mode"] = "load_artifacts_path"
    
    # MQTT-Konfiguration
    mqtt_broker_ip = MQTT_CONFIG['MQTT_BROKER_IP']
    mqtt_port = MQTT_CONFIG['MQTT_PORT']
    mqtt_topic = MQTT_CONFIG['MQTT_TOPIC']
    
    # --- Inferenz starten ---
    processor = XGBoostInference(infer_config, mqtt_broker_ip, mqtt_port, mqtt_topic, FOLDER_FLAG)
    processor.run()