# rf_inference.py

import pandas as pd
import numpy as np
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
from ML_Helpfunctions.base_inference import BaseInferenceProcessor
from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor

FOLDER_FLAG = "RandomForest"

class RFInference(BaseInferenceProcessor):
    """
    Spezialisierte Inferenzklasse für Random Forest, die den optimierten RealTimeDataProcessor verwendet.
    """

    # --- KORREKTUR 1: __init__ an die neue, schlanke Form angepasst ---
    def __init__(self, config: dict, folder_flag: str = FOLDER_FLAG):
        super().__init__(config, folder_flag)
        self.target_feature = config['base_features'][0]
        self.data_processor = RealTimeDataProcessor(config)

# In rf_inference.py

    def _prepare_input_data(self, payload: dict) -> tuple[np.ndarray | None, any, float | None]:
        """
        Bereitet einen 2D-Feature-Vektor für das RF-Modell vor.
        Diese Version ist robust gegenüber Groß- und Kleinschreibung im Payload.
        """
        if not payload:
            return None, None, None

        # --- NEU: Robuste, Case-Insensitive Behandlung des Payloads ---
        # Erstelle eine temporäre Version des Payloads, bei der alle Schlüssel klein geschrieben sind.
        try:
            payload_lower = {str(k).lower(): v for k, v in payload.items()}
        except AttributeError:
            logging.error("Fehler beim Konvertieren der Payload-Schlüssel in Kleinbuchstaben. Ist das Payload ein Dictionary?")
            return None, None, None
            
        # Der Datenprozessor erhält das Payload mit den nun garantierten Kleinbuchstaben-Schlüsseln.
        # Wichtig: Der data_processor muss so konfiguriert sein, dass er ebenfalls mit Kleinbuchstaben-Features arbeitet.
        featured_buffer = self.data_processor.update_and_process(payload_lower)
        
        if featured_buffer is None or featured_buffer.empty:
            return None, None, None

        # Letzten Vektor extrahieren
        last_vector_full = featured_buffer[self.feature_list].iloc[-1:]

        if last_vector_full.isnull().values.any():
            logging.warning("NaNs im finalen Inferenz-Vektor entdeckt. Überspringe Schritt.")
            return None, None, None

        # Skalieren
        X_live_scaled = self.scaler.transform(last_vector_full.values) if self.scaler else last_vector_full.values
        
        timestamp = last_vector_full.index[-1]
        
        # --- KORREKTUR: Suche in dem Dictionary mit den Kleinbuchstaben-Schlüsseln ---
        key_to_find = self.target_feature.lower()
        true_value = payload_lower.get(key_to_find)

        # Der Debug-Logger bleibt für den Fall, dass die Spalte komplett fehlt
        if true_value is None:
            logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.warning(f"FEHLER: 'true_value' konnte auch nach Umwandlung in Kleinbuchstaben nicht gefunden werden!")
            logging.warning(f"--> Gesuchter Schlüssel: '{key_to_find}'")
            available_keys = list(payload_lower.keys())
            logging.warning(f"--> Verfügbare Schlüssel (klein, Auszug): {available_keys[:10]}")
            logging.warning("--> Bitte prüfen: Ist die Spalte in der CSV/MQTT-Quelle überhaupt vorhanden?")
            logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        return X_live_scaled, timestamp, true_value
    # --- KORREKTUR 2: Fehlende abstrakte Methode implementiert ---
    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        """
        Für Random Forest sind die Vorhersagen bereits im korrekten, unskalierten Raum.
        Daher geben wir die Vorhersage einfach unverändert zurück.
        """
        return np.asarray(prediction_scaled).flatten()

# Der __main__-Block kann für Standalone-Tests bleiben, wird aber von der pipeline_web_app nicht genutzt.
if __name__ == "__main__":
    from config.config_ml_random_forest import random_forest
    from config.config_general import CONFIG_PATH, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS
    import argparse 

    parser = argparse.ArgumentParser(description="Standalone Random Forest Inference")
    parser.add_argument("--load_id", type=str, help="Optional: The specific run ID to load artifacts from.")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- MODE: Standalone RF Inference (Console Output) ---")
    
    infer_config = random_forest.copy()
    infer_config.update(CONFIG_LOAD_ARTIFACTS)
    infer_config['paths'] = CONFIG_PATH['paths']
    
    if args.load_id:
        infer_config['load_id'] = args.load_id
        infer_config['inference_mode'] = 'load_artifacts_path'
    
    # Erstellen der Instanz ohne MQTT-Parameter, da diese aus der Config kommen
    processor = RFInference(config=infer_config)
    
    # Die .run() Methode muss noch an die neue Iterator-Logik angepasst werden,
    # aber für den Web-App-Kontext ist das nicht notwendig.
    logging.info("Standalone-Ausführung beendet. Für den Pipeline-Betrieb ist dies korrekt.")