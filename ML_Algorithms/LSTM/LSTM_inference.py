# Kompletter Inhalt für Ihre LSTM_inference.py Datei

import pandas as pd
import numpy as np
import logging
import sys
import os
import tensorflow as tf

# --- Suppress scikit-learn feature name warnings ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
from ML_Helpfunctions.base_inference import BaseInferenceProcessor
from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor

# Dedicated logger for this module
logger = logging.getLogger("LSTMInference")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(_h)
level_name = os.environ.get("LOGLEVEL", "INFO").upper()
logger.setLevel(getattr(logging, level_name, logging.INFO))


FOLDER_FLAG = "LSTM"

class LSTMInference(BaseInferenceProcessor):
    """Spezialisierte Inferenzklasse für LSTM."""
    
    def __init__(self, config: dict, folder_flag: str = FOLDER_FLAG):
        super().__init__(config, folder_flag)
        self.data_processor = RealTimeDataProcessor(config) # Verwaltet den Puffer für Features
        self.lags = int(config.get("lags", 1))

    def _post_load_artifacts(self):
        """Versucht, nach dem Laden der Basis-Artefakte ein optimiertes TFLite-Modell zu laden."""
        try:
            models_dir = self.config["paths"].get("Models")
            tfl_name = self.config.get("model_filename", "model_quant_float16.tflite")
            tfl_path = os.path.join(models_dir, tfl_name)
            if os.path.exists(tfl_path):
                interpreter = tf.lite.Interpreter(model_path=tfl_path)
                interpreter.allocate_tensors()
                self.model = interpreter # Überschreibe das Keras-Modell
                logging.info(f"ℹ️ LSTM-Inferenz nutzt TFLite-Interpreter: {tfl_path}")
        except Exception as e:
            logging.warning(f"⚠️ TFLite-Interpreter konnte nicht geladen werden ({e}), nutze Keras-Modell.")

    def _prepare_input_data(self, payload: dict) -> tuple[np.ndarray | None, any, float | None]:
        """
        Bereitet ein 3D-Fenster für das LSTM-Modell vor.
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
        featured_buffer = self.data_processor.update_and_process(payload_lower)
        
        if featured_buffer is None or len(featured_buffer) < self.lags:
            return None, None, None
            
        window_df = featured_buffer[self.feature_list].iloc[-self.lags:]
        
        if window_df.isnull().values.any():
            logging.warning("NaNs im Inferenz-Fenster entdeckt. Überspringe Schritt.")
            return None, None, None
            
        window_scaled = self.scaler.transform(window_df.values)
        inference_window = np.expand_dims(window_scaled, axis=0)
        
        timestamp = window_df.index[-1]

        # --- KORREKTUR: Suche in dem Dictionary mit den Kleinbuchstaben-Schlüsseln ---
        key_to_find = self.target_feature.lower()
        true_value = payload_lower.get(key_to_find)

        # Hilfreicher Debug-Logger für den Fall, dass die Spalte komplett fehlt
        if true_value is None:
            logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.warning(f"FEHLER (LSTM): 'true_value' konnte auch nach Umwandlung in Kleinbuchstaben nicht gefunden werden!")
            logging.warning(f"--> Gesuchter Schlüssel: '{key_to_find}'")
            available_keys = list(payload_lower.keys())
            logging.warning(f"--> Verfügbare Schlüssel (klein, Auszug): {available_keys[:10]}")
            logging.warning("--> Bitte prüfen: Ist die Spalte in der CSV/MQTT-Quelle überhaupt vorhanden?")
            logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        return inference_window, timestamp, true_value

    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        """Skaliert die LSTM-Vorhersage mit dem y_scaler zurück."""
        if self.y_scaler:
            pred_reshaped = np.asarray(prediction_scaled).reshape(-1, 1)
            pred_unscaled_reshaped = self.y_scaler.inverse_transform(pred_reshaped)
            return pred_unscaled_reshaped.flatten()
        else:
            # Fallback, falls Target nicht skaliert wurde
            return np.asarray(prediction_scaled).flatten()