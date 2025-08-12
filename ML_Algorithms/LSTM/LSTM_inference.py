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
                
                # DEBUG-Logging für TFLite-Modell
                in_det = interpreter.get_input_details()[0]
                out_det = interpreter.get_output_details()[0]
                logger.info(f"TFLite Input: Shape={in_det['shape']}, DType={in_det['dtype']}")
                logger.info(f"TFLite Output: Shape={out_det['shape']}, DType={out_det['dtype']}")
                
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
        
                # --- DEBUG-Logging für Eingabedaten ---
        logger.debug(f"Input-Window Shape: {inference_window.shape}")
        logger.debug(f"Min/Max im skalierten Input: {np.min(inference_window):.4f} / {np.max(inference_window):.4f}")
        if np.allclose(np.min(inference_window), np.max(inference_window)):
            logger.warning("Alle Werte im skalierten Input-Fenster sind identisch!")

        
        return inference_window, timestamp, true_value

    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        """
        Skaliert die LSTM-Vorhersage ausschließlich mit dem dedizierten y_scaler zurück.
        *** KORRIGIERTE VERSION: Entfernt den fragilen Fallback-Mechanismus. ***
        """
        # Prüfen, ob der notwendige y_scaler vorhanden ist.
        if self.y_scaler is None:
            # Ein lauter Fehler ist besser als eine stille, falsche Vorhersage.
            raise RuntimeError(
                "Der 'y_scaler' wurde nicht gefunden oder geladen. Eine Rücktransformation "
                "der Vorhersage ist nicht möglich. Stellen Sie sicher, dass das Modell "
                "mit einem y_scaler gespeichert wurde."
            )

        # Die Vorhersage in die korrekte Form bringen (z.B. von (1, 5) zu (5, 1))
        pred_reshaped = np.asarray(prediction_scaled).reshape(-1, 1)

        # Die inverse Transformation mit dem dedizierten und sicheren y_scaler durchführen
        # .flatten() wandelt das Ergebnis wieder in ein 1D-Array um (z.B. [pred1, pred2, ...])
        return self.y_scaler.inverse_transform(pred_reshaped).flatten()