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
        self.data_processor = RealTimeDataProcessor(config) 
        self.lags = int(config.get("lags", 1))

    def _post_load_artifacts(self):
        if self.config.get("mode") == "retraining":
            logging.info("Retraining-Modus aktiv. Keras-Modell wird für die Inferenz verwendet.")
            return
        try:
            models_dir = self.config["paths"].get("Models")
            tfl_name = self.config.get("model_filename", "model_quant_float16.tflite")
            tfl_path = os.path.join(models_dir, tfl_name)
            if os.path.exists(tfl_path):
                interpreter = tf.lite.Interpreter(model_path=tfl_path)
                interpreter.allocate_tensors()
                
                in_det = interpreter.get_input_details()[0]
                out_det = interpreter.get_output_details()[0]
                logger.info(f"TFLite Input: Shape={in_det['shape']}, DType={in_det['dtype']}")
                logger.info(f"TFLite Output: Shape={out_det['shape']}, DType={out_det['dtype']}")
                
                self.model = interpreter
                logging.info(f"ℹ️ LSTM-Inferenz nutzt TFLite-Interpreter: {tfl_path}")
        except Exception as e:
            logging.warning(f"⚠️ TFLite-Interpreter konnte nicht geladen werden ({e}), nutze Keras-Modell.")

    def _on_artifacts_swapped(self):
        """
        Wird nach set_artifacts_from_memory() aufgerufen.
        LSTM nutzt im Retraining-Modus Keras anstelle von TFLite; der DataProcessor
        muss ggf. mit neuer Config/Lags/Horizon neu aufgebaut werden.
        """
        from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor
        self.data_processor = RealTimeDataProcessor(self.config)
        logging.info("LSTMInference: DataProcessor nach Hot-Swap neu initialisiert.")


    def _prepare_input_data(self, payload: dict) -> tuple[np.ndarray | None, any, float | None]:
        if not payload:
            return None, None, None
            
        try:
            payload_lower = {str(k).lower(): v for k, v in payload.items()}
        except AttributeError:
            logging.error("Fehler beim Konvertieren der Payload-Schlüssel in Kleinbuchstaben.")
            return None, None, None

        featured_buffer = self.data_processor.update_and_process(payload_lower)
        
        # FIX 5: Temporäre Diagnose zum Überprüfen der Zeitstempel
        logger.debug(
            f"t_payload={payload_lower.get('datetime')}, "
            f"t_buffer_last={self.data_processor._buffer.index[-1] if not self.data_processor._buffer.empty else 'N/A'}, "
            f"t_fe_last={featured_buffer.index[-1] if featured_buffer is not None and not featured_buffer.empty else 'N/A'}"
        )
        
        if featured_buffer is None or len(featured_buffer) < self.lags:
            return None, None, None
            
        window_df = featured_buffer[self.feature_list].iloc[-self.lags:]
        
        if window_df.isnull().values.any():
            logging.warning("NaNs im Inferenz-Fenster entdeckt. Überspringe Schritt.")
            return None, None, None
            
        window_scaled = self.scaler.transform(window_df.values)
        inference_window = np.expand_dims(window_scaled, axis=0)
        
        # FIX 1: Log-/Referenz-Zeit direkt aus dem Payload nehmen
        timestamp = pd.to_datetime(payload_lower.get('datetime'))
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.utcnow() # Fallback

        key_to_find = self.target_feature.lower()
        true_value = payload_lower.get(key_to_find)

        if true_value is None:
            logging.warning(f"FEHLER (LSTM): 'true_value' für Schlüssel '{key_to_find}' nicht im Payload gefunden.")

        logger.debug(f"Input-Window Shape: {inference_window.shape}")
        
        return inference_window, timestamp, true_value

    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        if self.y_scaler is None:
            raise RuntimeError(
                "Der 'y_scaler' wurde nicht gefunden oder geladen. Eine Rücktransformation ist nicht möglich."
            )
        pred_reshaped = np.asarray(prediction_scaled).reshape(-1, 1)
        return self.y_scaler.inverse_transform(pred_reshaped).flatten()
    
    