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
        # Optional: TFLite bevorzugen, wenn vorhanden (analog zur LSTM-Implementierung)
        try:
            models_dir = self.config["paths"].get("Models")
            tfl_name = self.config.get("model_filename", "model_quant_float16.tflite")
            tfl_path = os.path.join(models_dir, tfl_name)

            if os.path.exists(tfl_path) and tfl_name.endswith('.tflite'):
                interpreter = tf.lite.Interpreter(model_path=tfl_path)
                
                # --- KORRIGIERTE VERSION START ---
                # Den Interpreter EINMALIG nach dem Laden vorbereiten.
                # Dies verhindert die XNNPack-Fehler in der Inferenzschleife.
                interpreter.allocate_tensors()
                # --- KORRIGIERTE VERSION ENDE ---

                in_det = interpreter.get_input_details()[0]
                out_det = interpreter.get_output_details()[0]
                logger.info(f"CNN1D TFLite Input: Shape={in_det['shape']}, DType={in_det['dtype']}")
                logger.info(f"CNN1D TFLite Output: Shape={out_det['shape']}, DType={out_det['dtype']}")
                self.model = interpreter
                logging.info(f"CNN1D-Inferenz nutzt TFLite-Interpreter: {tfl_path}")
        except Exception as e:
            logging.warning(f"CNN1D: TFLite-Interpreter nicht geladen ({e}), nutze Keras-Modell.")

    # --- PATCH START ---
    # Patch: Robuste Methode zum Übernehmen des Puffers beim Hot-Swap
    def _on_artifacts_swapped(self):
        """
        Wird nach set_artifacts_from_memory() aufgerufen.
        Stellt sicher, dass der RealTimeDataProcessor mit der neuen Konfiguration
        synchron ist und der Puffer-Zustand für eine nahtlose Inferenz erhalten bleibt.
        """
        from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor
        old_buf = None
        try:
            # Puffer aus dem alten Prozessor sicher auslesen
            old_dp = getattr(self, "data_processor", None)
            if old_dp is not None and hasattr(old_dp, "_buffer"):
                old_buf = old_dp._buffer.copy()
        except Exception:
            old_buf = None # Im Fehlerfall mit leerem Puffer starten

        # DataProcessor mit der (potenziell neuen) Konfiguration neu initialisieren
        self.data_processor = RealTimeDataProcessor(self.config)

        # Warm-Start mit altem Puffer
        try:
            if old_buf is not None and not old_buf.empty:
                max_len = getattr(self.data_processor, "_max_buffer_size", len(old_buf))
                self.data_processor._buffer = old_buf.tail(max_len)
                logging.info(f"{self.__class__.__name__}: DataProcessor warm-started mit {len(self.data_processor._buffer)} Zeilen aus altem Puffer.")
            else:
                logging.info(f"{self.__class__.__name__}: DataProcessor neu initialisiert (kein alter Puffer verfügbar).")
        except Exception as e:
            logging.warning(f"{self.__class__.__name__}: Konnte alten Puffer nicht übernehmen: {e}")
    # --- PATCH END ---


    def _prepare_input_data(self, payload: dict) -> tuple[np.ndarray | None, any, float | None]:
        if not payload:
            return None, None, None
        try:
            payload_lower = {str(k).lower(): v for k, v in payload.items()}
        except AttributeError:
            logging.error("CNN1D: Fehler beim Normalisieren der Payload-Schlüssel.")
            return None, None, None

        featured_buffer = self.data_processor.update_and_process(payload_lower)
        if featured_buffer is None or len(featured_buffer) < self.lags:
            return None, None, None

        # --- KORRIGIERTE VERSION START ---
        # Graceful Skip: Prüfen, ob alle vom Training bekannten Features auch
        # im live generierten Puffer vorhanden sind.
        missing_features = [col for col in self.feature_list if col not in featured_buffer.columns]
        if missing_features:
            logger.warning(
                f"Warte: {len(missing_features)} Feature-Spalten fehlen noch (z.B. {missing_features[:3]}). "
                f"Puffer hat {len(featured_buffer)} Zeilen. Inferenz-Schritt wird übersprungen."
            )
            return None, None, None  # Signalisiert, dass der Schritt übersprungen werden soll
        
        window_df = featured_buffer[self.feature_list].iloc[-self.lags:]
        # --- KORRIGIERTE VERSION ENDE ---
        
        if window_df.isnull().values.any():
            logging.warning("CNN1D: NaNs im Inferenzfenster – Schritt übersprungen.")
            return None, None, None

        window_scaled = self.scaler.transform(window_df.values)
        inference_window = np.expand_dims(window_scaled, axis=0)  # (1, lags, features)

        timestamp = pd.to_datetime(payload_lower.get('datetime'))
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.utcnow()

        key_to_find = self.target_feature.lower()
        true_value = payload_lower.get(key_to_find)
        if true_value is None:
            logging.warning(f"CNN1D: Zielwert '{key_to_find}' nicht im Payload gefunden.")

        return inference_window, timestamp, true_value

    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        if self.y_scaler is None:
            raise RuntimeError(
                "Der 'y_scaler' wurde nicht gefunden oder geladen. Eine Rücktransformation ist nicht möglich."
            )
        pred_reshaped = np.asarray(prediction_scaled).reshape(-1, 1)
        return self.y_scaler.inverse_transform(pred_reshaped).flatten()
    
