
# CNN1D Inference

import pandas as pd
import numpy as np
import logging
import sys
import os
import tensorflow as tf

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ML_Helpfunctions.base_inference import BaseInferenceProcessor
from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor

logger = logging.getLogger("CNN1DInference")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(_h)
level_name = os.environ.get("LOGLEVEL", "INFO").upper()
logger.setLevel(getattr(logging, level_name, logging.INFO))

FOLDER_FLAG = "CNN1D"

class CNN1DInference(BaseInferenceProcessor):
    """Spezialisierte Inferenzklasse für 1D‑CNN Modelle."""
    def __init__(self, config: dict, folder_flag: str = FOLDER_FLAG):
        super().__init__(config, folder_flag)
        self.data_processor = RealTimeDataProcessor(config)
        self.lags = int(config.get("lags", 1))

    def _post_load_artifacts(self):
        """
        Ladepolitik (vereinheitlicht und robust):
        - Wenn --model_filename gesetzt ist:
            * Falls Pfad auf .tflite zeigt -> TFLite-Interpreter laden.
            * Sonst: Standard (Keras/SK) beibehalten.
        - Wenn KEIN model_filename und edge_device/enable_edge = True:
            * Versuche model_quant_float16.tflite zu laden.
        - Andernfalls: nichts tun (bereits geladenes Keras-Modell bleibt aktiv).
        """
        try:
            # Pfad zum Modellordner aus der Konfiguration holen
            models_dir = self.config.get("paths", {}).get("Models") or self.config.get("Models") or "."
            # Explizit übergebenen Modellnamen aus der Konfiguration holen
            explicit_filename = self.config.get("model_filename")

            # Priorität 1: Ein expliziter Modellname wurde übergeben
            if explicit_filename:
                chosen_path = explicit_filename if os.path.isabs(explicit_filename) else os.path.join(models_dir, explicit_filename)
                
                # Wenn es sich um eine TFLite-Datei handelt, lade den Interpreter
                if chosen_path.lower().endswith(".tflite") and os.path.exists(chosen_path):
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=chosen_path)
                    interpreter.allocate_tensors()
                    self.model = interpreter # WICHTIG: Überschreibe das Standard-Keras-Modell
                    logging.info(f"📦 CNN1D: Explizites TFLite-Modell geladen: {chosen_path}")
                else:
                    logging.info(f"📦 CNN1D: Explizites Modell ({explicit_filename}) ist kein TFLite-Modell, verwende Standard-Ladepfad.")
                return

            # Priorität 2 (Fallback): Wenn kein Modellname gegeben wurde, aber der Edge-Flag aktiv ist
            edge_flag = bool(self.config.get("edge_device", False) or self.config.get("enable_edge", False))
            if edge_flag:
                candidate_path = os.path.join(models_dir, "model_quant_float16.tflite")
                if os.path.exists(candidate_path):
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=candidate_path)
                    interpreter.allocate_tensors()
                    self.model = interpreter # WICHTIG: Überschreibe das Standard-Keras-Modell
                    logging.info(f"⚡ CNN1D: Edge-Flag aktiv, TFLite Float16 geladen: {candidate_path}")
                else:
                    logging.warning("⚠️ CNN1D: Edge-Flag aktiv, aber model_quant_float16.tflite nicht gefunden – nutze Standardmodell.")
        
        except Exception as e:
            logging.warning(f"⚠️ Fehler in _post_load_artifacts: Ladepolitik konnte nicht angewendet werden ({e}). Nutze Standardmodell.")

    def _on_artifacts_swapped(self):
        """Puffer‑Warmstart nach Hot‑Swap (analog LSTM/RF)."""
        from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor
        old_buf = None
        try:
            old_dp = getattr(self, "data_processor", None)
            if old_dp is not None and hasattr(old_dp, "_buffer"):
                old_buf = old_dp._buffer.copy()
        except Exception:
            old_buf = None

        self.data_processor = RealTimeDataProcessor(self.config)
        try:
            if old_buf is not None and not old_buf.empty:
                max_len = getattr(self.data_processor, "_max_buffer_size", len(old_buf))
                self.data_processor._buffer = old_buf.tail(max_len)
                logging.info("CNN1D: DataProcessor warm-started mit %d Zeilen.", len(self.data_processor._buffer))
            else:
                logging.info("CNN1D: DataProcessor neu initialisiert (kein alter Puffer verfügbar).")
        except Exception as e:
            logging.warning("CNN1D: Konnte alten Puffer nicht übernehmen: %s", e)

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
            raise RuntimeError("CNN1D: y_scaler fehlt – inverse Transformation nicht möglich.")
        pred_reshaped = np.asarray(prediction_scaled).reshape(-1, 1)
        return self.y_scaler.inverse_transform(pred_reshaped).flatten()
