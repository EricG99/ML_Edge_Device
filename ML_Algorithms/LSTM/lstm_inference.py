# ML_Algorithms/LSTM/LSTM_inference.py

import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
from ML_Helpfunctions.base_inference import BaseInferenceProcessor
from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor

# Dedicated logger
logger = logging.getLogger("LSTMInference")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(_h)
logger.setLevel(getattr(logging, os.environ.get("LOGLEVEL", "INFO").upper(), logging.INFO))

FOLDER_FLAG = "LSTM"


class LSTMInference(BaseInferenceProcessor):
    """Spezialisierte Inferenzklasse für LSTM."""

    def __init__(self, config: dict, folder_flag: str = FOLDER_FLAG):
        # Base-Konstruktor initialisiert bereits data_processor & lags
        super().__init__(config, folder_flag)
        self.lags = int(self.config.get("lags", 1))

    def _post_load_artifacts(self):
        """
        Wird nach dem Laden der Artefakte aufgerufen.
        Selektiver Merge: FE-kritisches aus training_config übernehmen,
        Runtime-Overrides (z. B. loading_strategy, model_filename, MQTT) beibehalten.
        Danach RealTimeDataProcessor mit finaler Config neu aufsetzen.
        """
        # --- 1) Runtime-Overrides sichern ---
        preserve_keys = (
            "loading_strategy", "inference_interval_sec",
            "mqtt_host", "mqtt_port", "mqtt_topic", "mqtt_username", "mqtt_password",
            "model_filename"
        )
        preserved = {k: self.config.get(k) for k in preserve_keys if k in self.config}

        # --- 2) FE-kritische Keys aus training_config übernehmen ---
        fe_keys = (
            "lags", "horizon",
            "rolling_window_size", "rolling_windows",
            "max_fe_window", "base_features", "derived_features"
        )
        if getattr(self, "training_config", None):
            for k in fe_keys:
                if k in self.training_config:
                    self.config[k] = self.training_config[k]

        # --- 3) Runtime-Overrides wieder auflegen ---
        self.config.update({k: v for k, v in preserved.items() if v is not None})
        self.lags = int(self.config.get("lags", 1))

        # --- 4) DataProcessor neu aufsetzen & vorhandenen Buffer übernehmen ---
        old_buf = getattr(getattr(self, "data_processor", None), "_buffer", None)
        self.data_processor = RealTimeDataProcessor(self.config)
        if old_buf is not None:
            try:
                if not getattr(old_buf, "empty", True):
                    self.data_processor._buffer = old_buf.tail(
                        getattr(self.data_processor, "_max_buffer_size", len(old_buf))
                    )
            except Exception as e:
                logger.warning("Konnte alten Buffer nicht übernehmen: %s", e)

        # --- 5) Klartext-Logging Datenquelle ---
        logger.info("Datenquelle nach Merge: %s", self.config.get("loading_strategy"))

        # --- 6) Optional: TFLite laden, wenn gefordert ---
        model_name = str(self.config.get("model_filename", ""))
        if model_name.endswith(".tflite"):
            try:
                models_dir = self.config.get("paths", {}).get("Models")
                tfl_path = os.path.join(models_dir, model_name) if models_dir else model_name
                interpreter = tf.lite.Interpreter(model_path=tfl_path)
                interpreter.allocate_tensors()
                self.model = interpreter
                logger.info("TFLite-Interpreter aktiv (%s).", model_name)
            except Exception as e:
                logger.warning("TFLite nicht geladen (%s) – fallback auf Keras.", e)
        else:
            logger.info("Keras-Modell aktiv (%s).", model_name or "model.keras")

    def _on_artifacts_swapped(self):
        """
        Wird nach set_artifacts_from_memory() aufgerufen.
        Stellt sicher, dass RealTimeDataProcessor mit neuer Config synchron ist
        und der Puffer-Zustand erhalten bleibt (Hot-Swap).
        """
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
                logger.info("%s: DataProcessor warm-started mit %d Zeilen.",
                            self.__class__.__name__, len(self.data_processor._buffer))
            else:
                logger.info("%s: DataProcessor neu initialisiert (kein alter Puffer).",
                            self.__class__.__name__)
        except Exception as e:
            logger.warning("%s: Konnte alten Puffer nicht übernehmen: %s",
                           self.__class__.__name__, e)

    def _prepare_input_data(self, payload: dict):
        """Bereitet aus Live-Payload ein (1, lags, features)-Fenster für die Inferenz auf."""
        if not payload:
            return None, None, None
        try:
            payload_lower = {str(k).lower(): v for k, v in payload.items()}
        except AttributeError:
            logger.error("LSTM: Fehler beim Normalisieren der Payload-Schlüssel.")
            return None, None, None

        featured_buffer = self.data_processor.update_and_process(payload_lower)
        if featured_buffer is None or len(featured_buffer) < self.lags:
            return None, None, None

        # Prüfen, ob alle Trainings-Features vorhanden sind
        missing = [col for col in self.feature_list if col not in featured_buffer.columns]
        if missing:
            logger.warning(
                "Warte: %d Feature-Spalten fehlen noch (z.B. %s). "
                "Puffer hat %d Zeilen. Inferenz-Schritt wird übersprungen.",
                len(missing), missing[:3], len(featured_buffer)
            )
            return None, None, None

        window_df = featured_buffer[self.feature_list].iloc[-self.lags:]
        if window_df.isnull().values.any():
            logger.warning("LSTM: NaNs im Inferenzfenster – Schritt übersprungen.")
            return None, None, None

        window_scaled = self.scaler.transform(window_df.values)
        inference_window = np.expand_dims(window_scaled, axis=0)  # (1, lags, features)

        timestamp = pd.to_datetime(payload_lower.get('datetime'))
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.utcnow()

        key_to_find = (self.target_feature or "").lower()
        true_value = payload_lower.get(key_to_find)
        if true_value is None:
            logger.warning("LSTM: Zielwert '%s' nicht im Payload gefunden.", key_to_find)

        return inference_window, timestamp, true_value

    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        if self.y_scaler is None:
            raise RuntimeError(
                "Der 'y_scaler' wurde nicht gefunden oder geladen. Eine Rücktransformation ist nicht möglich."
            )
        pred_reshaped = np.asarray(prediction_scaled).reshape(-1, 1)
        return self.y_scaler.inverse_transform(pred_reshaped).flatten()
