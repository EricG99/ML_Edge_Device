# ML_Algorithms/SVM/svm_inference.py
import os, sys, logging
import numpy as np
import pandas as pd
from ML_Helpfunctions.base_inference import BaseInferenceProcessor
from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor

logger = logging.getLogger("SVMInference")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

FOLDER_FLAG = "SVM"

class SVMInference(BaseInferenceProcessor):
    """Inference for SVM/LinearSVR using 2D features (X_t -> y(t+1..H))."""

    def __init__(self, config: dict, folder_flag: str = FOLDER_FLAG):
        super().__init__(config, folder_flag)
        self.lags = int(self.config.get("lags", 1))

    def _post_load_artifacts(self):
        """
        Nach Laden der Trainings-Artefakte: training_config in self.config mergen,
        Laufzeit-Keys erhalten und DataProcessor neu initialisieren (Buffer übernehmen).
        """
        preserve_keys = ("loading_strategy", "inference_interval_sec", "model_filename", "inference_steps")
        preserved = {k: self.config.get(k) for k in preserve_keys if k in self.config}

        if getattr(self, "training_config", None):
            self.config.update(self.training_config)

        self.config.update(preserved)

        # DataProcessor warm-start
        old_buf = getattr(getattr(self, "data_processor", None), "_buffer", None)
        self.data_processor = RealTimeDataProcessor(self.config)
        if old_buf is not None:
            try:
                if not getattr(old_buf, "empty", True):
                    max_len = getattr(self.data_processor, "_max_buffer_size", len(old_buf))
                    self.data_processor._buffer = old_buf.tail(max_len)
                    logger.info("SVMInference: DataProcessor warm-started mit %d Zeilen.", len(self.data_processor._buffer))
            except Exception as e:
                logger.warning("SVMInference: Konnte alten Buffer nicht übernehmen: %s", e)

    def _prepare_input_data(self, payload: dict):
        if not payload:
            return None, None, None
        try:
            payload_lower = {str(k).lower(): v for k, v in payload.items()}
        except Exception:
            return None, None, None

        featured_buffer = self.data_processor.update_and_process(payload_lower)
        if featured_buffer is None or featured_buffer.empty:
            return None, None, None

        missing = [c for c in self.feature_list if c not in featured_buffer.columns]
        if missing:
            logger.debug("Warte: fehlende Feature-Spalten %s", missing[:3])
            return None, None, None

        X_row = featured_buffer[self.feature_list].iloc[-1:]
        if X_row.isnull().values.any():
            return None, None, None

        X_in = X_row.values
        if self.scaler is not None:
            try:
                X_in = self.scaler.transform(X_in)
            except Exception as e:
                logger.warning("Scaler transform failed: %s", e)

        ts = pd.to_datetime(payload_lower.get("datetime"))
        if pd.isna(ts):
            ts = pd.Timestamp.utcnow()

        key_to_find = (self.target_feature or "").lower()
        true_value = payload_lower.get(key_to_find)

        return X_in, ts, true_value

    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        pred = np.asarray(prediction_scaled)

        # Auf 2D (1, H) normalisieren – egal, ob (H,), (1,H) oder (H,1) reinkommt
        if pred.ndim == 1:
            pred = pred.reshape(1, -1)
        elif pred.ndim == 2 and pred.shape[0] != 1 and pred.shape[1] == 1:
            pred = pred.reshape(1, -1)

        if getattr(self, "y_scaler", None) is None:
            return pred.ravel()

        return self.y_scaler.inverse_transform(pred).ravel()

