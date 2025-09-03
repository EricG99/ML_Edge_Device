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
        if getattr(self, "y_scaler", None) is None:
            return np.asarray(prediction_scaled).reshape(-1)
        arr = np.asarray(prediction_scaled).reshape(-1, 1)
        return self.y_scaler.inverse_transform(arr).flatten()
