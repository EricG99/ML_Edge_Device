# ML_Algorithms/Light_XGBOOST/Light_XGBOOST_inference.py
import os, sys, logging, numpy as np, pandas as pd
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ML_Helpfunctions.base_inference import BaseInferenceProcessor  # type: ignore
from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor  # type: ignore

logger = logging.getLogger("LightGBMInference")
if not logger.handlers:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(h)
logger.setLevel(os.environ.get("LOGLEVEL", "INFO").upper())

FOLDER_FLAG = "Light_XGBOOST"

class LightXGBoostInference(BaseInferenceProcessor):
    """Inferenzprozessor für LightGBM (tabular, 2D)."""
    def __init__(self, config: dict, folder_flag: str = FOLDER_FLAG):
        super().__init__(config, folder_flag)
        self.target_feature = config['base_features'][0]
        self.data_processor = RealTimeDataProcessor(config)

    def _post_load_artifacts(self):
        """
        Wird nach dem initialen Laden der Artefakte aufgerufen.
        Mergt die Trainings-Konfiguration mit der Laufzeit-Konfiguration.
        """
        # 1. Wichtige Laufzeit-Parameter sichern
        preserve_keys = ("loading_strategy", "inference_interval_sec", "model_filename")
        preserved_config = {k: self.config.get(k) for k in preserve_keys if k in self.config}

        # 2. FE-kritische Keys aus der geladenen training_config übernehmen
        if getattr(self, "training_config", None):
            self.config.update(self.training_config)

        # 3. Gesicherte Laufzeit-Parameter wieder anwenden (sie haben Vorrang)
        self.config.update(preserved_config)
        logger.info("Konfiguration nach dem Laden der Artefakte erfolgreich gemergt.")

        # 4. DataProcessor mit der finalen Konfiguration neu aufsetzen und Buffer erhalten
        old_buf = getattr(getattr(self, "data_processor", None), "_buffer", None)
        self.data_processor = RealTimeDataProcessor(self.config)
        if old_buf is not None and not old_buf.empty:
            max_len = getattr(self.data_processor, "_max_buffer_size", len(old_buf))
            self.data_processor._buffer = old_buf.tail(max_len)
            logger.info("DataProcessor wurde mit %d Zeilen aus dem alten Puffer warm-gestartet.", len(self.data_processor._buffer))

    # ANGEPASST: Vereinheitlichte Logik für den "Hot-Swap" von Modellen.
    def _on_artifacts_swapped(self):
        """
        Wird nach einem Hot-Swap (set_artifacts_from_memory) aufgerufen.
        Stellt sicher, dass der data_processor synchronisiert ist und der Puffer erhalten bleibt.
        """
        old_buf = None
        try:
            if getattr(self, "data_processor", None) is not None and hasattr(self.data_processor, "_buffer"):
                old_buf = self.data_processor._buffer.copy()
        except Exception:
            old_buf = None

        self.data_processor = RealTimeDataProcessor(self.config)
        try:
            if old_buf is not None and not old_buf.empty:
                max_len = getattr(self.data_processor, "_max_buffer_size", len(old_buf))
                self.data_processor._buffer = old_buf.tail(max_len)
                logger.info("Hot-Swap: DataProcessor warm-started mit %d Zeilen.", len(self.data_processor._buffer))
        except Exception as e:
            logger.warning("Hot-Swap: Alter Puffer konnte nicht übernommen werden: %s", e)

    def _run_inference_unified(self, input_data):
        import time
        t0 = time.perf_counter()
        y_hat = self.model.predict(input_data)
        t_ms = (time.perf_counter() - t0) * 1000.0
        y_hat = np.asarray(y_hat)
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(1, -1)
        elif y_hat.ndim > 2:
            y_hat = y_hat.reshape(1, -1)
        return y_hat, t_ms

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

        last_vector_full = featured_buffer[self.feature_list].iloc[-1:]
        if last_vector_full.isnull().values.any():
            logging.warning("Light_XGBoostInference: NaNs im finalen Inferenz-Vektor entdeckt. Überspringe Schritt.")
            return None, None, None

        X_live_scaled = self.scaler.transform(last_vector_full.values) if self.scaler else last_vector_full.values

        timestamp = last_vector_full.index[-1]
        key_to_find = self.target_feature.lower()
        true_value = payload_lower.get(key_to_find)
        if true_value is None:
            logging.warning("Light_XGBoostInference: true_value ('%s') nicht im Payload gefunden.", key_to_find)

        return X_live_scaled, timestamp, true_value

    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        # analog XGBoost: Rückskalierung optional (hier: Identität)
        return np.asarray(prediction_scaled).flatten()
