
# ML_Algorithms/XGBOOST/XGBOOST_inference.py
import os, sys, logging, numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ML_Helpfunctions.base_inference import BaseInferenceProcessor  # type: ignore
from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor  # type: ignore

FOLDER_FLAG = "XGBOOST"

class XGBoostInference(BaseInferenceProcessor):
    """Inferenzprozessor für XGBoost (tabular, 2D)."""
    def __init__(self, config: dict, folder_flag: str = FOLDER_FLAG):
        super().__init__(config, folder_flag)
        self.target_feature = config['base_features'][0]
        self.data_processor = RealTimeDataProcessor(config)

    def _on_artifacts_swapped(self):
        # Warm-Start: Puffer übernehmen, Pending neu seed-en
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
                logging.info("XGBoostInference: Warm-Start mit %d Zeilen.", len(self.data_processor._buffer))
        except Exception as e:
            logging.warning("XGBoostInference: Konnte alten Puffer nicht übernehmen: %s", e)

        try:
            if getattr(self, "_last_input_data", None) is None:
                return
            pred_scaled, t_inf_ms = self._run_inference_unified(self._last_input_data)
            pred_unscaled = self._inverse_transform_prediction(pred_scaled).reshape(-1)

            H = int(self.config.get("horizon", 1))
            if pred_unscaled.size < H:
                pad = np.full(H - pred_unscaled.size, np.nan, dtype=float)
                pred_unscaled = np.concatenate([pred_unscaled, pad], axis=0)
            elif pred_unscaled.size > H:
                pred_unscaled = pred_unscaled[:H]

            import pandas as pd
            dfb = getattr(self.data_processor, "_buffer", None)
            if dfb is not None and len(dfb) > 0:
                if isinstance(dfb.index, pd.DatetimeIndex):
                    last_ts = dfb.index[-1]
                elif "datetime" in dfb.columns:
                    last_ts = pd.to_datetime(dfb["datetime"].iloc[-1])
                else:
                    last_ts = pd.Timestamp.utcnow()
            else:
                last_ts = pd.Timestamp.utcnow()
            dt_next = last_ts + pd.Timedelta(seconds=float(self.config.get("inference_interval_sec", 1.0)))

            try:
                from ML_Helpfunctions.Pipeline_Utils import PipelineUtils  # type: ignore
                cpu = float(PipelineUtils.get_cpu_usage())
                ram = float(PipelineUtils.get_memory_usage())
            except Exception:
                cpu, ram = None, None

            self._pending_entry = {
                "datetime": dt_next,
                "prediction": float(pred_unscaled[0]) if np.isfinite(pred_unscaled[0]) else None,
                "true_value": None,
                "rolling_forecast": pred_unscaled.tolist(),
                "cpu_percent": cpu,
                "ram_mb": ram,
                "model_inference_time_ms": float(t_inf_ms),
                "total_processing_time_ms": 0.0,
            }
            logging.info("XGBoostInference: Pending nach Hot-Swap gesetzt.")
        except Exception as e:
            logging.warning("XGBoostInference: Reseed Pending fehlgeschlagen (%s).", e)

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
            logging.warning("NaNs im finalen Inferenz-Vektor entdeckt. Überspringe Schritt.")
            return None, None, None

        X_live_scaled = self.scaler.transform(last_vector_full.values) if self.scaler else last_vector_full.values

        timestamp = last_vector_full.index[-1]
        key_to_find = self.target_feature.lower()
        true_value = payload_lower.get(key_to_find)
        if true_value is None:
            logging.warning("XGBoostInference: true_value ('%s') nicht im Payload gefunden.", key_to_find)

        return X_live_scaled, timestamp, true_value

    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        return np.asarray(prediction_scaled).flatten()
