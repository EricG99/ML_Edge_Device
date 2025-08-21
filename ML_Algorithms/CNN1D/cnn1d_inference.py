# ML_Algorithms/CNN1D/cnn1d_inference.py

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

# --- Application Imports ---
from ML_Helpfunctions.base_inference import BaseInferenceProcessor
from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor

# Dedicated logger for this module
logger = logging.getLogger("CNN1DInference")
if not logger.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logger.addHandler(_h)
level_name = os.environ.get("LOGLEVEL", "INFO").upper()
logger.setLevel(getattr(logging, level_name, logging.INFO))


FOLDER_FLAG = "CNN1D"


class CNN1DInference(BaseInferenceProcessor):
    """Spezialisierte Inferenzklasse für 1D-CNN."""

    def __init__(self, config: dict, folder_flag: str = FOLDER_FLAG):
        super().__init__(config, folder_flag)
        # Vor-Init; wird nach dem Laden der Trainings-Config erneut korrekt initialisiert
        self.data_processor = RealTimeDataProcessor(config)
        self.lags = int(config.get("lags", 1))

    def _post_load_artifacts(self):
        """
        Nach dem Laden der Artefakte:
        - FE-relevante Parameter aus 'training_config.json' übernehmen,
          Runtime-Overrides (loading_strategy, inference_interval_sec, model_filename, MQTT...) erhalten.
        - RealTimeDataProcessor mit finaler Config neu initialisieren (inkl. Buffer-Warmstart).
        - Optional TFLite-Interpreter laden, wenn .tflite gewählt ist.
        """
        # 1) Runtime-Overrides sichern
        preserve_keys = (
            "loading_strategy", "inference_interval_sec",
            "mqtt_host", "mqtt_port", "mqtt_topic", "mqtt_username", "mqtt_password",
            "model_filename"
        )
        preserved = {k: self.config.get(k) for k in preserve_keys if k in self.config}

        # 2) Nur FE-kritische Keys aus Trainings-Config übernehmen
        fe_keys = (
            "lags", "horizon",
            "rolling_window_size", "rolling_windows",
            "max_fe_window", "base_features", "derived_features"
        )
        if getattr(self, "training_config", None):
            for k in fe_keys:
                if k in self.training_config:
                    self.config[k] = self.training_config[k]

        # 3) Runtime-Overrides wieder auflegen (haben Vorrang)
        self.config.update({k: v for k, v in preserved.items() if v is not None})

        # 4) RealTimeDataProcessor mit finaler Config neu aufsetzen & Buffer übernehmen
        old_buf = getattr(getattr(self, "data_processor", None), "_buffer", None)
        self.data_processor = RealTimeDataProcessor(self.config)
        if old_buf is not None and not getattr(old_buf, "empty", True):
            self.data_processor._buffer = old_buf.tail(
                getattr(self.data_processor, "_max_buffer_size", len(old_buf))
            )
        logging.info("Datenquelle nach Merge: %s", self.config.get("loading_strategy"))

        # 5) Nur TFLite laden, wenn wirklich .tflite verlangt ist (sonst Keras behalten)
        model_name = str(self.config.get("model_filename", ""))
        if model_name.endswith(".tflite"):
            try:
                tfl_path = os.path.join(self.config["paths"]["Models"], model_name)
                interpreter = tf.lite.Interpreter(model_path=tfl_path)
                interpreter.allocate_tensors()
                self.model = interpreter
                logging.info("TFLite-Interpreter aktiv (%s).", model_name)
            except Exception as e:
                logging.warning("TFLite nicht geladen (%s) – fallback auf Keras.", e)
        else:
            logging.info("Keras-Modell aktiv (%s).", model_name or "model.keras")

    def _prepare_input_data(self, payload: dict) -> tuple[np.ndarray | None, any, float | None]:
        """Bereitet einen Inferenz-Schritt vor und gibt (X_window[1,L,F], timestamp, true_value) zurück."""
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

        # Prüfe, ob alle trainierten Features vorhanden sind
        missing_features = [col for col in self.feature_list if col not in featured_buffer.columns]
        if missing_features:
            logger.warning(
                f"Warte: {len(missing_features)} Feature-Spalten fehlen noch (z.B. {missing_features[:3]}). "
                f"Puffer hat {len(featured_buffer)} Zeilen. Inferenz-Schritt wird übersprungen."
            )
            return None, None, None

        window_df = featured_buffer[self.feature_list].iloc[-self.lags:]

        if window_df.isnull().values.any():
            logging.warning("CNN1D: NaNs im Inferenzfenster – Schritt übersprungen.")
            return None, None, None

        # Skaliert & zu (1, L, F) formen
        window_scaled = self.scaler.transform(window_df.values)
        inference_window = np.expand_dims(window_scaled, axis=0)

        # Zeitstempel & wahrer Wert
        timestamp = pd.to_datetime(payload_lower.get('datetime'))
        if pd.isna(timestamp):
            timestamp = pd.Timestamp.utcnow()

        key_to_find = self.target_feature.lower()
        true_value = payload_lower.get(key_to_find)
        if true_value is None:
            logging.warning(f"CNN1D: Zielwert '{key_to_find}' nicht im Payload gefunden.")

        return inference_window, timestamp, true_value

    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        """Rücktransformation der Modellvorhersage mit dem gespeicherten y_scaler."""
        if self.y_scaler is None:
            raise RuntimeError(
                "Der 'y_scaler' wurde nicht gefunden oder geladen. Eine Rücktransformation ist nicht möglich."
            )
        pred_reshaped = np.asarray(prediction_scaled).reshape(-1, 1)
        return self.y_scaler.inverse_transform(pred_reshaped).flatten()
