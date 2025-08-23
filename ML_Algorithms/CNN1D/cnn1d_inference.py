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
        Wird nach dem Laden der Artefakte aufgerufen.
        Strukturierte Reihenfolge:
        1. Konfiguration synchronisieren (Training -> dann Runtime-Overrides beibehalten).
        2. Endgültiges Modellobjekt (Keras oder TFLite) bestimmen.
        3. `self.lags` mit dem finalen Modell synchronisieren (und DataProcessor danach ggf. neu aufsetzen).
        """
        # --- SCHRITT 1: Konfiguration mit Trainings-Artefakten synchronisieren ---
        if getattr(self, "training_config", None):
            runtime_cfg = dict(self.config)  # enthält CLI-Overrides (z. B. inference_steps=20)
            # optional: Puffer sichern, wenn schon vorhanden
            old_buf = None
            if hasattr(self, "data_processor") and hasattr(self.data_processor, "buffer_df"):
                old_buf = getattr(self.data_processor, "buffer_df", None)

            # Trainings-Config als Basis…
            self.config.clear()
            self.config.update(self.training_config)

            # …und explizit die Runtime-Overrides beibehalten (wie bei LSTM)
            preserved_keys = (
                "inference_steps",
                "inference_interval_sec",
                "loading_strategy",
                "quantization",
                "model_filename",
            )
            preserved = {k: runtime_cfg[k] for k in preserved_keys if k in runtime_cfg}
            self.config.update(preserved)

            # DataProcessor mit gemergter Config neu initialisieren, Puffer (falls vorhanden) zurücklegen
            self.data_processor = RealTimeDataProcessor(self.config)
            if old_buf is not None:
                try:
                    self.data_processor.buffer_df = old_buf.copy()
                except Exception:
                    pass
            logging.info("RealTimeDataProcessor re-initialized with loaded training config.")

        # --- SCHRITT 2: Endgültiges Modellobjekt bestimmen ---
        model_name = str(self.config.get("model_filename", ""))
        if model_name.endswith(".tflite"):
            try:
                model_path = os.path.join(self.config["paths"]["Models"], model_name)
                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                self.model = interpreter  # TFLite ist aktiv
                logging.info(f"📦 TFLite-Interpreter ist jetzt das aktive Modell ({model_name}).")
            except Exception as e:
                logging.warning(f"⚠️ TFLite konnte nicht geladen werden ({model_name}): {e} – Fallback auf Standardmodell.")
        else:
            logging.info(f"📦 Keras-Modell ist das aktive Modell ({model_name or 'model.keras'}).")

        # --- SCHRITT 3: `self.lags` robust mit dem FINALEN Modell synchronisieren ---
        try:
            # Basiswert aus Config
            self.lags = int(self.config.get("lags", self.lags))
            model_lags = None

            # Keras?
            if hasattr(self.model, "input_shape") and self.model.input_shape:
                model_lags = self.model.input_shape[1]
            # TFLite?
            elif hasattr(self.model, "get_input_details"):
                inp = self.model.get_input_details()[0]
                if "shape" in inp and len(inp["shape"]) > 1:
                    model_lags = int(inp["shape"][1])

            # Abgleich
            if isinstance(model_lags, int) and model_lags > 0 and model_lags != self.lags:
                logging.warning(f"Modell-Input verlangt lags={model_lags}; korrigiere self.lags von {self.lags} -> {model_lags}")
                self.lags = model_lags
                self.config["lags"] = self.lags  # in Config spiegeln

                # DataProcessor nach Lags-Änderung erneut neu aufsetzen und Puffer retten
                old_buf = None
                if hasattr(self, "data_processor") and hasattr(self.data_processor, "buffer_df"):
                    old_buf = getattr(self.data_processor, "buffer_df", None)
                self.data_processor = RealTimeDataProcessor(self.config)
                if old_buf is not None:
                    try:
                        self.data_processor.buffer_df = old_buf
                    except Exception:
                        pass

            logging.info(f"Finale Synchronisierung: `self.lags` ist jetzt auf {self.lags} gesetzt.")
        except Exception as e:
            logging.error(f"Fehler bei der finalen Synchronisierung von self.lags: {e}")


    def _prepare_input_data(self, payload: dict):
        """
        Bereitet das Inferenzfenster (Shape: (1, lags, n_features)) für das 1D-CNN auf.
        Rückgabe:
        - X_input: np.ndarray mit Shape (1, lags, n_features) oder None, wenn noch nicht bereit
        - timestamp: Zeitstempel der aktuellen Zeile (falls vorhanden), sonst UTC now
        - y_true: der echte Wert aus dem Payload (für Logging/Metriken), sonst None/np.nan
        """
        try:
            # Keys vereinheitlichen
            payload_lower = {str(k).lower(): v for k, v in (payload or {}).items()}

            # 1) Nur den Feature-Puffer holen (KEIN Tuple!)
            featured_buffer = self.data_processor.update_and_process(payload_lower)

            # 2) Puffer noch nicht ausreichend gefüllt?
            if featured_buffer is None or len(featured_buffer) < self.lags:
                need = self.lags
                have = 0 if featured_buffer is None else len(featured_buffer)
                logging.info(f"Datenpuffer wird gefüllt... {have}/{need}")
                return None, None, None

            # 3) Exakte Trainings-Feature-Reihenfolge und Fenster ziehen
            window_df = featured_buffer[self.feature_list].iloc[-self.lags:]

            # 4) NaN-Check wie bei LSTM
            if window_df.isnull().values.any():
                logger.warning("CNN1D: NaNs im Inferenzfenster – Schritt übersprungen.")
                return None, None, None

            # 5) Skalieren (wie bei LSTM) und für 1D-CNN reshapen
            window_scaled = self.scaler.transform(window_df.values)
            X_input = np.expand_dims(window_scaled, axis=0)  # (1, lags, features)

            # 6) Timestamp robust bestimmen
            timestamp = pd.to_datetime(payload_lower.get('datetime'))
            if pd.isna(timestamp):
                timestamp = pd.Timestamp.utcnow()

            # 7) True-Value robust holen (kleinschreibung)
            key_to_find = (self.target_feature or "").lower()
            y_true = payload_lower.get(key_to_find)
            if y_true is None:
                logger.warning("CNN1D: Zielwert '%s' nicht im Payload gefunden.", key_to_find)

            return X_input, timestamp, y_true

        except Exception as e:
            logger.error(f"Fehler bei _prepare_input_data: {e}", exc_info=True)
            return None, None, None
        

    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        """Rücktransformation der Modellvorhersage mit dem gespeicherten y_scaler."""
        if self.y_scaler is None:
            raise RuntimeError(
                "Der 'y_scaler' wurde nicht gefunden oder geladen. Eine Rücktransformation ist nicht möglich."
            )
        pred_reshaped = np.asarray(prediction_scaled).reshape(-1, 1)
        return self.y_scaler.inverse_transform(pred_reshaped).flatten()
