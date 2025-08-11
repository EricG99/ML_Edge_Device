import pandas as pd
import numpy as np
import logging
import sys
import os
import argparse

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
from config.config_ml_lstm import param_lstm_test
from config.config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG

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
    """
    Spezialisierte Inferenzklasse für LSTM, die den optimierten RealTimeDataProcessor verwendet.
    Nutzt dieselbe Fenster-/Feature-Logik wie im Training.
    """
    def __init__(self, config: dict, broker_ip: str, port: int, topic: str, folder_flag: str):
        super().__init__(config, broker_ip, port, topic, folder_flag)
        self.lags = int(config.get('lags', 10))
        self.target_feature = config.get('base_features', [None])[0]
        if self.target_feature is None:
            raise KeyError("config['base_features'][0] fehlt – Ziel-Feature unbekannt.")

        self.data_processor = RealTimeDataProcessor(config)

        logger.info(f"[Init] lags={self.lags}, target_feature='{self.target_feature}', "
                    f"inference_interval={self.inference_interval}s")

    def _prepare_input_data(self):
        """
        Bereitet ein 3D-Fenster [1, lags, features] für das LSTM-Modell vor.
        Gibt (input_3d, timestamp, true_value) oder (None, None, None) zurück.
        """
        if self.latest_payload is None:
            logger.debug("[prepare] Kein Payload vorhanden – überspringe.")
            return None, None, None

        # 1) Daten an Prozessor übergeben (berechnet Features; verwaltet Puffer)
        try:
            logger.debug(f"[prepare] Raw payload keys: {list(self.latest_payload.keys())}")
            featured_buffer = self.data_processor.update_and_process(self.latest_payload)
        except Exception as e:
            logger.error(f"[prepare] update_and_process() fehlgeschlagen: {e}", exc_info=True)
            return None, None, None

        if featured_buffer is None:
            logger.debug("[prepare] featured_buffer=None (Puffer/Warmup) – überspringe.")
            return None, None, None

        # 2) Feature-Liste prüfen
        if not self.feature_list:
            logger.error("[prepare] feature_list ist leer/None – Artefakte schon geladen?")
            return None, None, None

        missing = [f for f in self.feature_list if f not in featured_buffer.columns]
        if missing:
            logger.warning(f"[prepare] Es fehlen {len(missing)} Feature(s) im Buffer (zeige bis 5): {missing[:5]}")
            logger.debug(f"[prepare] Buffer columns sample (bis 20): {list(featured_buffer.columns[:20])}")
            return None, None, None

        # 3) Letztes Fenster (lags)
        if len(featured_buffer) < self.lags:
            logger.debug(f"[prepare] Buffer zu klein ({len(featured_buffer)}/{self.lags}) – überspringe.")
            return None, None, None

        window_df = featured_buffer[self.feature_list].iloc[-self.lags:]
        logger.debug(f"[prepare] window_df shape={window_df.shape}")

        # 4) NaN-Check
        if window_df.isnull().values.any():
            n_nan = int(np.isnan(window_df.values).sum())
            logger.warning(f"[prepare] NaNs im Fenster entdeckt (n={n_nan}). Überspringe Schritt.")
            return None, None, None

        # 5) Skalieren & in 3D-Format
        try:
            window_scaled = self.scaler.transform(window_df.values)
        except Exception as e:
            logger.error(f"[prepare] scaler.transform() fehlgeschlagen: {e}", exc_info=True)
            return None, None, None

        logger.debug(f"[prepare] window_scaled min/max: {window_scaled.min():.6f}/{window_scaled.max():.6f}")
        inference_window = np.expand_dims(window_scaled, axis=0)

        # 6) Metadaten
        timestamp = window_df.index[-1]
        payload_lower = {str(k).lower(): v for k, v in self.latest_payload.items()}
        true_value = payload_lower.get(str(self.target_feature).lower())

        logger.debug(f"[prepare] timestamp={timestamp}, true_value={true_value}")
        return inference_window, timestamp, true_value


# --- Standalone CLI zum schnellen Testen ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone LSTM Inference")
    parser.add_argument("--load_id", type=str, help="Optional: The specific run ID to load artifacts from.")
    parser.add_argument("--model_filename", type=str, help="Optional: The specific model filename.")
    parser.add_argument("--loglevel", type=str, default=os.environ.get("LOGLEVEL", "INFO"),
                        help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.loglevel.upper(), logging.INFO))
    logger.setLevel(getattr(logging, args.loglevel.upper(), logging.INFO))

    logger.info("--- MODE: Standalone LSTM Inference (Console Output) ---")

    infer_config = param_lstm_test.copy()
    infer_config.update(CONFIG_LOAD_ARTIFACTS)
    infer_config['paths'] = CONFIG_PATH['paths']

    if args.load_id:
        infer_config['load_id'] = args.load_id
        infer_config['inference_mode'] = 'load_artifacts_path'
    if args.model_filename:
        infer_config['model_filename'] = args.model_filename

    mqtt_broker_ip = MQTT_CONFIG['MQTT_BROKER_IP']
    mqtt_port = MQTT_CONFIG['MQTT_PORT']
    mqtt_topic = MQTT_CONFIG['MQTT_TOPIC']

    processor = LSTMInference(infer_config, mqtt_broker_ip, mqtt_port, mqtt_topic, FOLDER_FLAG)
    # BaseInferenceProcessor.run() steuert die komplette Strategie (split/live_mqtt) & 1-Hz-Takt
    processor.run()

    # Zum Abschluss letztes Payload (falls vorhanden) kurz loggen
    if processor.latest_payload:
        pl = {str(k).lower(): v for k, v in processor.latest_payload.items()}
        logger.info(f"[main] last payload snapshot: keys={list(pl.keys())[:8]}")
