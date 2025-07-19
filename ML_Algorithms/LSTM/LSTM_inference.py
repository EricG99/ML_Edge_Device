# LSTM/lstm_inference.py
import time
import pandas as pd
import numpy as np
import threading
import logging
import sys
import os

# --- Suppress scikit-learn feature name warnings ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
from ML_Helpfunctions import Feature_Engeneering as fe
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline3D # Für die Typ-Hinweise

from config.config_ml_lstm import param_lstm_test
from config.config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LiveInferenceProcessorLSTM:
    """
    Verarbeitet Live-Daten für die LSTM-Inferenz.
    Baut einen 3D-Tensor als Eingabe für das LSTM-Modell auf.
    """
    def __init__(self, model, scaler, feature_list, config, on_prediction_callback=None):
        self.model = model
        self.scaler = scaler
        self.feature_list = feature_list
        self.config = config
        self.on_prediction_callback = on_prediction_callback
        self._live_data_buffer = pd.DataFrame()
        self.latest_payload = None
        self.lock = threading.Lock()
        self.is_running = False

        # LSTM-spezifische Parameter
        self.lags = config.get('lags', 10)
        self.target_feature = config['base_features'][0]
        self.inference_interval_sec = config.get("inference_interval_sec", 1.0)
        
        logging.info(f"LSTM Processor initialisiert mit einer Fenstergröße (lags) von {self.lags}.")

    def update_latest_data(self, data: dict):
        """Callback, um die neueste MQTT-Nachricht zu erhalten."""
        with self.lock:
            self.latest_payload = data

    def _prepare_live_window(self):
        """
        Bereitet ein 3D-Fenster [1, lags, features] aus dem internen Datenpuffer vor.
        """
        if self.latest_payload is None:
            return None, None

        # Neuen Datenpunkt zum Puffer hinzufügen
        new_row = pd.DataFrame([self.latest_payload])
        new_row['datetime'] = pd.to_datetime(new_row['datetime'])
        new_row = new_row.set_index('datetime')
        
        self._live_data_buffer = pd.concat([self._live_data_buffer, new_row]).sort_index()
        max_history = self.config.get('max_fe_window', 50) + self.lags
        if len(self._live_data_buffer) > max_history:
            self._live_data_buffer = self._live_data_buffer.iloc[-max_history:]

        # Prüfen, ob genügend Daten für ein vollständiges LSTM-Fenster vorhanden sind
        if len(self._live_data_buffer) < self.lags:
            logging.warning(
                f"Puffer füllt sich für LSTM-Fenster... {len(self._live_data_buffer)}/{self.lags} Zeilen vorhanden. Überspringe Inferenz."
            )
            return None, None

        # Feature Engineering auf dem gesamten Puffer durchführen
        featured_buffer, _ = fe.add_all_features(self._live_data_buffer.copy(), self.config)
        
        # Erneut prüfen, da FE Zeilen mit NaNs am Anfang entfernen könnte
        if len(featured_buffer) < self.lags:
            return None, None

        # Die letzten 'lags' Zeilen für das Fenster extrahieren
        window_df = featured_buffer[self.feature_list].iloc[-self.lags:]
        
        if window_df.isnull().values.any():
            logging.warning("NaN-Werte im finalen Inferenz-Fenster gefunden. Überspringe Inferenz.")
            return None, None

        # Fenster skalieren
        window_scaled = self.scaler.transform(window_df.values)
        
        # Für LSTM-Eingabe umformen: von [lags, features] zu [1, lags, features]
        inference_window = np.expand_dims(window_scaled, axis=0)
        
        return inference_window, window_df.index[-1]

    def run_inference_loop(self):
        """Die Kernschleife, die periodisch die Inferenz ausführt."""
        self.is_running = True
        while self.is_running:
            start_time = time.time()
            
            with self.lock:
                if self.latest_payload is None:
                    time.sleep(self.inference_interval_sec)
                    continue
                
                inference_window, timestamp_obj = self._prepare_live_window()

            if inference_window is not None:
                prediction, inference_duration_ms = PipelineUtils.run_timed_inference(
                    model=self.model, input_data=inference_window
                )
                
                # Log-Ausgabe für die Konsole
                logging.info(f"Prediction at {timestamp_obj.isoformat()}: {prediction[0].tolist()}")

            elapsed_time = time.time() - start_time
            sleep_time = max(0, self.inference_interval_sec - elapsed_time)
            time.sleep(sleep_time)
        logging.info("Inference loop stopped.")
        
    def stop(self):
      self.is_running = False

def run_standalone_inference(config, broker_ip, port, topic):
    """
    Verwaltet den MQTT-Client und die Inferenzschleife für die eigenständige Ausführung.
    """
    try:
        # Lädt Modell (.keras oder .tflite), scaler (.joblib) und features (.joblib)
        scaler, features, model = PipelineUtils.load_model_artifacts_for_inference(config)
    except (FileNotFoundError, ValueError) as e:
        logging.error(f"Fehler beim Laden der Artefakte: {e}")
        sys.exit(1)
        
    processor = LiveInferenceProcessorLSTM(model, scaler, features, config)
    
    # Starte die Inferenzschleife in einem separaten Thread
    inference_loop_thread = threading.Thread(target=processor.run_inference_loop, daemon=True)
    inference_loop_thread.start()

    # Richte den MQTT-Client ein, um Daten zu empfangen
    mqtt_client = MqttInferenceClient(
        broker_ip=broker_ip, port=port, topic=topic,
        on_message_callback=processor.update_latest_data
    )
    logging.info("Starting MQTT client for LSTM inference...")
    mqtt_client.run() # Blockiert, bis der Client gestoppt wird

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Shutting down LSTM standalone inference...")
        processor.stop()
        mqtt_client.client.loop_stop()
        mqtt_client.client.disconnect()
        inference_loop_thread.join()
        logging.info("Shutdown complete.")

if __name__ == "__main__":
    logging.info("--- MODE: Standalone LSTM Inference (Console Output) ---")
    
    # --- Basiskonfiguration für den Test ---
    # Nutzt die Logik aus dem Haupt-Inferenzskript
    infer_config = param_lstm_test.copy()
    infer_config.update(CONFIG_LOAD_ARTIFACTS)
    infer_config['paths'] = CONFIG_PATH['paths']
    
    # Beispiel-Pfade für den 'fast' mode (müssen durch die Trainingsartefakte ersetzt werden)
    infer_config['model_path_static'] = "trained_lstm_model.keras" 
    infer_config['scaler_path_static'] = "trained_lstm_scaler.joblib"
    infer_config['features_path_static'] = "trained_lstm_features.joblib"

    infer_config['model_filename'] = "model.keras"

    
    # MQTT-Konfiguration
    mqtt_broker_ip = MQTT_CONFIG['MQTT_BROKER_IP']
    mqtt_port = MQTT_CONFIG['MQTT_PORT']
    mqtt_topic = MQTT_CONFIG['MQTT_TOPIC']
    
    run_standalone_inference(infer_config, mqtt_broker_ip, mqtt_port, mqtt_topic)