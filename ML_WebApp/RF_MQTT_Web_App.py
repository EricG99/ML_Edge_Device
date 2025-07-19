# RF_MQTT_Web_App.py
import sys
import os
import logging
import numpy as np
import pandas as pd
import json
import time
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, jsonify
from threading import Thread, Lock, Event
import queue
import atexit
from pathlib import Path
import paho.mqtt.client as mqtt

# --- Pfad-Konfiguration ---
try:
    # Geht zwei Ebenen vom aktuellen Skriptordner nach oben, um zum Projekt-Root zu gelangen
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    
    from ML_Helpfunctions.Load_Prepare_Data import DataPipeline2D
    # KORRIGIERT: Import an die von Ihnen gezeigte Schreibweise angepasst.
    from ML_Helpfunctions import Feature_Engeneering as fe 
except ImportError as e:
    print(f"Konnte Module nicht laden. Fehler: {e}. Stellen Sie die korrekte Projektstruktur sicher.")
    sys.exit(1)

# --- Logging Konfiguration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Globale Status- und Datenverwaltung ---
APP_STATE = {
    "status": "initializing",
    "message": "Anwendung wird initialisiert.",
    "model": None,
    "pipeline": None,
}
STATE_LOCK = Lock()
DATA_QUEUE = queue.Queue()
SHUTDOWN_EVENT = Event()

# --- MQTT CLIENT KLASSE ---
class MqttInferenceClient:
    def __init__(self, model, pipeline: DataPipeline2D, broker_ip: str, port: int, topic: str, sampling_interval_sec: float):
        self.model = model
        self.pipeline = pipeline
        self.topic = topic
        self.sampling_interval = sampling_interval_sec
        self.last_message_time = 0
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.broker_ip = broker_ip
        self.port = port

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            logging.info(f"MQTT: Erfolgreich mit Broker unter {self.broker_ip} verbunden.")
            client.subscribe(self.topic)
        else:
            logging.error(f"MQTT: Verbindung fehlgeschlagen mit Code: {reason_code}")

    def _on_message(self, client, userdata, msg):
        if time.time() - self.last_message_time < self.sampling_interval:
            return
        try:
            payload_str = msg.payload.decode('utf-8')
            data = json.loads(payload_str)
            inference_vector = self.pipeline.prepare_live_data_point(data)
            
            if inference_vector is not None:
                prediction_scaled = self.model.predict(inference_vector)
                prediction_descaled = self.pipeline.y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
                final_prediction = prediction_descaled.flatten()[0]
                
                result_for_frontend = {
                    "date": data.get('datetime', 'N/A'),
                    "predicted_value": final_prediction
                }
                DATA_QUEUE.put(result_for_frontend)
            
            self.last_message_time = time.time()
        except Exception as e:
            logging.error(f"MQTT-Fehler in _on_message: {e}", exc_info=True)

    def run(self):
        logging.info(f"MQTT: Versuche Verbindung zu {self.broker_ip}:{self.port}...")
        try:
            self.client.connect(self.broker_ip, self.port, 60)
            self.client.loop_start()
            SHUTDOWN_EVENT.wait()
            self.client.loop_stop()
            self.client.disconnect()
            logging.info("MQTT: Verbindung sauber getrennt.")
        except Exception as e:
            logging.critical(f"MQTT: Kritischer Fehler: {e}", exc_info=True)
            with STATE_LOCK:
                APP_STATE["status"] = "error"
                APP_STATE["message"] = f"MQTT-Verbindung fehlgeschlagen: {e}"

# --- Trainings- und Inferenz-Logik ---
def train_and_run_inference(config):
    global APP_STATE
    try:
        with STATE_LOCK:
            APP_STATE["status"] = "training"
            APP_STATE["message"] = "Modell wird trainiert..."
        
        logging.info("--- Starte Trainingsphase im Hintergrund ---")
        data_pipeline = DataPipeline2D(config)
        X_train, y_train = data_pipeline.prepare_training_data()
        
        logging.info("Trainiere RandomForest-Modell...")
        rf_model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", None),
            random_state=config.get("random_state", 42),
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        with STATE_LOCK:
            APP_STATE["model"] = rf_model
            APP_STATE["pipeline"] = data_pipeline
            APP_STATE["status"] = "inference_running"
            APP_STATE["message"] = "Training abgeschlossen. Warte auf Live-Daten..."
        logging.info("--- Trainingsphase erfolgreich abgeschlossen. ---")

    except Exception as e:
        logging.error(f"Fehler während der Trainingsphase: {e}", exc_info=True)
        with STATE_LOCK:
            APP_STATE["status"] = "error"
            APP_STATE["message"] = f"Trainingsfehler: {e}"
        return

    if SHUTDOWN_EVENT.is_set(): return

    logging.info("\n--- Starte Live-Inferenz-Phase. ---")
    inference_client = MqttInferenceClient(
        model=APP_STATE["model"],
        pipeline=APP_STATE["pipeline"],
        broker_ip=config["mqtt_broker_ip"],
        port=config["mqtt_port"],
        topic=config["mqtt_topic"],
        sampling_interval_sec=config["sampling_interval_sec"]
    )
    inference_client.run()

# --- Flask Web App ---
script_dir = Path(__file__).resolve().parent
template_folder_path = script_dir / 'templates'
logging.info(f"Flask sucht nach Templates im Ordner: {template_folder_path}")
app = Flask(__name__, template_folder=str(template_folder_path))

@app.route('/')
@app.route('/dashboard.html')
def index():
    # KORRIGIERT: Verweist jetzt auf den Dateinamen, der wahrscheinlich bei Ihnen existiert.
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    with STATE_LOCK:
        status_copy = {k: v for k, v in APP_STATE.items() if k not in ['model', 'pipeline']}
        return jsonify(status_copy)

@app.route('/api/data')
def get_data():
    try:
        data_point = DATA_QUEUE.get_nowait()
        return jsonify(data_point)
    except queue.Empty:
        return '', 204

# --- Haupt-Startpunkt ---
if __name__ == "__main__":
    config = {
        "train_csv_path": r"C:\DEV\RevPi_ML\ML_Edge_Device\Input\Input_Data\train_data_sample.csv",
        "loading_strategy": "live_mqtt",
        "lags": 5, 
        "horizon": 1,
        "base_features": ["Group4-2_S6_MassFlowRate", "Group4-2_S6_Pressure", "Group4-2_S6_Temperature"],
        "scale_target": True, 
        "scale_other_features": True,
        "min_fe_window": 20, 
        "max_fe_window": 60,
        "rolling_window_size": 4, 
        "include_roll_mean": True, 
        "include_roll_std": True,
        "n_estimators": 150, 
        "max_depth": 10, 
        "random_state": 42,
        "mqtt_broker_ip": "192.168.0.101", 
        "mqtt_port": 1883,
        "mqtt_topic": "sim/data/20240341/S6", 
        "sampling_interval_sec": 0.1
    }

    logging.info("Starte den ML-Hintergrund-Thread...")
    ml_thread = Thread(target=train_and_run_inference, args=(config,))
    ml_thread.start()

    def shutdown_handler():
        logging.info("Shutdown-Handler wird aufgerufen...")
        SHUTDOWN_EVENT.set()
    atexit.register(shutdown_handler)

    logging.info("Starte den Flask Web Server auf http://0.0.0.0:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)
