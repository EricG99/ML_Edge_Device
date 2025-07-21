# Random_Forest_Web_App.py
import sys
import os
import logging
import numpy as np
import pandas as pd
import json
import time
from sklearn.ensemble import RandomForestRegressor

# Keine Notwendigkeit für subprocess mehr

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# 2. Füge den Projekt-Root zum sys.path hinzu, falls er noch nicht da ist.
#    Dies ist der EINZIGE sys.path.append, den Sie benötigen!
if project_root not in sys.path:
    sys.path.append(project_root)

# --- JETZT FUNKTIONIEREN ALLE IMPORTE ---
# Alle Importe werden jetzt als "absolute" Importe vom Projekt-Root aus behandelt.

from ML_WebApp import Flask_App
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
from ML_Algorithms import RF_Run_Pipeline as RFRunPipeline
from ML_Helpfunctions import RF_Utils as RFUtils
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline2D

# Annahme: config.py liegt direkt im Projekt-Root
from ML_Algorithms import CONFIG_PATH, param_rf

# Kombinierte Konfiguration
CONFIG_RF_ALL = {**CONFIG_PATH, **param_rf}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


import paho.mqtt.client as mqtt

class MqttInferenceClient:
    """
    Ein Client, der eine Daten-Pipeline und ein trainiertes Modell verwaltet, 
    MQTT-Nachrichten abonniert, diese verarbeitet, eine Inferenz durchführt 
    und das Ergebnis ausgibt.
    """
    def __init__(self, model, pipeline: DataPipeline2D, broker_ip: str, port: int, topic: str, sampling_interval_sec: float = 1.0):
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
            logging.info(f"Erfolgreich mit MQTT Broker unter {self.broker_ip} verbunden.")
            client.subscribe(self.topic)
            logging.info(f"Abonniert auf Topic: '{self.topic}' mit Abtastrate: {self.sampling_interval}s")
        else:
            logging.error(f"Verbindung zum MQTT Broker fehlgeschlagen mit Code: {reason_code}")

    def _on_message(self, client, userdata, msg):
        """
        Callback für den Empfang von Nachrichten. Führt den gesamten Inferenzprozess aus.
        """
        current_time = time.time()
        if (current_time - self.last_message_time) < self.sampling_interval:
            return

        try:
            payload_str = msg.payload.decode('utf-8')
            data = json.loads(payload_str)
            
            # 1. DATEN VORBEREITEN: Live-Datenpunkt durch die Pipeline schicken
            inference_vector = self.pipeline.prepare_live_data_point(data)
            
            if inference_vector is not None:
                # 2. INFERENZ DURCHFÜHREN
                # Das Modell erwartet eine 2D-Array, unser Vektor ist bereits (1, n_features)
                prediction_scaled = self.model.predict(inference_vector)
                
                # 3. ERGEBNIS DESKALIEREN
                # Wir müssen das Ergebnis in die richtige Form für den y_scaler bringen
                prediction_descaled = self.pipeline.y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1))
                
                # 4. ERGEBNIS AUSGEBEN
                date_str = data.get('datetime', 'Unbekanntes Datum')
                final_prediction = prediction_descaled.flatten()[0]
                
                logging.info(f"[{date_str}] VORHERSAGE: {final_prediction:.4f}")
            
            self.last_message_time = current_time

        except json.JSONDecodeError:
            logging.warning("Empfangene MQTT-Nachricht ist kein gültiges JSON.")
        except Exception as e:
            logging.error(f"Fehler in _on_message: {e}", exc_info=True)

    def run(self):
        """Startet den Client und die Endlosschleife."""
        logging.info(f"Versuche, eine Verbindung zum MQTT Broker herzustellen: {self.broker_ip}:{self.port}...")
        try:
            self.client.connect(self.broker_ip, self.port, 60)
            self.client.loop_forever()
        except ConnectionRefusedError:
            logging.error("Verbindung verweigert. Prüfen Sie IP, Port und Broker-Status.")
        except KeyboardInterrupt:
            logging.info("Skript durch Benutzer beendet. Trenne die Verbindung...")
            self.client.disconnect()
            logging.info("Verbindung getrennt.")
        except Exception as e:
            logging.critical(f"Ein kritischer Fehler ist aufgetreten: {e}", exc_info=True)


# --- NEUE, ZENTRALE TRAININGSFUNKTION ---
def train_model_pipeline(config: dict):
    """
    Führt den gesamten Trainingsprozess aus:
    1. Initialisiert die Datenpipeline.
    2. Bereitet die Trainingsdaten vor.
    3. Trainiert ein RandomForest-Modell.
    
    Returns:
        tuple: Ein Tuple mit (trainiertes Modell, initialisierte Datenpipeline).
    """
    logging.info("--- Starte Trainingsphase ---")
    
    # 1. Pipeline initialisieren
    data_pipeline = DataPipeline2D(config)
    
    # 2. Trainingsdaten vorbereiten
    X_train, y_train = data_pipeline.prepare_training_data()
    
    if X_train is None or y_train is None:
        logging.error("Trainingsdaten konnten nicht vorbereitet werden. Breche ab.")
        return None, None
        
    logging.info(f"Trainingsdaten erfolgreich vorbereitet. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    # 3. RandomForest-Modell trainieren
    logging.info("Trainiere RandomForest-Modell...")
    rf_model = RandomForestRegressor(
        n_estimators=config.get("n_estimators", 100),
        max_depth=config.get("max_depth", None),
        random_state=config.get("random_state", 42),
        n_jobs=-1 # Nutze alle verfügbaren CPU-Kerne
    )
    
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    end_time = time.time()
    
    logging.info(f"Modell-Training abgeschlossen in {end_time - start_time:.2f} Sekunden.")
    
    # Gib das trainierte Modell und die initialisierte Pipeline zurück.
    # Die Pipeline enthält die wichtigen Scaler für die Live-Inferenz.
    return rf_model, data_pipeline


# --- ZENTRALER STARTPUNKT DER ANWENDUNG ---
if __name__ == "__main__":
    # 1. KONFIGURATION
    # Alle Einstellungen an einem Ort
    config = {
        "loading_strategy": "live_mqtt", # Wichtig: Sagt der Pipeline, dass sie für Live-Daten vorbereitet wird
        "train_csv_path": r"C:\Users\ericg\Documents\Mechatronik M Sc\6. Semster\MA\Dev_Ma\ML_Edge_Device\Input\Input_Data\train_data_sample.csv",        
        "lags": 5,
        "horizon": 1, # Bei Live-Inferenz ist ein Horizont von 1 typisch
        "base_features": ["Group4-2_S6_MassFlowRate", "Group4-2_S6_Pressure", "Group4-2_S6_Temperature"],
        "scale_target": True,
        "scale_other_features": True,
        "min_fe_window": 20, # Mindestanzahl an Datenpunkten für rollierende Features
        "max_fe_window": 60, # Maximale Länge des internen Puffers
        "rolling_window_size": 4,
        "include_roll_mean": True,
        "include_roll_std": True,
        
        # Modell-Hyperparameter
        "n_estimators": 150,
        "max_depth": 10,
        "random_state": 42,
        
        # MQTT-Einstellungen
        "mqtt_broker_ip": "192.168.0.101",
        "mqtt_port": 1883,
        "mqtt_topic": "sim/data/20240341/S6",
        "sampling_interval_sec": 0.1
    }

    # 2. MODELL TRAINIEREN
    # Führe die Trainingspipeline aus, um das Modell und die initialisierte Datenpipeline zu erhalten.
    model, pipeline = train_model_pipeline(config)

    # 3. LIVE-INFERENZ STARTEN
    # Prüfe, ob das Training erfolgreich war, bevor der MQTT-Client gestartet wird.
    if model and pipeline:
        logging.info("\n--- Trainingsphase erfolgreich. Starte Live-Inferenz-Phase. ---")
        
        inference_client = MqttInferenceClient(
            model=model,
            pipeline=pipeline,
            broker_ip=config["mqtt_broker_ip"],
            port=config["mqtt_port"],
            topic=config["mqtt_topic"],
            sampling_interval_sec=config["sampling_interval_sec"]
        )
        
        # Diese Funktion blockiert und läuft, bis das Skript beendet wird.
        inference_client.run()
    else:
        logging.critical("Das Training ist fehlgeschlagen. Die Anwendung wird nicht gestartet.")
        sys.exit(1)
