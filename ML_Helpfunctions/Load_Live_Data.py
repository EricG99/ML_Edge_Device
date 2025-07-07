# Diese Datei könnte z.B. `Load_Lice_Data.py` heißen
import paho.mqtt.client as mqtt
import json
import time
import pandas as pd
import numpy as np

from Load_Prepare_Data import DataPipeline2D
# Importieren Sie Ihre DataPipeline2D Klasse
# from data_pipeline import DataPipeline2D 

class MqttInferenceClient:
    """
    Ein Client, der eine Daten-Pipeline verwaltet, MQTT-Nachrichten abonniert,
    diese verarbeitet und für die Inferenz vorbereitet.
    """
    def __init__(self, pipeline: DataPipeline2D, broker_ip: str, port: int, topic: str, sampling_interval_sec: float = 1.0):
        self.pipeline = pipeline
        self.topic = topic
        self.sampling_interval = sampling_interval_sec
        self.last_message_time = 0
        
        # Initialisiere den MQTT-Client
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        
        self.broker_ip = broker_ip
        self.port = port

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """Callback für die Verbindung."""
        if reason_code == 0:
            print(f"Erfolgreich mit MQTT Broker unter {self.broker_ip} verbunden.")
            client.subscribe(self.topic)
            print(f"Abonniert auf Topic: '{self.topic}'")
            print(f"Abtastrate eingestellt auf: {self.sampling_interval} Sekunden")
        else:
            print(f"Verbindung fehlgeschlagen mit Code: {reason_code}")

    def _on_message(self, client, userdata, msg):
        """Callback für den Empfang von Nachrichten."""
        current_time = time.time()
        if (current_time - self.last_message_time) < self.sampling_interval:
            return # Nachricht überspringen, wenn das Intervall noch nicht erreicht ist

        try:
            payload_str = msg.payload.decode('utf-8')
            data = json.loads(payload_str)
            
            # Leite die geparsten Daten an die Pipeline weiter
            inference_vector = self.pipeline.prepare_live_data_point(data)
            
            if inference_vector is not None:
                # Der Vektor ist fertig für das Modell
                print(f"[{data.get('datetime')}] Bereit für Inferenz. Vektor-Shape: {inference_vector.shape}")
                
                # Hier würde der Aufruf an Ihr Machine-Learning-Modell erfolgen:
                # prediction = your_ml_model.predict(inference_vector)
                # print(f"Vorhersage: {prediction}")
            
            self.last_message_time = current_time

        except json.JSONDecodeError:
            print("Fehler: Die empfangene Nachricht ist kein gültiges JSON.")
        except Exception as e:
            print(f"Ein unerwarteter Fehler in on_message ist aufgetreten: {e}")

    def run(self):
        """Startet den Client und die Endlosschleife."""
        print(f"Versuche, eine Verbindung zum MQTT Broker herzustellen: {self.broker_ip}:{self.port}...")
        try:
            self.client.connect(self.broker_ip, self.port, 60)
            print("Warte auf Nachrichten... (Beenden mit Strg+C)")
            self.client.loop_forever()
        except ConnectionRefusedError:
            print("\n[FEHLER] Die Verbindung wurde verweigert. Überprüfen Sie IP, Port und Broker-Status.")
        except KeyboardInterrupt:
            print("\nSkript durch Benutzer beendet. Trenne die Verbindung...")
            self.client.disconnect()
            print("Verbindung getrennt. Auf Wiedersehen!")
        except Exception as e:
            print(f"\nEin kritischer Fehler ist aufgetreten: {e}")


# Haupt-Ausführungsskript, z.B. `main.py`

# Annahme, die Klassen sind in den entsprechenden Dateien
# from data_pipeline import DataPipeline2D
# from live_inference import MqttInferenceClient 

if __name__ == "__main__":
    # --- 1. Trainingsphase: Pipeline initialisieren und trainieren ---
    
    # Konfiguration für das Training und die Live-Inferenz
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
    }

    # Erstellen Sie die Daten-Pipeline
    data_pipeline = DataPipeline2D(config)

    # Führen Sie das Training durch, um die Scaler und Feature-Listen zu erstellen
    # In einem echten Szenario würden Sie hier auch das ML-Modell trainieren.
    print("--- Starte Trainingsphase, um Pipeline zu initialisieren ---")
    data_pipeline.prepare_training_data()
    print("\n--- Trainingsphase abgeschlossen. Pipeline ist bereit für Live-Daten. ---")
    
    
    # --- 2. Inferenzphase: MQTT-Client starten ---

    # Konfiguration für den MQTT-Broker
    MQTT_BROKER_IP = "192.168.0.101"
    MQTT_PORT = 1883
    MQTT_TOPIC = "sim/data/20240341/S6"
    SAMPLING_INTERVAL_SECONDS = 2.0 # Lese nur alle 2 Sekunden eine Nachricht

    # Erstellen Sie den Inferenz-Client und übergeben Sie die initialisierte Pipeline
    inference_client = MqttInferenceClient(
        pipeline=data_pipeline,
        broker_ip=MQTT_BROKER_IP,
        port=MQTT_PORT,
        topic=MQTT_TOPIC,
        sampling_interval_sec=SAMPLING_INTERVAL_SECONDS
    )
    
    # Starten Sie den Client. Er läuft nun in einer Endlosschleife.
    inference_client.run()