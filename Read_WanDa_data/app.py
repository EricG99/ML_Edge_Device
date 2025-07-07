import threading
import json
import time
from collections import deque
from flask import Flask, render_template
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt

# --- Konfiguration ---
# HINWEIS: Passen Sie diese Werte bei Bedarf an Ihre Umgebung an.
MQTT_BROKER_IP = "192.168.0.101"  # IP-Adresse Ihres MQTT-Brokers
MQTT_PORT = 1883
MQTT_TOPIC_TO_SUBSCRIBE = "sim/data/20240341/S6"
# Daten werden alle 0.5 Sekunden (2x pro Sekunde) an die Webseite gesendet
UPDATE_INTERVAL_SECONDS = 0.5

# --- Globale, Thread-sichere Daten-Warteschlange ---
# Eine deque ist effizient für das Hinzufügen (append) und Entfernen (popleft) von Elementen.
data_queue = deque()
data_lock = threading.Lock()

# --- Flask und SocketIO App initialisieren ---
# Wir gehen davon aus, dass sich die 'templates' im selben Verzeichnis befinden.
app = Flask(__name__)
app.config['SECRET_KEY'] = 'sehr_geheimer_schluessel_123'
socketio = SocketIO(app)


# --- MQTT Client Setup ---
# Diese Funktionen werden als Callbacks für den MQTT-Client verwendet.

def on_connect(client, userdata, flags, reason_code, properties):
    """Wird aufgerufen, wenn die MQTT-Verbindung erfolgreich hergestellt wurde."""
    if reason_code == 0:
        print("MQTT-Verbindung erfolgreich hergestellt.")
        # Abonniert das gewünschte Topic nach erfolgreicher Verbindung.
        client.subscribe(MQTT_TOPIC_TO_SUBSCRIBE)
        print(f"Abonniert auf Topic: {MQTT_TOPIC_TO_SUBSCRIBE}")
    else:
        print(f"MQTT-Verbindung fehlgeschlagen mit Code {reason_code}")

def on_message(client, userdata, msg):
    """Wird bei jeder eingehenden MQTT-Nachricht aufgerufen."""
    try:
        # Dekodiert die Nachricht (Payload) von Bytes in einen String und dann in ein JSON-Objekt.
        data = json.loads(msg.payload.decode('utf-8'))
        # Verwendet einen Lock, um Thread-sicheren Zugriff auf die Warteschlange zu gewährleisten.
        with data_lock:
            data_queue.append(data)
    except Exception as e:
        print(f"Fehler beim Empfangen/Dekodieren der MQTT-Nachricht: {e}")

def data_aggregator_thread():
    """
    Ein separater Hintergrund-Thread, der Daten aus der Warteschlange sammelt,
    aggregiert und über SocketIO an alle verbundenen Web-Clients sendet.
    """
    while True:
        # Wartet für das definierte Intervall.
        time.sleep(UPDATE_INTERVAL_SECONDS)

        batch = []
        # Greift sicher auf die Warteschlange zu, um alle anstehenden Nachrichten zu holen.
        with data_lock:
            while data_queue:
                batch.append(data_queue.popleft())

        # Wenn keine neuen Nachrichten vorhanden sind, wird die Schleife fortgesetzt.
        if not batch:
            continue

        try:
            count = len(batch)
            # Berechnet den Durchschnitt des Massenflusses aus allen Nachrichten im Batch.
            # .get() wird verwendet, um Fehler zu vermeiden, falls der Schlüssel nicht existiert.
            avg_mass_flow = sum(d.get("Group4-2_S6_MassFlowRate", 0.0) for d in batch) / count

            # Nimmt den Zeitstempel des letzten Elements für die X-Achse.
            last_datetime = batch[-1].get("datetime", "1970-01-01 00:00:00.000")
            # Extrahiert nur den Zeit-Teil (HH:MM:SS).
            time_part = last_datetime.split(" ")[-1].split(".")[0]

            # Erstellt das Datenpaket, das an das Frontend gesendet wird.
            update_data = {
                'mass_flow': round(avg_mass_flow, 4), # Rundet auf 4 Dezimalstellen
                'timestamp': time_part
            }

            # Sendet das 'update_data'-Ereignis an alle verbundenen Clients.
            socketio.emit('update_data', update_data)
            print(f"Update gesendet: {update_data}")

        except Exception as e:
            print(f"Fehler bei der Datenaggregation: {e}")


# --- Flask Routen ---
@app.route('/')
def index():
    """Rendert die Hauptseite der Anwendung."""
    return render_template('index.html')


# --- Hauptausführung ---
if __name__ == '__main__':
    print("Starte MQTT-Client und Hintergrund-Thread...")
    # Initialisiert den MQTT-Client mit der neueren Callback-API-Version.
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message

    try:
        # Versucht, die Verbindung zum Broker herzustellen.
        mqtt_client.connect(MQTT_BROKER_IP, MQTT_PORT, 60)
    except Exception as e:
        print(f"Konnte keine Verbindung zum MQTT Broker herstellen unter {MQTT_BROKER_IP}:{MQTT_PORT}. Fehler: {e}")
        print("Bitte überprüfen Sie die MQTT-Konfiguration und stellen Sie sicher, dass der Broker erreichbar ist.")
        exit() # Beendet das Skript, wenn keine Verbindung möglich ist.


    # Startet die MQTT-Client-Schleife in einem eigenen Thread.
    mqtt_client.loop_start()

    # Startet den Datenaggregator-Thread.
    aggregator = threading.Thread(target=data_aggregator_thread)
    aggregator.daemon = True # Stellt sicher, dass der Thread mit der Hauptanwendung beendet wird.
    aggregator.start()

    print("Starte Flask-SocketIO-Server...")
    # Startet die Web-Anwendung. allow_unsafe_werkzeug wird für neuere Flask-Versionen benötigt.
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)