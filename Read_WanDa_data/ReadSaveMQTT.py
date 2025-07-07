import paho.mqtt.client as mqtt
import time
import json
import csv
import os
from datetime import datetime

# --- Konfiguration ---
MQTT_BROKER_IP = "192.168.0.101"  # IP-Adresse oder Hostname Ihres MQTT-Brokers
MQTT_PORT = 1883                   # Standard-MQTT-Port
MQTT_TOPIC = "sim/data/20240341/S6" # Das spezifische Topic, das aufgezeichnet werden soll

def record_mqtt_to_csv(output_path: str, duration_seconds: int):
    """
    Verbindet sich mit einem MQTT-Broker, zeichnet Nachrichten von einem spezifischen
    Topic für eine bestimmte Dauer auf und speichert sie in einer CSV-Datei.

    Args:
        output_path (str): Der vollständige Pfad zur CSV-Datei, in die gespeichert werden soll.
        duration_seconds (int): Die Dauer der Aufzeichnung in Sekunden.
    """
    print("Initialisiere MQTT-Rekorder...")
    
    # Eine Liste, um die empfangenen Nachrichten (als Dictionaries) zu speichern
    received_messages = []
    
    # --- Interne Callback-Funktionen ---
    def on_connect(client, userdata, flags, reason_code, properties):
        """Wird aufgerufen, wenn die Verbindung zum Broker steht."""
        if reason_code == 0:
            print(f"Erfolgreich mit MQTT Broker unter {MQTT_BROKER_IP} verbunden.")
            client.subscribe(MQTT_TOPIC)
            print(f"Abonniert auf Topic: '{MQTT_TOPIC}'")
        else:
            print(f"Verbindung fehlgeschlagen mit Code: {reason_code}")
            # Beendet die Funktion, wenn keine Verbindung hergestellt werden kann
            return

    def on_message(client, userdata, msg):
        """Wird bei jeder eingehenden Nachricht aufgerufen."""
        try:
            # Dekodiere die JSON-Payload in ein Python-Dictionary
            data = json.loads(msg.payload.decode('utf-8'))
            
            # Füge einen Zeitstempel hinzu, wann die Nachricht empfangen wurde
            data['recording_timestamp'] = datetime.now().isoformat()
            
            # Füge das Dictionary zur Liste hinzu
            received_messages.append(data)
        except json.JSONDecodeError:
            print(f"Warnung: Konnte die Nachricht nicht als JSON dekodieren. Topic: {msg.topic}")
        except Exception as e:
            print(f"Ein Fehler ist in on_message aufgetreten: {e}")

    # --- Hauptlogik der Funktion ---
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER_IP, MQTT_PORT, 60)
    except Exception as e:
        print(f"Kritischer Fehler: Verbindung zum Broker {MQTT_BROKER_IP} nicht möglich. {e}")
        return

    # Starte die Netzwerk-Schleife in einem Hintergrund-Thread
    client.loop_start()

    print(f"--- Aufzeichnung für {duration_seconds} Sekunden gestartet ---")
    print("Drücken Sie Strg+C, um die Aufzeichnung vorzeitig zu beenden.")
    
    try:
        # Warte für die angegebene Dauer
        time.sleep(duration_seconds)
    except KeyboardInterrupt:
        print("\nAufzeichnung durch Benutzer vorzeitig beendet.")

    print("--- Aufzeichnung beendet ---")

    # Stoppe die Netzwerk-Schleife und trenne die Verbindung
    client.loop_stop()
    client.disconnect()
    print("Verbindung zum MQTT Broker getrennt.")

    # --- Speichern der Daten in CSV ---
    if not received_messages:
        print("Keine Nachrichten empfangen. Es wird keine CSV-Datei erstellt.")
        return

    print(f"Speichere {len(received_messages)} Nachrichten in '{output_path}'...")

    try:
        # Stelle sicher, dass das Verzeichnis existiert
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Extrahiere die Spaltenüberschriften aus dem ersten Datensatz
        headers = list(received_messages[0].keys())
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            
            # Schreibe die Kopfzeile
            writer.writeheader()
            
            # Schreibe alle gesammelten Nachrichten
            writer.writerows(received_messages)
            
        print(f"✅ CSV-Datei erfolgreich unter '{output_path}' gespeichert.")

    except Exception as e:
        print(f"Fehler beim Schreiben der CSV-Datei: {e}")


# --- Beispiel für die Ausführung ---
if __name__ == "__main__":
    # Definieren Sie hier den Speicherort und die Aufnahmedauer
    AUFNAHMEDAUER_IN_SEKUNDEN = 10
    SPEICHERORT_CSV = "aufzeichnungen/mqtt_data.csv"
    
    # Rufe die Hauptfunktion auf
    record_mqtt_to_csv(output_path=SPEICHERORT_CSV, duration_seconds=AUFNAHMEDAUER_IN_SEKUNDEN)
