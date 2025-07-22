import paho.mqtt.client as mqtt
import time
import json
import csv
import os
from datetime import datetime

# --- Konfiguration ---
MQTT_BROKER_IP = "192.168.0.101"  # IP-Adresse oder Hostname Ihres MQTT-Brokers
MQTT_PORT = 1883                  # Standard-MQTT-Port
MQTT_TOPIC = "sim/data/20240341/S6" # Das spezifische Topic, das aufgezeichnet werden soll

def record_mqtt_to_csv(output_path: str, duration_seconds: int, max_rate_per_second: int | None = None):
    """
    Verbindet sich mit einem MQTT-Broker, zeichnet Nachrichten von einem spezifischen
    Topic für eine bestimmte Dauer auf und speichert sie in einer CSV-Datei.

    Args:
        output_path (str): Der vollständige Pfad zur CSV-Datei, in die gespeichert werden soll.
        duration_seconds (int): Die Dauer der Aufzeichnung in Sekunden.
        max_rate_per_second (int | None, optional): 
            Die maximale Anzahl an Nachrichten, die pro Sekunde gespeichert werden sollen.
            Wenn None (Standard), werden alle Nachrichten ohne Limit gespeichert.
    """
    if max_rate_per_second is None:
        print("Initialisiere MQTT-Rekorder... (Kein Ratenlimit, alle Daten werden erfasst)")
    else:
        print(f"Initialisiere MQTT-Rekorder... (Ratenlimit: max. {max_rate_per_second} Nachrichten/Sekunde)")
    
    received_messages = []
    
    # Variablen für die Ratenbegrenzung
    last_reset_time = time.time()
    messages_this_second = 0

    # --- Interne Callback-Funktionen ---
    def on_connect(client, userdata, flags, reason_code, properties):
        """Wird aufgerufen, wenn die Verbindung zum Broker steht."""
        if reason_code == 0:
            print(f"Erfolgreich mit MQTT Broker unter {MQTT_BROKER_IP} verbunden.")
            client.subscribe(MQTT_TOPIC)
            print(f"Abonniert auf Topic: '{MQTT_TOPIC}'")
        else:
            print(f"Verbindung fehlgeschlagen mit Code: {reason_code}")
            return

    def on_message(client, userdata, msg):
        """Wird bei jeder eingehenden Nachricht aufgerufen."""
        nonlocal last_reset_time, messages_this_second

        # Wenn kein Limit gesetzt ist, immer speichern
        if max_rate_per_second is None:
            pass # Direkt zur Speicherlogik übergehen
        else:
            current_time = time.time()
            # Prüfen, ob eine neue Sekunde begonnen hat
            if current_time - last_reset_time >= 1.0:
                last_reset_time = current_time
                messages_this_second = 0
            
            # Prüfen, ob das Limit für diese Sekunde erreicht ist
            if messages_this_second >= max_rate_per_second:
                return # Nachricht verwerfen und Funktion verlassen

            messages_this_second += 1

        # Logik zum Speichern der Nachricht
        try:
            data = json.loads(msg.payload.decode('utf-8'))
            data['recording_timestamp'] = datetime.now().isoformat()
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

    client.loop_start()

    print(f"--- Aufzeichnung für {duration_seconds} Sekunden gestartet ---")
    print("Drücken Sie Strg+C, um die Aufzeichnung vorzeitig zu beenden.")
    
    try:
        for i in range(duration_seconds, 0, -1):
            status_text = (
                f"Aufzeichnung läuft... "
                f"Verbleibende Zeit: {i-1}s | "
                f"Gespeicherte Nachrichten: {len(received_messages)}"
            )
            print(status_text + "   ", end='\r')
            time.sleep(1)
        print()

    except KeyboardInterrupt:
        print("\nAufzeichnung durch Benutzer vorzeitig beendet.")

    print("--- Aufzeichnung beendet ---")

    client.loop_stop()
    client.disconnect()
    print("Verbindung zum MQTT Broker getrennt.")

    if not received_messages:
        print("Keine Nachrichten zur Speicherung erfasst. Es wird keine CSV-Datei erstellt.")
        return

    print(f"Speichere {len(received_messages)} Nachrichten in '{output_path}'...")

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Sicherstellen, dass es Nachrichten gibt, bevor auf Schlüssel zugegriffen wird
        if not received_messages:
            print("Warnung: Keine Nachrichten zum Speichern vorhanden.")
            return
        headers = list(received_messages[0].keys())
        
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
            writer.writerows(received_messages)
            
        print(f"✅ CSV-Datei erfolgreich unter '{output_path}' gespeichert.")

    except Exception as e:
        print(f"Fehler beim Schreiben der CSV-Datei: {e}")


# --- Beispiel für die Ausführung ---
if __name__ == "__main__":
    AUFNAHMEDAUER_IN_SEKUNDEN = 60 * 60
    SPEICHERORT_CSV = "aufzeichnungen/mqtt_data_rate_limited.csv"
    
    # Geben Sie hier die maximale Aufnahmerate pro Sekunde an.
    # Für unbegrenzt: None oder den Parameter weglassen.
    # Beispiel: 10 = Maximal 10 Nachrichten pro Sekunde.
    AUFNAHMERATE_PRO_SEKUNDE = 2 
    
    # Rufe die Hauptfunktion mit der Ratenbegrenzung auf
    record_mqtt_to_csv(
        output_path=SPEICHERORT_CSV,
        duration_seconds=AUFNAHMEDAUER_IN_SEKUNDEN,
        max_rate_per_second=AUFNAHMERATE_PRO_SEKUNDE
    )

    # Beispiel für Aufruf ohne Limit (speichert alle Nachrichten)
    # print("\n--- Starte zweite Aufzeichnung ohne Ratenlimit ---")
    # record_mqtt_to_csv(
    #     output_path="aufzeichnungen/mqtt_data_full.csv",
    #     duration_seconds=60
    # )