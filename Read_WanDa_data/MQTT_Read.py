import paho.mqtt.client as mqtt
import time

# --- Konfiguration ---
# Bitte passen Sie diese Werte an Ihre MQTT-Broker-Konfiguration an.
MQTT_BROKER_IP = "192.168.0.101"  # IP-Adresse oder Hostname Ihres MQTT-Brokers
MQTT_PORT = 1883                   # Standard-MQTT-Port
MQTT_TOPIC = "sim/data/20240341/S6"          

# NEU: Abtastrate in Sekunden. Eine Nachricht wird nur alle X Sekunden ausgegeben.
SAMPLING_INTERVAL_SECONDS = 1.0    # Beispiel: Alle 1.0 Sekunden eine Nachricht ausgeben

# --- Globale Variable für das Timing ---
last_print_time = 0

# --- Callback-Funktionen ---

def on_connect(client, userdata, flags, reason_code, properties):
    """
    Diese Funktion wird aufgerufen, wenn der Client eine CONNACK-Antwort vom Server erhält.
    """
    if reason_code == 0:
        print(f"Erfolgreich mit MQTT Broker unter {MQTT_BROKER_IP} verbunden.")
        # Abonniere das gewünschte Topic nach erfolgreicher Verbindung
        client.subscribe(MQTT_TOPIC)
        print(f"Abonniert auf Topic: '{MQTT_TOPIC}'")
        print(f"Abtastrate eingestellt auf: {SAMPLING_INTERVAL_SECONDS} Sekunden")
    else:
        print(f"Verbindung fehlgeschlagen mit Code: {reason_code}")
        # Mögliche Fehlercodes:
        # 1: Falsche Protokollversion
        # 2: Ungültige Client-ID
        # 3: Server nicht verfügbar
        # 4: Falscher Benutzername oder Passwort
        # 5: Nicht autorisiert

import json # Stellen Sie sicher, dass json importiert ist
import time


def on_message(client, userdata, msg):
    """
    Diese Funktion wird aufgerufen, wenn eine Nachricht empfangen wird.
    Sie liest jetzt spezifische Werte aus der JSON-Payload aus.
    """
    global last_print_time
    current_time = time.time()

    if (current_time - last_print_time) >= SAMPLING_INTERVAL_SECONDS:
        try:
            # 1. Dekodiere die Nachricht in einen String
            payload_str = msg.payload.decode('utf-8')
            
            # 2. Wandle den JSON-String in ein Python-Dictionary um
            data = json.loads(payload_str)
            
            # 3. Greife auf die gewünschten Werte über ihre Schlüssel zu
            mass_flow = data.get("Group4-2_S6_MassFlowRate", "N/A") # .get() ist sicherer, falls der Schlüssel fehlt
            datetime_val = data.get("datetime", "N/A")
            pressure = data.get("Group4-2_S6_Pressure", "N/A")

            # 4. Gib nur die ausgelesenen Werte aus
            print("--- Spezifische Daten extrahiert ---")
            print(f"Zeitstempel (aus Daten): {datetime_val}")
            print(f"Massenfluss:              {mass_flow}")
            print(f"Druck:                    {pressure}")
            print("------------------------------------")
            
            # Aktualisiere den Zeitpunkt der letzten Ausgabe
            last_print_time = current_time

        except json.JSONDecodeError:
            print("Fehler: Die empfangene Nachricht ist kein gültiges JSON.")
        except KeyError as e:
            print(f"Fehler: Der Schlüssel {e} wurde in der Nachricht nicht gefunden.")
        except Exception as e:
            print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")


# --- Hauptskript ---

if __name__ == "__main__":
    # Erstelle eine neue MQTT-Client-Instanz mit der neueren Callback-API
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    # Weise die Callback-Funktionen zu
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Versuche, eine Verbindung zum MQTT Broker herzustellen: {MQTT_BROKER_IP}:{MQTT_PORT}...")

    try:
        # Stelle die Verbindung zum Broker her
        client.connect(MQTT_BROKER_IP, MQTT_PORT, 60)

        # loop_forever() ist eine blockierende Funktion, die den Client am Laufen hält,
        # um auf Nachrichten zu lauschen und die Verbindung aufrechtzuerhalten.
        # Das Skript läuft, bis es manuell beendet wird (z.B. mit Strg+C).
        print("Warte auf Nachrichten... (Beenden mit Strg+C)")
        client.loop_forever()

    except ConnectionRefusedError:
        print("\n[FEHLER] Die Verbindung wurde verweigert. Bitte überprüfen Sie:")
        print(f"  1. Ist die IP-Adresse des Brokers ('{MQTT_BROKER_IP}') korrekt?")
        print("  2. Läuft der MQTT-Broker und ist er im Netzwerk erreichbar?")
        print(f"  3. Ist der Port ('{MQTT_PORT}') korrekt und nicht durch eine Firewall blockiert?")
    except KeyboardInterrupt:
        print("\nSkript durch Benutzer beendet. Trenne die Verbindung...")
        client.disconnect()
        print("Verbindung getrennt. Auf Wiedersehen!")
    except Exception as e:
        print(f"\nEin unerwarteter Fehler ist aufgetreten: {e}")
