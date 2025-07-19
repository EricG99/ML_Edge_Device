# Mqtt_Client.py
import paho.mqtt.client as mqtt
import logging
import json

# Konfiguriert das Logging-Format für eine klare Ausgabe.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MqttInferenceClient:
    """
    Eine wiederverwendbare MQTT-Client-Klasse.
    Sie verbindet sich mit einem Broker, abonniert ein Topic und ruft bei jeder
    eingehenden Nachricht eine übergebene Callback-Funktion auf.
    """
    def __init__(self, broker_ip: str, port: int, topic: str, on_message_callback: callable):
        """
        Initialisiert den MQTT-Client.

        Args:
            broker_ip (str): Die IP-Adresse des MQTT-Brokers.
            port (int): Der Port des MQTT-Brokers.
            topic (str): Das Topic, das abonniert werden soll.
            on_message_callback (callable): Eine Funktion, die bei Nachrichtenempfang aufgerufen wird.
                                            Sie erhält die dekodierte JSON-Payload als Argument.
        """
        self.broker_ip = broker_ip
        self.port = port
        self.topic = topic
        self.on_message_callback = on_message_callback

        # Initialisiert den Paho-MQTT-Client mit der neueren Callback-API-Version.
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """Callback, der beim Verbindungsaufbau zum Broker ausgeführt wird."""
        if reason_code == 0:
            logging.info(f"✅ Erfolgreich mit MQTT Broker unter {self.broker_ip} verbunden.")
            client.subscribe(self.topic)
            logging.info(f"👂 Abonniert auf Topic: '{self.topic}'")
        else:
            logging.error(f"❌ Verbindung fehlgeschlagen mit Code: {reason_code}")

    def _on_message(self, client, userdata, msg):
        """
        Interner Callback für den Nachrichtenempfang.
        Dekodiert die Payload und ruft den benutzerdefinierten Callback auf.
        """
        try:
            payload_str = msg.payload.decode('utf-8')
            data = json.loads(payload_str)
            if self.on_message_callback:
                self.on_message_callback(data)
        except json.JSONDecodeError:
            logging.warning("⚠️ Empfangene Nachricht ist kein gültiges JSON.")
        except Exception as e:
            logging.error(f"🔥 Ein unerwarteter Fehler in on_message ist aufgetreten: {e}", exc_info=True)

    def run(self):
        """
        Stellt die Verbindung zum Broker her und startet die Netzwerk-Schleife
        in einem nicht-blockierenden Hintergrund-Thread.
        """
        logging.info(f"🔌 Versuche, eine Verbindung zum MQTT Broker herzustellen: {self.broker_ip}:{self.port}...")
        try:
            self.client.connect(self.broker_ip, self.port, 60)
            self.client.loop_start()  # Startet den Netzwerk-Thread im Hintergrund
            logging.info("🚀 MQTT-Client läuft im Hintergrund.")
        except ConnectionRefusedError:
            logging.error("\n[FEHLER] Die Verbindung wurde verweigert. Überprüfen Sie IP, Port und Broker-Status.")
        except Exception as e:
            logging.critical(f"\nEin kritischer Fehler beim Starten des MQTT-Clients ist aufgetreten: {e}")
