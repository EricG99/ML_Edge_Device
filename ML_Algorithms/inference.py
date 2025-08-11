# ML_Algorithms/inference.py
import argparse
import logging
import sys
import os

# --- Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config_general import CONFIG_PATH, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS
from config.config_ml_random_forest import param_rf_test
from Random_Forest.rf_inference import run_standalone_inference as run_rf_standalone_inference
# from LSTM.lstm_inference import run_standalone_inference as run_lstm_standalone_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Zentrales Inferenz-Skript (Standalone).")
    parser.add_argument('--algorithm', type=str, required=True, choices=['random_forest', 'lstm'])
    args = parser.parse_args()

    logging.info(f"--- MODE: inference (standalone) | ALGORITHM: {args.algorithm} ---")
    
    # --- 1. Konfiguration zusammenbauen ---
    # Diese Logik kombiniert alle relevanten Config-Teile für die Inferenz
    base_config = param_rf_test.copy()
    base_config.update(CONFIG_LOAD_ARTIFACTS)
    base_config['paths'] = CONFIG_PATH['paths']
    
    # MQTT-Details
    broker_ip = MQTT_CONFIG['MQTT_BROKER_IP']
    port = MQTT_CONFIG['MQTT_PORT']
    topic = MQTT_CONFIG['MQTT_TOPIC']

    # --- 2. Spezifische Inferenz-Funktion aufrufen ---
    if args.algorithm == 'random_forest':
        run_rf_standalone_inference(config=base_config, broker_ip=broker_ip, port=port, topic=topic)
    
    elif args.algorithm == 'lstm':
        logging.warning("LSTM Standalone-Inferenz noch nicht implementiert.")
        # run_lstm_standalone_inference(...)
        pass
        
    else:
        logging.error(f"Unbekannter Algorithmus: {args.algorithm}")
        sys.exit(1)

if __name__ == "__main__":
    main()