# ML_Algorithms/train.py
import argparse
import logging
import sys
import os



# --- Projektpfad-Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Allgemeine Konfigurationen ---
from config.config_general import CONFIG_PATH, generate_run_id

# --- Algorithmus-spezifische Module ---
from Random_Forest.rf_train import run_training as run_rf_training
# from LSTM.lstm_train import run_training as run_lstm_training # Beispiel für die Zukunft

# --- Algorithmus-spezifische Konfigurationen ---
from config.config_ml_random_forest import param_rf_test

from LSTM.lstm_train import run_training as run_lstm_training
from config.config_ml_lstm import param_lstm_test, param_lstm_server


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Zentrales Trainings-Skript für ML-Modelle.")
    parser.add_argument('--algorithm', type=str, required=True, choices=['random_forest', 'lstm'], help="Der zu trainierende Algorithmus.")
    parser.add_argument('--config', type=str, default='test', choices=['test', 'server'], help="Das zu verwendende Konfigurations-Set (z.B. 'test' oder 'server').")
    args = parser.parse_args()

    logging.info(f"--- MODE: train | ALGORITHM: {args.algorithm} | CONFIG: {args.config} ---")

    # --- 1. Konfiguration zusammenbauen ---
    base_config = CONFIG_PATH.copy()
    
    if args.algorithm == 'random_forest':
        # Wähle das RF-spezifische Parameter-Set
        algo_params = param_rf_test # Standardmäßig 'test'
        if args.config == 'server':
            # Hier könnte man die 'param_rf_server_train' laden
            # from config.config_ml.rf import param_rf_server_train
            # algo_params = param_rf_server_train
            pass
    elif args.algorithm == 'lstm':
    # Wähle das LSTM-spezifische Parameter-Set
        algo_params = param_lstm_test
        if args.config == 'server':
            algo_params = param_lstm_server
            
        # Kombiniere allgemeine und spezifische Konfigs
        base_config.update(algo_params)
        
        # Laufzeit-spezifische Werte hinzufügen
        run_id = generate_run_id()
        base_config['run_id'] = run_id
        base_config['time_stamp'] = run_id.split('_')[1]

        # Training durchführen
        run_lstm_training(config=base_config, save_artifacts=True)
        
        # Kombiniere allgemeine und spezifische Konfigs
        base_config.update(algo_params)
        
        # Laufzeit-spezifische Werte hinzufügen
        run_id = generate_run_id()
        base_config['run_id'] = run_id
        base_config['time_stamp'] = run_id.split('_')[1]

        # --- 2. Spezifische Trainingsfunktion aufrufen ---
        run_rf_training(config=base_config, save_artifacts=True)

    else:
        logging.error(f"Unbekannter Algorithmus: {args.algorithm}")
        sys.exit(1)

    logging.info("\n✅ Training abgeschlossen.")

if __name__ == "__main__":
    main()