# lstm_train_3d.py
import logging
import argparse
import sys
import os
import tensorflow as tf
import numpy as np
import joblib

# --- Project Path Setup ---
# Stellt sicher, dass das Hauptverzeichnis des Projekts im Python-Pfad ist
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
# Import der neuen 3D-Datenpipeline und der Hilfsfunktionen
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline3D, load_test_data_by_fraction
from ML_Helpfunctions import Pipeline_Utils
from ML_Helpfunctions import LSTM_Utils
from ML_Helpfunctions.base_trainer import BaseTrainer


# Import der Konfigurationen
from config.config_ml_lstm import lstm
# from config.config_general import CONFIG_PATH # Auskommentiert, falls nicht vorhanden

# Konfiguriere das Logging-Format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FOLDER_FLAG = "LSTM"



class LSTMTrainer(BaseTrainer):
    """Spezialisierte Trainingsklasse für LSTM Modelle."""
    
    def _setup_pipeline(self):
        """Initialisiert die 3D-Datenpipeline."""
        return DataPipeline3D(self.config)

    def _train_model(self, X_train, y_train):
        """Trainiert das LSTM Modell."""
        # Hinweis: Das Speichern der Artefakte für Keras-Modelle (history, etc.)
        # könnte hier noch verfeinert werden, indem die `history` zurückgegeben wird.
        self.model, history, self.train_time = LSTM_Utils.train_model_LSTM(
            config=self.config,
            X_train=X_train,
            y_train=y_train,
            features=self.features
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 3D LSTM Training Pipeline.")
    args = parser.parse_args()

    # --- Lade die Konfigurationen ---
    from config.config_general import CONFIG_PATH
    from config.config_ml_lstm import param_lstm_test

    logging.info("Merging general and model-specific configurations...")
    training_config = param_lstm_test.copy()
    
    # Die Basis-Pfade aus der allgemeinen Konfiguration holen
    training_config["paths"] = CONFIG_PATH.get("paths", {})
    
    # WICHTIG: Setze die Strategie für das Laden der Trainingsdaten
    # Dies ist für die DataPipeline3D entscheidend
    training_config["loading_strategy"] = "split" 

    # ==============================================================================
    # === ENTSCHEIDENDE KORREKTUR: Experiment-Setup explizit aufrufen ===
    # ==============================================================================
    # Diese Funktion erstellt die versionierten Ordner und gibt die vollständigen Pfade zurück
    logging.info("Setting up experiment directory structure...")
    exp_name = FOLDER_FLAG

    final_config, versioned_paths = Pipeline_Utils.setup_experiment(
        training_config, 
        exp_name, 
        run_type='train'
    )
    
    # Stelle sicher, dass die Konfiguration die neuen, vollständigen Pfade enthält
    final_config['paths'] = versioned_paths
    # ==============================================================================

    # Debug-Ausgabe zur Überprüfung der finalen Pfade
    logging.info(f"Final paths for experiment: {final_config['paths']}")
    
    # Führe die gesamte Pipeline mit der korrekten, vollständigen Konfiguration aus
    trainer = LSTMTrainer(config=final_config, folder_flag=FOLDER_FLAG)
    trainer.run(save_artifacts=True)