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
from config.config_ml_lstm import param_lstm_test
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
    # 1. Importiere die allgemeine Basiskonfiguration (enthält die korrekten Pfade)
    from config.config_general import CONFIG_PATH
    # 2. Importiere die modellspezifischen LSTM-Parameter
    from config.config_ml_lstm import param_lstm_test

    logging.info("Merging general and model-specific configurations...")

    # --- Intelligente Verschmelzung der Konfigurationen ---
    # Starte mit einer Kopie der modellspezifischen Parameter
    training_config = param_lstm_test.copy()

    # Hole die allgemeinen Pfade und die spezifischen Pfade
    general_paths = CONFIG_PATH.get("paths", {})
    model_specific_paths = training_config.get("paths", {})

    # Verschmelze die Pfad-Wörterbücher.
    # Allgemeine Pfade dienen als Basis. Spezifische Pfade können sie überschreiben.
    # Wichtig: Der Schlüssel 'input' aus CONFIG_PATH bleibt so erhalten.
    merged_paths = {**general_paths, **model_specific_paths}

    # Setze das vollständig verschmolzene Pfad-Wörterbuch in der finalen Konfiguration
    training_config["paths"] = merged_paths
    training_config["inference_mode"] = "load_artifacts_path" # "load_artifacts_fast" oder "load_artifacts_path"


    # Debug-Ausgabe zur Überprüfung der finalen Pfade
    logging.info(f"Final paths for experiment: {training_config['paths']}")
    
    # Führe die gesamte Pipeline mit der korrekten Konfiguration aus
    # run_training_pipeline_3d(config=training_config)
    trainer = LSTMTrainer(config=training_config, folder_flag= FOLDER_FLAG)
    trainer.run(save_artifacts=True)