# rf_train.py
import logging
import argparse
import joblib
import sys
import os

# --- Project Path Setup ---
# Stellt sicher, dass die Projekt-Root im Python-Pfad ist, um die Module zu finden
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
# NEU: Direkter Import der DataPipeline2D-Klasse
from ML_Helpfunctions.base_trainer import BaseTrainer
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline2D
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
from ML_Helpfunctions import RF_Utils as RFUtils

from config.config_ml_rf import param_rf_test
from config.config_general import CONFIG_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
FOLDER_FLAG = "RandomForest"


class RandomForestTrainer(BaseTrainer):
    """Spezialisierte Trainingsklasse für Random Forest Modelle."""
    
    def _setup_pipeline(self):
        """Initialisiert die 2D-Datenpipeline."""
        return DataPipeline2D(self.config)

    def _train_model(self, X_train, y_train):
        """Trainiert das Random Forest Modell."""
        self.model, self.train_time = RFUtils.train_random_forest_model(
            config=self.config,
            X_train=X_train,
            y_train=y_train,
            features=self.features
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Random Forest Model Trainer")
    # Hier könnten Argumente für Konfigurationsdateien etc. hinzugefügt werden
    args = parser.parse_args()

    # --- Basiskonfiguration (unverändert) ---
    training_config = {**CONFIG_PATH, **param_rf_test}
    training_config.update({
        "dataset": "train_data_sample.csv",
        "scaler_path": "trained_rf_scaler.joblib",
        "features_path": "trained_rf_features.joblib",
        "model_path": "trained_rf_model.joblib",
        # Wichtiger Schlüssel für die DataPipeline2D
        "loading_strategy": "live_mqtt" #"split", "separate_csv", "live_mqtt"
    })
    
    # Führe das Training aus und speichere die Artefakte standardmäßig
    trainer = RandomForestTrainer(config=training_config, folder_flag= FOLDER_FLAG)
    trainer.run(save_artifacts=True)  
    # run_training(config=training_config, save_artifacts=True)
    logging.info("\n✅ Training complete. Artifacts have been saved.")