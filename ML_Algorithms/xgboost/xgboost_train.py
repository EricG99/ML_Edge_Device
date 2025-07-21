# xgboost_train.py
import logging
import argparse
import sys
import os

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
from ML_Helpfunctions.base_trainer import BaseTrainer
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline2D
from ML_Helpfunctions import xgboost_utils as XGBoostUtils

# --- Configuration Imports ---
# NEU: Import der spezifischen XGBoost-Konfiguration
from config.config_ml_xgboost import param_xgb_test
from config.config_general import CONFIG_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FOLDER_FLAG = "XGBoost"

class XGBoostTrainer(BaseTrainer):
    """Spezialisierte Trainingsklasse für XGBoost Modelle."""
    
    def _setup_pipeline(self):
        """Initialisiert die 2D-Datenpipeline."""
        return DataPipeline2D(self.config)

    def _train_model(self, X_train, y_train):
        """Trainiert das XGBoost Modell."""
        self.model, self.train_time = XGBoostUtils.train_xgboost_model(
            config=self.config,
            X_train=X_train,
            y_train=y_train,
            features=self.features
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone XGBoost Model Trainer")
    args = parser.parse_args()
    
    # --- Konfigurationen zusammenführen ---
    # Wir starten mit der importierten Test-Konfiguration für XGBoost
    training_config = {**CONFIG_PATH, **param_xgb_test}
    
    # Allgemeine Einstellungen, die für diesen Lauf gelten sollen
    training_config.update({
        "inference_mode": "load_artifacts_path", 
        "loading_strategy": "live_mqtt"
    })
    
    # --- Training starten ---
    trainer = XGBoostTrainer(config=training_config, folder_flag=FOLDER_FLAG)
    trainer.run(save_artifacts=True)  
    
    logging.info("\n✅ XGBoost Training complete. Artifacts have been saved.")