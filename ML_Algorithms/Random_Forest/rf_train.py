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

from config.config_ml_random_forest import random_forest
from config.config_general import CONFIG_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
FOLDER_FLAG = "RandomForest"


class RandomForestTrainer(BaseTrainer):
    """Spezialisierte Trainingsklasse für Random Forest Modelle."""
    
    def _setup_pipeline(self):
        """Initialisiert die 2D-Datenpipeline."""
        return DataPipeline2D(self.config)

    def _train_model(self, X_train, y_train):
        """Trainiert das Random Forest Modell (Multi-Output wird bei H>1 sichergestellt)."""
        import time
        import numpy as np
        import logging
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.multioutput import MultiOutputRegressor

        H = int(self.config.get("horizon", 1))
        rf_params = (self.config.get("rf_params") or {}).copy()

        # Safety: y als 2D-Array
        y_arr = np.asarray(y_train)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        # Modell bauen: bei mehrspaltigem y => MultiOutput, sonst je nach H
        base_rf = RandomForestRegressor(**rf_params)
        if y_arr.shape[1] > 1:
            model = MultiOutputRegressor(base_rf)
            logging.info("Random Forest: MultiOutputRegressor wird für horizon=%d verwendet.", y_arr.shape[1])
        else:
            model = base_rf if H == 1 else MultiOutputRegressor(base_rf)
            if H > 1 and y_arr.shape[1] == 1:
                logging.warning("RF-Training: y_train hat nur 1 Spalte, horizon=%d. "
                                "Verwende MultiOutputRegressor (semantisch sollten Trainingslabels (N,H) sein).", H)

        t0 = time.perf_counter()
        model.fit(X_train, y_arr)
        t1 = time.perf_counter()

        self.model = model
        self.train_time = t1 - t0

        logging.info("Random Forest-Modell Training abgeschlossen.")
        logging.info("Trainingszeit für Random Forest: %.2f Sekunden.", self.train_time)
        logging.info("Model type after training: %s", type(self.model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Random Forest Model Trainer")
    # Hier könnten Argumente für Konfigurationsdateien etc. hinzugefügt werden
    args = parser.parse_args()

    # --- Basiskonfiguration (unverändert) ---
    training_config = {**CONFIG_PATH, **param_rf_test}
    training_config.update({
        "dataset": "mqtt_data_rate_limited.csv",
        "scaler_path": "trained_rf_scaler.joblib",
        "features_path": "trained_rf_features.joblib",
        "model_path": "trained_rf_model.joblib",
        # Wichtiger Schlüssel für die DataPipeline2D
        "loading_strategy": "live_mqtt",  #"split", "separate_csv", "live_mqtt"
        "inference_mode": "load_artifacts_path"
    })
    
    # Führe das Training aus und speichere die Artefakte standardmäßig
    trainer = RandomForestTrainer(config=training_config, folder_flag= FOLDER_FLAG)
    trainer.run(save_artifacts=True)  
    # run_training(config=training_config, save_artifacts=True)
    logging.info("\n✅ Training complete. Artifacts have been saved.")