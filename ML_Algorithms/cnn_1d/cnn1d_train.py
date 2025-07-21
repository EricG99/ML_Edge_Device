# cnn1d_train.py
import logging
import sys
import os

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
from ML_Helpfunctions.base_trainer import BaseTrainer
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline3D
from ML_Helpfunctions import cnn1d_utils as CNN1D_Utils

# --- Configuration Imports ---
from config.config_general import CONFIG_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Eindeutiger Ordner-Name für die Ergebnisse
FOLDER_FLAG = "CNN1D"

class CNN1DTrainer(BaseTrainer):
    """Spezialisierte Trainingsklasse für 1D-CNN Modelle."""
    
    def _setup_pipeline(self):
        """Initialisiert die 3D-Datenpipeline, perfekt für Sequenzmodelle wie CNNs."""
        return DataPipeline3D(self.config)

    def _train_model(self, X_train, y_train):
        """Trainiert das 1D-CNN Modell mithilfe der Logik aus den Utils."""
        self.model, self.history, self.train_time = CNN1D_Utils.train_model_cnn1d(
            config=self.config,
            X_train=X_train,
            y_train=y_train,
            features=self.features
        )

if __name__ == "__main__":
    # Beispielkonfiguration für 1D-CNN (normalerweise in einer separaten Datei)
    param_cnn1d_test = {
        'model_name': 'cnn1d_test',
        'dataset': "train_data_sample.csv",
        'model_filename': 'model.keras',
        
        # Modellarchitektur
        "num_conv_layers": 2,
        "filters": 64,
        "kernel_size": 3,
        "pool_size": 2,
        "dense_units": 100,
        "dropout": 0.2,

        # Trainingsparameter
        "epochs": 5, # Für einen schnellen Test
        "batch_size": 32,
        "loss": "mae",
        "validation_fraction": 0.2,
        
        # Zeitreihenparameter
        "lags": 10,
        "horizon": 1,
        "train_fraction": 0.3,
        "rolling_window_size": 4,
        
        # Features
        "base_features": ['Group4-2_S6_MassFlowRate'],
        "scale_target": True,
    }
    
    # --- Konfigurationen zusammenführen ---
    training_config = {**CONFIG_PATH, **param_cnn1d_test}
    training_config.update({
        "inference_mode": "load_artifacts_path", 
        "loading_strategy": "split"
    })
    
    # --- Training starten ---
    trainer = CNN1DTrainer(config=training_config, folder_flag=FOLDER_FLAG)
    trainer.run(save_artifacts=True)  
    
    logging.info(f"\n✅ {FOLDER_FLAG} Training complete. Artifacts have been saved.")