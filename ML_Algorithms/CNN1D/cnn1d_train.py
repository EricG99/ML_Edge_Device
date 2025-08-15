
import logging
import sys
import os
from typing import Tuple

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ML_Helpfunctions.base_trainer import BaseTrainer
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline3D
from ML_Helpfunctions.cnn1d_utils import train_model_CNN1D

FOLDER_FLAG = "CNN1D"

class CNN1DTrainer(BaseTrainer):
    """Spezialisierte Trainingsklasse für 1D‑CNN Modelle."""
    
    def _setup_pipeline(self):
        return DataPipeline3D(self.config)

    def _train_model(self, X_train, y_train):
        self.model, history, self.train_time = train_model_CNN1D(
            config=self.config,
            X_train=X_train,
            y_train=y_train,
            features=self.features
        )

# Optional standalone run (dev): python -m ML_Algorithms.CNN1D.cnn1d_train
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("CNN1DTrainer kann über die Pipeline gestartet werden. Standalone-Run ist optional.")
