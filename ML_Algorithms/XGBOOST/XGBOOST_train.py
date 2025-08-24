
# ML_Algorithms/XGBOOST/XGBOOST_train.py
import logging, os, sys, numpy as np
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ML_Helpfunctions.base_trainer import BaseTrainer  # type: ignore
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline2D  # type: ignore
from ML_Helpfunctions import xgboost_utils as XGBUtils  # type: ignore

FOLDER_FLAG = "XGBOOST"

class XGBoostTrainer(BaseTrainer):
    """Trainer für XGBoost (2D-Tabular)."""

    def _setup_pipeline(self):
        return DataPipeline2D(self.config)

    def _train_model(self, X_train, y_train):
        # Multi-Output wird innerhalb von XGBUtils trainiert (falls Horizon > 1)
        model, train_time = XGBUtils.train_xgboost_model(self.config, X_train, y_train)
        self.model = model
        self.train_time = float(train_time)
        logging.info("XGBoost-Training abgeschlossen in %.2f s.", self.train_time)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("XGBoostTrainer: wird von pipeline_web_app.py genutzt.")
