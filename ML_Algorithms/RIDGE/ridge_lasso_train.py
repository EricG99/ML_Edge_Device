# ML_Algorithms/RidgeLasso/ridge_lasso_train.py
import logging
import argparse
import os, sys
from ML_Helpfunctions.base_trainer import BaseTrainer
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline2D
from ML_Helpfunctions import pipeline_utils
from ML_Helpfunctions.ridge_lasso_utils import train_model as _train_model_impl

FOLDER_FLAG = "RIDGE_LASSO"

class RidgeLassoTrainer(BaseTrainer):
    """Trainer for Ridge/Lasso (2D pipeline, multi-output capable)."""
    def _setup_pipeline(self):
        return DataPipeline2D(self.config)

    def _train_model(self, X_train, y_train):
        self.model, self.train_time = _train_model_impl(self.config, X_train, y_train)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ridge/Lasso model (2D, multi-output).")
    parser.add_argument("--set", action="append", default=[], help="Override config entries: key=value")
    args = parser.parse_args()

    # Minimal bootstrapping: expect a config dict in environment or construct a small default
    from config.config_general import CONFIG_PATH  # project-specific
    base_cfg = {
        "model_name": "ridge",
        "algorithm": "ridge",
        "loading_strategy": "split",
        "horizon": 1,
        "lags": 1,
        "scale_other_features": True,
        "scale_target": False,
        "paths": CONFIG_PATH.get("paths", {}),
    }
    # apply CLI overrides
    for kv in args.set:
        if "=" in kv:
            k, v = kv.split("=", 1)
            # naive type casting
            if v.lower() in ("true","false"):
                v = v.lower() == "true"
            else:
                try:
                    v = float(v) if "." in v or "e" in v.lower() else int(v)
                except Exception:
                    pass
            base_cfg[k] = v

    # setup experiment folders & run
    final_config, paths = pipeline_utils.setup_experiment(base_cfg, FOLDER_FLAG, run_type='train')
    final_config["paths"] = paths

    trainer = RidgeLassoTrainer(config=final_config, folder_flag=FOLDER_FLAG)
    trainer.run(save_artifacts=True)
