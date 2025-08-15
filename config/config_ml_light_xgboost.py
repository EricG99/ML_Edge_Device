# config/config_ml_light_xgboost.py
"""
Konfiguration für LightGBM ("Light_XGBoost").
Hinweis: Passe base_features/Features, Fenstergrößen und Pfade an dein Projekt an.
"""
from datetime import datetime

_now = datetime.now().strftime("%Y%m%d_%H%M%S")

light_xgboost = Light_XGBoost = {
    "model_name": "Light_XGBoost",
    "dataset": "data.csv",              # <- ggf. anpassen
    "base_features": ["target"],        # <- WICHTIG: Zielspaltenname aus deinem Datensatz
    "lags": 24,
    "horizon": 1,
    "loading_strategy": "split",        # "split" (Batch) oder "live_mqtt"
    "train_fraction": 0.7,
    "inference_interval_sec": 1.0,
    "max_fe_window": 150,
    "rolling_window_size": 24,
    "rolling_windows": [6, 12, 24],
    "scaler_type": "standard",          # "standard" | "minmax" | "robust"

    # LightGBM Hyperparameter (Beispielwerte)
    "lgbm_params": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": -1,
        "num_leaves": 63,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
        "random_state": 42,
        "n_jobs": -1,
    },

    # Retraining-Settings
    "lgbm_retrain_hist_rows": 5000,
    "lgbm_additional_estimators": 200,
    "lgbm_early_stopping_rounds": 0,

    # Pfad-/Run-Metadaten werden von config_general ergänzt
    "run_id": _now,
    "time_stamp": _now,
}
