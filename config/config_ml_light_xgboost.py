# config/config_ml_light_xgboost.py
# -----------------------------------------------------------------------------
# LightGBM Konfigurationen im gleichen Stil wie die Random-Forest-Config
# Erwartete Profil-Variablen:
#   - light_xgboost_edge
#   - light_xgboost_server
# Die Aliase 'light_xgboost' und 'Light_XGBoost' bleiben erhalten (zeigen auf Server).
# -----------------------------------------------------------------------------
from datetime import datetime

_now = datetime.now().strftime("%Y%m%d_%H%M%S")

_COMMON = {
    "model_filename": "model.joblib",
    "model_name": "Light_XGBoost",
    "dataset": "mqtt_data_filtered.csv",
    "loading_strategy": "split",
    "train_fraction": 0.8,

    "lags": 4,
    "horizon": 4,

    "base_features": ["Group4-2_S6_VolumetricFlowRate", "Group4-2_S6_MassFlowRate"],
    "time_features": [],
    "target_feature": "Group4-2_S6_VolumetricFlowRate",

    "scale_other_features": True,
    "scale_target": True,
    "scaler_type": "robust",

    "inference_interval_sec": 1.0,

    "edge_device": False,
    "enable_edge": False,

    # Retraining/ES (analog XGB)
    "lgbm_retrain_hist_rows": 5000,
    "lgbm_additional_estimators": 200,
    "lgbm_early_stopping_rounds": 0,

    # Metadaten
    "run_id": _now,
    "time_stamp": _now,
}

light_xgboost_edge = {
    **_COMMON,
    "model_name": "Light_XGBoost_Edge",
    "edge_device": True,
    "enable_edge": True,

    # Top-Level + verschachtelt unter 'lgbm_params'
    "n_estimators": 270,
    "learning_rate": 0.017791233812280628,
    "num_leaves": 19,
    "min_child_samples": 6,
    "subsample": 0.9062038935056944,
    "colsample_bytree": 0.6326381496292752,
    "reg_alpha": 0.047011771871281784,
    "reg_lambda": 0.05653426575923761,
    "max_bin": 219,
    "random_state": 42,
    "n_jobs": -1,

    "lgbm_params": {
        "n_estimators": 270,
        "learning_rate": 0.017791233812280628,
        "num_leaves": 19,
        "min_child_samples": 6,
        "subsample": 0.9062038935056944,
        "colsample_bytree": 0.6326381496292752,
        "reg_alpha": 0.047011771871281784,
        "reg_lambda": 0.05653426575923761,
        "max_bin": 219,
        "random_state": 42,
        "n_jobs": -1,
    },

    "include_roll_mean": True,
    "include_roll_std": False,
}

light_xgboost_server = {
    **_COMMON,
    "model_name": "Light_XGBoost_Server",
    "edge_device": False,
    "enable_edge": False,

    "n_estimators": 970,
    "learning_rate": 0.015550401298300518,
    "num_leaves": 242,
    "min_child_samples": 70,
    "subsample": 0.8107183654530435,
    "colsample_bytree": 0.7009668672307786,
    "reg_alpha": 0.002212547141374078,
    "reg_lambda": 0.015409214983086094,
    "max_bin": 321,
    "random_state": 42,
    "n_jobs": -1,

    "lgbm_params": {
        "n_estimators": 970,
        "learning_rate": 0.015550401298300518,
        "num_leaves": 242,
        "min_child_samples": 70,
        "subsample": 0.8107183654530435,
        "colsample_bytree": 0.7009668672307786,
        "reg_alpha": 0.002212547141374078,
        "reg_lambda": 0.015409214983086094,
        "max_bin": 321,
        "random_state": 42,
        "n_jobs": -1,
    },

    "include_roll_mean": True,
    "include_roll_std": True,
}

# Aliase
light_xgboost = { **light_xgboost_server }
Light_XGBoost = { **light_xgboost_server }
