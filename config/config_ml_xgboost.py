# config/config_ml_xgboost.py
# -----------------------------------------------------------------------------
# XGBoost Konfigurationen im gleichen Stil wie die Random-Forest-Config
# Erwartete Profil-Variablen:
#   - xgboost_edge
#   - xgboost_server
# Die Aliase 'xgb_default' und 'xgboost' bleiben erhalten (zeigen auf Server).
# -----------------------------------------------------------------------------

_COMMON = {
    "model_filename": "model.joblib",
    "model_name": "xgboost_csv_split",
    "dataset": "mqtt_data_filtered.csv",
    "loading_strategy": "split",
    "train_fraction": 0.8,

    "lags": 4,
    "horizon": 4,

    "base_features": ["Group4-2_S6_VolumetricFlowRate", "Group4-2_S6_MassFlowRate"],
    "time_features": [],
    "target_feature": "Group4-2_S6_VolumetricFlowRate",

    "scale_other_features": True,
    "scale_target": False,
    "scaler_type": "robust",

    "inference_interval_sec": 1.0,

    "edge_device": False,
    "enable_edge": False,

    # Retraining/ES
    "xgb_additional_estimators": 200,
    "xgb_early_stopping_rounds": 20,
    "xgb_retrain_hist_rows": 5000,
}

xgboost_edge = {
    **_COMMON,
    "model_name": "xgboost_edge",
    "edge_device": True,
    "enable_edge": True,

    # Top-Level + verschachtelt unter 'xgb_params'
    "n_estimators": 381,
    "max_depth": 3,
    "learning_rate": 0.01341667278704857,
    "subsample": 0.6056676644840041,
    "colsample_bytree": 0.9432709566049063,
    "min_child_weight": 1,
    "gamma": 2.642968257622274,
    "reg_lambda": 2.5836066114108527,
    "reg_alpha": 0.00011240547738775451,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "objective": "reg:squarederror",

    "xgb_params": {
        "n_estimators": 381,
        "max_depth": 3,
        "learning_rate": 0.01341667278704857,
        "subsample": 0.6056676644840041,
        "colsample_bytree": 0.9432709566049063,
        "min_child_weight": 1,
        "gamma": 2.642968257622274,
        "reg_lambda": 2.5836066114108527,
        "reg_alpha": 0.00011240547738775451,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        "objective": "reg:squarederror",
    },

    "include_roll_mean": True,
    "include_roll_std": False,
}

xgboost_server = {
    **_COMMON,
    "model_name": "xgboost_server",
    "edge_device": False,
    "enable_edge": False,

    "n_estimators": 741,
    "max_depth": 6,
    "learning_rate": 0.01712074497782818,
    "subsample": 0.7265122358314671,
    "colsample_bytree": 0.6060047587276111,
    "min_child_weight": 12,
    "gamma": 4.609223360867172,
    "reg_lambda": 0.014589208100621021,
    "reg_alpha": 3.32758330659318e-06,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
    "objective": "reg:squarederror",

    "xgb_params": {
        "n_estimators": 741,
        "max_depth": 6,
        "learning_rate": 0.01712074497782818,
        "subsample": 0.7265122358314671,
        "colsample_bytree": 0.6060047587276111,
        "min_child_weight": 12,
        "gamma": 4.609223360867172,
        "reg_lambda": 0.014589208100621021,
        "reg_alpha": 3.32758330659318e-06,
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        "objective": "reg:squarederror",
    },

    "include_roll_mean": True,
    "include_roll_std": True,
}

# Aliase (Kompatibilität mit bestehendem Code)
xgb_default = { **xgboost_server }
xgboost = { **xgboost_server }
