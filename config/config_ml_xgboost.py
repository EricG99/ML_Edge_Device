# config/config_ml_xgboost.py

# Konfiguration für einen schnellen Testlauf von XGBoost.
param_xgb_test = {
    # Experimentinformationen
    "model_name": "xgb_test",
    "dataset": "train_data_sample.csv",
    "model_filename": "model.joblib",  # XGBoost wird oft als JSON gespeichert
    "load_id": "2025-07-21_214705_9106_train", # Wird zur Laufzeit gesetzt oder für Inferenz manuell eingetragen

    # Modellparameter für XGBoost
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "early_stopping_rounds": 25, # Wichtig für die Effizienz
    "random_state": 42,
    "n_jobs": -1,

    # Zeitreihenparameter
    "lags": 5,
    "horizon": 3,
    "train_fraction": 0.3,
    "rolling_window_size": 4,
    "validation_fraction": 0.2, # Wichtig für Early Stopping

    # Features
    "base_features": ['Group4-2_S6_MassFlowRate'],
    "time_features": ['hour', 'day_of_week'],
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": True,
    "scale_target": True,
}

# Eine robustere Konfiguration für das eigentliche Training auf dem Server.
param_xgb_server_train = {
    "model_name": "xgboost_server_trained",
    "dataset": "dataset3_m_v1.csv",
    "model_filename": "model.json",

    # Modellparameter
    "n_estimators": 2000,
    "max_depth": 6,
    "learning_rate": 0.02,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "gamma": 0.2,
    "early_stopping_rounds": 50,
    "random_state": 42,
    "n_jobs": -1,

    # Zeitreihenparameter
    "lags": 12,
    "horizon": 4,
    "train_fraction": 0.8,
    "rolling_window_size": 15,
    "validation_fraction": 0.15,

    # Feature-Konfiguration
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": [
        "minute_sin", "minute_cos", "hour_sin", "hour_cos",
        "day_of_month", "day_of_week", "is_weekend"
    ],
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": True,
    "scale_target": True,
}