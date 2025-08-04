# config/config_ml_rf.py

# Konfiguration für einen schnellen Testlauf (wenige Bäume, kleine Datenmenge).
# Ideal für Debugging und Funktionsüberprüfungen.
param_rf_test = {
    # Experimentinformationen
    "model_name": "rf_test",
    "dataset": "mqtt_data_rate_limited.csv",
    "model_filename": "model.joblib", # Wichtig für das Speichern
    "load_id": "2025-07-22_160652_7540_train",

    # Modellparameter
    "n_estimators": 10,
    "max_depth": 2,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": 1,
    

    # Zeitreihenparameter
    "lags": 1,
    "horizon": 1,
    "train_fraction": 0.8,
    "rolling_window_size": 2,
    # Features
    "base_features": ['group4-2_s6_massflowrate'],
    "time_features": [],
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": True, # Für RF oft nicht nötig, aber zur Konsistenz
    "scale_target": False,
}

# Eine robustere Konfiguration für das eigentliche Training auf dem Server.
param_rf_server_train = {
    "model_name": "random_forest_server_trained",
    "dataset": "dataset3_m_v1.csv",

    # Modellparameter
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1, # Alle verfügbaren CPU-Kerne nutzen

    # Zeitreihenparameter
    "lags": 8,
    "horizon": 4,
    "train_fraction": 0.8,
    "rolling_window_size": 10,

    # Feature-Konfiguration
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": [
        "minute_sin", "minute_cos", "hour_sin", "hour_cos",
        "day_of_month", "day_of_week", "is_weekend"
    ],
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": False,
    "scale_target": False,
}