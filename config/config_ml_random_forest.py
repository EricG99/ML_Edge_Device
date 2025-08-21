# config_ml_random_forest.py
# -----------------------------------------------------------------------------
# Random Forest Konfigurationen für die Experiment-Pipeline
# Erwartete Profil-Variablen laut experiment_pipeline.py:
#   - random_forest_server
#   - random_forest_edge
# -----------------------------------------------------------------------------

# Gemeinsame Defaults
_COMMON = {
    # Artefakt-Dateiname (wird von der Pipeline whitelisted)
    "model_filename": "model.joblib",
    # Datensatz & Ladestrategie
    "dataset": "mqtt_data_filtered.csv",
    "loading_strategy": "split",
    "train_fraction": 0.8,

    # Zeitreihen-Defaults (werden von der Experiment-Pipeline zur Laufzeit
    # per CLI-Grid überschrieben; diese Werte sind nur Fallbacks)
    "lags": 4,
    "horizon": 4,

    # Feature-Setup — nur die zwei gewünschten Spalten; Target zuerst!
    "base_features": ["Group4-2_S6_VolumetricFlowRate", "Group4-2_S6_MassFlowRate"],
    "time_features": [],
    "target_feature": "Group4-2_S6_VolumetricFlowRate",

    # Scaling-Flags (bei Bedarf von eurer FE/Trainer-Logik genutzt)
    "scale_other_features": False,
    "scale_target": False,
    "scaler_type": "robust",  # 'robust' oder 'minmax'

    # Sonstige Flags
    "edge_device": False,
    "enable_edge": False,
}

# -----------------------
# EDGE-Profil
# -----------------------
random_forest_edge = {
    **_COMMON,
    "model_name": "random_forest_edge",
    "edge_device": True,
    "enable_edge": True,

    # Beste Edge-Hyperparameter aus deiner Optimierung
    # (zusätzlich Top-Level für Rückwärtskompatibilität)
    "n_estimators": 202,
    "max_depth": 5,
    "min_samples_split": 7,
    "min_samples_leaf": 7,
    "max_features": 0.7185043122763849,
    "bootstrap": True,
    "n_jobs": 1,
    "random_state": 42,

    # Bevorzugt von Trainer/Runtime genutzt
    "model_params": {
        "n_estimators": 202,
        "max_depth": 5,
        "min_samples_split": 7,
        "min_samples_leaf": 7,
        "max_features": 0.7185043122763849,
        "bootstrap": True,
        "n_jobs": 1,
        "random_state": 42,
    },

    # Edge-Feature-Engineering bewusst schlank
    "include_roll_mean": True,
    "include_roll_std": False,
}

# -----------------------
# SERVER-Profil
# -----------------------
random_forest_server = {
    **_COMMON,
    "model_name": "random_forest_server",
    "edge_device": False,
    "enable_edge": False,

    # Beste Server-Hyperparameter aus deiner Optimierung
    "n_estimators": 563,
    "max_depth": None,
    "min_samples_split": 4,
    "min_samples_leaf": 5,
    "max_features": 0.43129532446351315,
    "bootstrap": True,
    "n_jobs": -1,
    "random_state": 42,

    "model_params": {
        "n_estimators": 563,
        "max_depth": None,
        "min_samples_split": 4,
        "min_samples_leaf": 5,
        "max_features": 0.43129532446351315,
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42,
    },

    # Server-FE: zusätzlich Std-Statistik zulassen
    "include_roll_mean": True,
    "include_roll_std": True,
}
