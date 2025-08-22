# config/config_ml_lstm.py
# -----------------------------------------------------------------------------
# LSTM Konfigurationen im gleichen Stil wie die Random-Forest-Config
# Erwartete Profil-Variablen für die Pipeline:
#   - lstm_edge
#   - lstm_server
# Zusätzlich bleibt 'lstm' als Alias für Abwärtskompatibilität erhalten.
# -----------------------------------------------------------------------------

_COMMON = {
    # Artefakt-Dateiname
    "model_filename": "model.keras",
    # Datensatz & Ladestrategie
    "dataset": "mqtt_data_filtered.csv",
    "loading_strategy": "split",
    "train_fraction": 0.8,

    # Zeitreihen-Defaults
    "lags": 4,
    "horizon": 4,

    # Feature-Setup — Target zuerst!
    "base_features": ["Group4-2_S6_VolumetricFlowRate", "Group4-2_S6_MassFlowRate"],
    "time_features": [],
    "target_feature": "Group4-2_S6_VolumetricFlowRate",

    # Scaling
    "scale_other_features": True,
    "scale_target": True,
    "scaler_type": "robust",  # 'robust' oder 'minmax'

    # Inferenz
    "inference_interval_sec": 1.0,

    # Sonstige Flags
    "edge_device": False,
    "enable_edge": False,
}

# -----------------------
# EDGE-Profil (beste gefundenen Hyperparameter)
# -----------------------
lstm_edge = {
    **_COMMON,
    "model_name": "lstm_edge",
    "edge_device": True,
    "enable_edge": True,

    # Training/Architektur (Top-Level + in model_params für Rückwärtskompatibilität)
    "num_layers": 1,
    "initial_units": 45,
    "dropout": 0.3316009122167494,
    "batch_size": 64,
    "epochs": 52,
    "learning_rate": 0.0029252518249814905,
    "loss": "mse",
    "optimizer": "nadam",
    "clipnorm": 1.3892118636615982,
    "weight_decay": 3.0770036787863434e-06,

    # optionale Trainer-Flags
    "validation_fraction": 0.2,
    "early_stopping_patience": 10,

    "model_params": {
        "num_layers": 1,
        "initial_units": 45,
        "dropout": 0.3316009122167494,
        "batch_size": 64,
        "epochs": 52,
        "learning_rate": 0.0029252518249814905,
        "loss": "mse",
        "optimizer": "nadam",
        "clipnorm": 1.3892118636615982,
        "weight_decay": 3.0770036787863434e-06,
    },

    # FE-Flags analog RF-Style
    "include_roll_mean": True,
    "include_roll_std": False,
}

# -----------------------
# SERVER-Profil (beste gefundenen Hyperparameter)
# -----------------------
lstm_server = {
    **_COMMON,
    "model_name": "lstm_server",
    "edge_device": False,
    "enable_edge": False,

    "num_layers": 3,
    "initial_units": 108,
    "dropout": 0.121406912410838,
    "batch_size": 32,
    "epochs": 105,
    "learning_rate": 0.0021559307960495964,
    "loss": "mse",
    "optimizer": "rmsprop",
    "clipnorm": 3.4070802901915265,
    "weight_decay": 1.083605702492933e-05,

    "validation_fraction": 0.2,
    "early_stopping_patience": 10,

    "model_params": {
        "num_layers": 3,
        "initial_units": 108,
        "dropout": 0.121406912410838,
        "batch_size": 32,
        "epochs": 105,
        "learning_rate": 0.0021559307960495964,
        "loss": "mse",
        "optimizer": "rmsprop",
        "clipnorm": 3.4070802901915265,
        "weight_decay": 1.083605702492933e-05,
    },

    "include_roll_mean": True,
    "include_roll_std": True,
}

# Alias für Abwärtskompatibilität (bisher hieß das Profil 'lstm')
lstm = lstm_edge

param_lstm_edge = lstm_edge

param_lstm_test = lstm_edge
