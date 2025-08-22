# config/config_ml_cnn1d.py
# -----------------------------------------------------------------------------
# 1D-CNN Konfigurationen im gleichen Stil wie die Random-Forest-Config
# Erwartete Profil-Variablen:
#   - cnn1d_edge
#   - cnn1d_server
# Zusätzlich bleibt 'cnn1d' als Alias für Abwärtskompatibilität erhalten.
# -----------------------------------------------------------------------------

_COMMON = {
    "model_filename": "model.keras",
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
}

# -----------------------
# EDGE-Profil (beste gefundenen Hyperparameter)
# -----------------------
cnn1d_edge = {
    **_COMMON,
    "model_name": "cnn1d_edge",
    "edge_device": True,
    "enable_edge": True,

    # Architektur/Training
    "cnn_blocks": 1,
    "cnn_base_filters": 53,
    "cnn_kernel_size": 4,
    "cnn_dropout": 0.065602573828099,
    "cnn_activation": "relu",
    "batch_size": 64,
    "epochs": 45,
    "optimizer": "adam",
    "learning_rate": 0.0037156674895811818,
    "clipnorm": 1.919118551198806,

    "validation_fraction": 0.2,
    "early_stopping_patience": 10,
    "loss": "huber",  # Trainingsverlust (kann von Trainer überschrieben werden)

    "model_params": {
        "cnn_blocks": 1,
        "cnn_base_filters": 53,
        "cnn_kernel_size": 4,
        "cnn_dropout": 0.065602573828099,
        "cnn_activation": "relu",
        "batch_size": 64,
        "epochs": 45,
        "optimizer": "adam",
        "learning_rate": 0.0037156674895811818,
        "clipnorm": 1.919118551198806,
        "loss": "huber",
    },

    "include_roll_mean": True,
    "include_roll_std": False,
}

# -----------------------
# SERVER-Profil (beste gefundenen Hyperparameter)
# -----------------------
cnn1d_server = {
    **_COMMON,
    "model_name": "cnn1d_server",
    "edge_device": False,
    "enable_edge": False,

    "cnn_blocks": 3,
    "cnn_base_filters": 179,
    "cnn_kernel_size": 8,
    "cnn_dropout": 0.3526360292774605,
    "cnn_activation": "relu",
    "batch_size": 64,
    "epochs": 93,
    "optimizer": "adam",
    "learning_rate": 0.0011596536884349142,
    "clipnorm": 0.9715310388202569,
    "weight_decay": 2.4406586941580645e-05,

    "validation_fraction": 0.2,
    "early_stopping_patience": 10,
    "loss": "huber",

    "model_params": {
        "cnn_blocks": 3,
        "cnn_base_filters": 179,
        "cnn_kernel_size": 8,
        "cnn_dropout": 0.3526360292774605,
        "cnn_activation": "relu",
        "batch_size": 64,
        "epochs": 93,
        "optimizer": "adam",
        "learning_rate": 0.0011596536884349142,
        "clipnorm": 0.9715310388202569,
        "weight_decay": 2.4406586941580645e-05,
        "loss": "huber",
    },

    "include_roll_mean": True,
    "include_roll_std": True,
}

# Alias für Abwärtskompatibilität (bisher hieß das Profil 'cnn1d')
cnn1d = cnn1d_edge
