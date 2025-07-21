# config/config_ml_cnn1d.py

# Konfiguration für einen schnellen Testlauf des 1D-CNN-Modells.
param_cnn1d_test = {
    # Experimentinformationen
    "model_name": "cnn1d_test",
    "dataset": "train_data_sample.csv",
    "model_filename": "model.keras", # Keras-Modelle werden oft mit .keras oder .h5 gespeichert
    "load_id": None, # Wird zur Laufzeit gesetzt oder für Inferenz manuell eingetragen

    # Modellarchitektur
    "num_conv_layers": 2,
    "filters": 32,
    "kernel_size": 3,
    "pool_size": 2,
    "dense_units": 64,
    "dropout": 0.2,

    # Trainingsparameter
    "epochs": 10, # Wenige Epochen für einen schnellen Test
    "batch_size": 32,
    "loss": "mae",
    "optimizer": "adam",
    "metrics": ["mse"],
    "validation_fraction": 0.2,
    "use_early_stopping": True, # Callbacks können hier gesteuert werden
    
    # Zeitreihenparameter
    "lags": 10,
    "horizon": 1,
    "train_fraction": 0.3,
    "rolling_window_size": 4,
    
    # Features
    "base_features": ['Group4-2_S6_MassFlowRate'],
    "scale_target": True,
}

# Eine robustere Konfiguration für das eigentliche Training auf dem Server.
param_cnn1d_server_train = {
    # Experimentinformationen
    "model_name": "cnn1d_server_trained",
    "dataset": "dataset3_m_v1.csv",
    "model_filename": "model.keras",
    
    # Modellarchitektur
    "num_conv_layers": 3,
    "filters": 128,
    "kernel_size": 5,
    "pool_size": 2,
    "dense_units": 256,
    "dropout": 0.3,

    # Trainingsparameter
    "epochs": 100,
    "batch_size": 64,
    "loss": "huber_loss",
    "optimizer": "adam",
    "metrics": ["mae", "mape"],
    "validation_fraction": 0.15,
    "use_early_stopping": True,
    "early_stopping_patience": 15,
    
    # Zeitreihenparameter
    "lags": 24, # Längeres Fenster für mehr Kontext
    "horizon": 4,
    "train_fraction": 0.8,
    "rolling_window_size": 15,
    
    # Feature-Konfiguration
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": [
        "minute_sin", "minute_cos", "hour_sin", "hour_cos",
        "day_of_month", "day_of_week", "is_weekend"
    ],
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_target": True,
}