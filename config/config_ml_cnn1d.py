
# config/config_ml_cnn1d.py
# -----------------------------------------------------------------------------
# Konfigurationen für ein 1D-CNN (Zeitsereien-Forecasting) in der bestehenden Pipeline
# Diese Datei folgt der Struktur von config_ml_lstm.py und ergänzt CNN-spezifische Parameter.
# -----------------------------------------------------------------------------

# ==========================
# Basiskonfigurationen (Profile)
# ==========================

param_cnn1d_default = {
    "model_name": "CNN1D_standard",
    "dataset": "dataset2_m_v1.csv",

    # Zeitreihen-Parameter
    "lags": 10,
    "horizon": 5,
    "train_fraction": 0.8,
    "rolling_window_size": 10,

    # Feature Engineering
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": [
        "minute_sin", "minute_cos", "hour_sin", "hour_cos",
        "day_of_week", "is_weekend", "month_sin", "month_cos"
    ],

    # Neue (einheitliche) FE-Flags
    "add_lag_features": True,
    "add_rolling_features": True,

    # Skalierung
    "scale_other_features": True,
    "scale_target": True,

    # CNN1D-Architektur
    "cnn_blocks": 2,             # Anzahl Conv-Blöcke
    "cnn_base_filters": 64,      # Start-Filter (pro Block halbiert)
    "cnn_kernel_size": 5,
    "cnn_dropout": 0.1,
    "cnn_activation": "relu",

    # Training
    "epochs": 100,
    "batch_size": 32,
    "validation_fraction": 0.2,
    "early_stopping_patience": 10,
    "loss": "huber",
    "optimizer": "adam",
}

# Edge-Profil: kleiner/fixer für Geräte mit wenig Ressourcen
param_cnn1d_edge = {
    "model_name": "EDGE_CNN1D",
    "dataset": "dataset2_m_v1.csv",
    "enable_edge": True,  # Flag zum Speichern/Export (z. B. TFLite, falls verfügbar)

    # Zeitreihen-Parameter
    "lags": 4,
    "horizon": 4,
    "train_fraction": 0.8,
    "rolling_window_size": 4,

    # Feature Engineering
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": ["minute_sin", "minute_cos", "hour_sin", "hour_cos", "day_of_week", "is_weekend"],
    "add_lag_features": True,
    "add_rolling_features": True,

    # Skalierung
    "scale_other_features": True,
    "scale_target": True,

    # CNN1D-Architektur (kleiner)
    "cnn_blocks": 2,
    "cnn_base_filters": 32,
    "cnn_kernel_size": 5,
    "cnn_dropout": 0.1,
    "cnn_activation": "relu",

    # Training
    "epochs": 50,
    "batch_size": 32,
    "validation_fraction": 0.2,
    "early_stopping_patience": 8,
    "loss": "huber",
    "optimizer": "adam",
}

# Kurzer Testlauf (schnell, kleine Daten/Parameter)
param_cnn1d_test = {
    "model_name": "cnn1d",
    "dataset": "mqtt_data_rate_limited.csv",
    "model_filename": "model_quant_float16.tflite",  # optional; falls Export aktiviert ist

    "edge_device": True,

    # Architektur
    "cnn_blocks": 2,
    "cnn_base_filters": 64,
    "cnn_kernel_size": 5,
    "cnn_dropout": 0.1,
    "cnn_activation": "relu",

    # Zeitreihen
    "lags": 1,
    "horizon": 20,
    "train_fraction": 0.7,
    "rolling_window_size": 2,

    # Training kurz
    "epochs": 1,
    "batch_size": 32,
    "validation_fraction": 0.2,
    "early_stopping_patience": 5,
    "loss": "huber",
    "optimizer": "adam",

    # Features
    "base_features": ['group4-2_s6_massflowrate'],
    "time_features": [],
    "add_lag_features": True,
    "add_rolling_features": True,
    "scale_other_features": True,
    "scale_target": True,

    # Inferenz
    "inference_interval_sec": 1.0,
}

# Server/Training ausführlicher
param_cnn1d_server = {
    **param_cnn1d_test,
    "cnn_base_filters": 128,
    "epochs": 100,
    "batch_size": 64
}

# ==========================
# Pipeline-Default (wie 'lstm' in LSTM-Config)
# ==========================
cnn1d = {
    # --- Allgemein ---
    "model_name": "cnn1d_csv_split",
    "dataset": "mqtt_data_filtered.csv",
    "model_filename": "model.keras",
    "loading_strategy": "split",
    "train_fraction": 0.8,
    "edge_device": True,  # optionaler Export (z. B. TFLite), wenn im Saver implementiert

    # --- CNN1D-Architektur ---
    "cnn_blocks": 2,
    "cnn_base_filters": 64,
    "cnn_kernel_size": 5,
    "cnn_dropout": 0.1,
    "cnn_activation": "relu",

    # --- Training ---
    "epochs": 20,
    "batch_size": 32,
    "validation_fraction": 0.2,
    "early_stopping_patience": 10,
    "loss": "huber",

    # --- Zeitreihen ---
    "lags": 4,
    "horizon": 4,
    "rolling_window_size": 2,

    # --- Feature Engineering ---
    "base_features": ['group4-2_s6_massflowrate'],
    "add_lag_features": True,
    "add_rolling_features": True,
    "scale_other_features": True,
    "scale_target": True,

    # --- Inferenz ---
    "inference_interval_sec": 1.0,
}

# =================================================================================
# Vollständige Laufzeit-Konfiguration (Beispiel für CNN1D)
# =================================================================================
full_runtime_config = {
    # --- Basis ---
    "model_name": "cnn1d_csv_split",
    "dataset": "mqtt_data_rate_limited.csv",
    "model_filename": "model_quant_float16.tflite",
    "loading_strategy": "split",
    "train_fraction": 0.7,
    "edge_device": True,

    # --- CNN1D ---
    "cnn_blocks": 2,
    "cnn_base_filters": 64,
    "cnn_kernel_size": 5,
    "cnn_dropout": 0.1,
    "cnn_activation": "relu",

    # --- Training ---
    "epochs": 1,
    "batch_size": 32,
    "validation_fraction": 0.2,
    "early_stopping_patience": 10,
    "loss": "huber",

    # --- Zeitreihe ---
    "lags": 1,
    "horizon": 20,
    "rolling_window_size": 2,

    # --- Features ---
    "base_features": ['Group4-2_S6_MassFlowRate'],
    "time_features": [],
    "add_lag_features": True,
    "add_rolling_features": True,
    "scale_other_features": True,
    "scale_target": True,

    # --- Inferenz- & Retraining-Steuerung ---
    "inference_interval_sec": 1.0,
    "inference_steps": 500,
    "retraining_cycles": 3,

    # --- Artefakt-Lade-Strategie ---
    "inference_mode": "load_artifacts_path",  # "load_artifacts_fast" oder "load_artifacts_path"

    # --- MQTT (nur falls loading_strategy = "live_mqtt") ---
    "MQTT_BROKER_IP": "192.168.0.101",
    "MQTT_PORT": 1883,
    "MQTT_TOPIC": "sim/data/20240341/S6",

    # --- Pfade (werden in der App evtl. erweitert) ---
    "paths": {
        "input": "C:/DEV/RevPi_ML/ML_Edge_Device/Input",
        "output": "C:/DEV/RevPi_ML/ML_Edge_Device/Output"
    }
}
