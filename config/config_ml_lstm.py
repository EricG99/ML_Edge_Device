# config/config_ml_lstm.py

# Konfiguration für ein Standard-LSTM-Modell.
param_lstm_default = {
    "model_name": "LSTM_standard",
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
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": True,
    "scale_target": True,

    # LSTM-Modell-Parameter
    "num_layers": 2,
    "initial_units": 64,
    "dropout": 0.2,
    "epochs": 100,
    "batch_size": 32,
    "loss": "mse",
    "optimizer": "adam"
}

# Eine optimierte Konfiguration für den Einsatz auf einem Edge-Gerät.
# Kleinere Architektur, weniger Lags für schnellere Inferenz.
param_lstm_edge = {
    "model_name": "EDGE_LSTM",
    "dataset": "dataset2_m_v1.csv",
    "enable_edge": True, # Flag zum Speichern von Edge-Artefakten

    # Zeitreihen-Parameter
    "lags": 4,
    "horizon": 4,
    "train_fraction": 0.8,
    "rolling_window_size": 4,

    # Feature Engineering
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": [
        "minute_sin", "minute_cos", "hour_sin", "hour_cos",
        "day_of_week", "is_weekend"
    ],
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": True,
    "scale_target": True,

    # LSTM-Modell-Parameter
    "num_layers": 1,
    "initial_units": 32,
    "dropout": 0.1,
    "epochs": 50,
    "batch_size": 32,
    "loss": "mse",
    "optimizer": "adam"
}

param_lstm_test = {
    "model_name": "lstm",
    "dataset": "mqtt_data_rate_limited.csv",
    "model_filename": "model_quant_float16.tflite", # model_quant_float16.tflite Wichtig für das Speichern
    # "load_id": "2025-07-20_230601_4018_train", # Beispiel-ID

    "edge_device": True, #quamtisierung des Modells

    # Modellarchitektur
    "num_layers": 3,
    "initial_units": 128,
    "dropout": 0.2,

    "train_fraction": 0.3,

    
    # Training
    "epochs":  1,
    "batch_size": 32,
    "validation_fraction": 0.2,
    "early_stopping_patience": 10,
    "loss": "huber",    
    # Zeitreihenparameter
    "lags": 1,
    "horizon": 20,
    "train_fraction": 0.7,
    "rolling_window_size": 2,

    # Features
    "base_features": ['group4-2_s6_massflowrate'],
    "time_features": [],
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": True, 
    "scale_target": True,
    
    # Inferenz
    "inference_interval_sec": 1.0,
    
}

param_lstm_server = {
    **param_lstm_test,
    "num_layers": 3,
    "initial_units": 128,
    "epochs": 100,
    "batch_size": 64
}

lstm = {
    # --- Allgemeine Konfiguration ---
    "model_name": "lstm_csv_split",
    "dataset": "mqtt_data_rate_limited.csv", # Diese Datei wird für Training und Inferenz genutzt
    "model_filename": "model_quant_float16.tflite",

    # --- Laden & Aufteilen der Daten ---
    # "split": Lädt die CSV und teilt sie in Trainings- & Testdaten auf
    # "live_mqtt": Nutzt Live-Daten vom MQTT-Broker
    "loading_strategy": "split", 
    "train_fraction": 0.7, # 70% der CSV für Training, 30% für die anschließende Inferenz

    # --- Quantisierung ---
    "edge_device": True, # Aktiviert die TFLite-Quantisierung nach dem Training

    # --- Modellarchitektur ---
    "num_layers": 3,
    "initial_units": 128,
    "dropout": 0.2,

    # --- Trainingseinstellungen ---
    "epochs": 1,
    "batch_size": 32,
    "validation_fraction": 0.2, # 20% der Trainingsdaten werden zur Validierung während des Trainings genutzt
    "early_stopping_patience": 10,
    "loss": "huber",     

    # --- Zeitreihenparameter ---
    "lags": 1,
    "horizon": 20,
    "rolling_window_size": 2,

    # --- Feature Engineering ---
    "base_features": ['group4-2_s6_massflowrate'],
    "time_features": [],
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": True, 
    "scale_target": True,
    
    # --- Inferenz (wird im "split"-Modus nicht verwendet) ---
    "inference_interval_sec": 1.0,
}