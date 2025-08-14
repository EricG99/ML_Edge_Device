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
    "dataset": "mqtt_data_filtered.csv",
    "model_filename": "model.keras",

    #"model_filename": "model_quant_float16.tflite",
    "loading_strategy": "split",
    "train_fraction": 0.8,
    "edge_device": True,

    # --- Modellarchitektur ---
    "num_layers": 3,
    "initial_units": 128,
    "dropout": 0.2,

    # --- Trainingseinstellungen (OPTIMIERT) ---
    "epochs": 20,  # ERHÖHT: Geben Sie dem Modell mehr Zeit zum Lernen.
    "batch_size": 32,
    "validation_fraction": 0.2,
    "early_stopping_patience": 10,
    "loss": "huber",

    # --- Zeitreihenparameter ---
    "lags": 4,
    "horizon": 4,
    "rolling_window_size": 2,

    # --- Feature Engineering (OPTIMIERT) ---
    "base_features": ['group4-2_s6_massflowrate'],
    
    # Aktiviert die Erstellung von 3 Lag-Features (lag_1, lag_2, lag_3)
    "add_lag_features": True,
    
    # Aktiviert die Erstellung von roll. Mittelwert und Standardabweichung
    "add_rolling_features": True,
    
    "scale_other_features": True,
    "scale_target": True,
    
    # --- Inferenz ---
    "inference_interval_sec": 1.0,
}
# =================================================================================
# Vollständige Laufzeit-Konfiguration (Beispiel für LSTM)
# =================================================================================
# Dies ist die kombinierte Konfiguration, die nach dem Starten der App verwendet wird.
full_runtime_config = {
    
    # --- Basis-Konfiguration (aus config/config_ml_lstm.py) ---
    "model_name": "lstm_csv_split",
    "dataset": "mqtt_data_rate_limited.csv",
    "model_filename": "model_quant_float16.tflite",
    "loading_strategy": "split", 
    "train_fraction": 0.7,
    "edge_device": True,
    "num_layers": 3,
    "initial_units": 128,
    "dropout": 0.2,
    "epochs": 1,
    "batch_size": 32,
    "validation_fraction": 0.2,
    "early_stopping_patience": 10,
    "loss": "huber",
    "lags": 1,
    "horizon": 20,
    "rolling_window_size": 2,
    "base_features": ['Group4-2_S6_MassFlowRate'],
    "time_features": [],
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": True, 
    "scale_target": True,
    
    # --- Inferenz- & Retraining-Steuerung ---
    "inference_interval_sec": 1.0, # Zeit zwischen den Schritten im MQTT-Modus
    # NEU: Anzahl der Inferenzschritte im "--no-retraining" Modus (wird bei MQTT genutzt)
    "inference_steps": 500,
    # NEU: Anzahl der Zyklen (Datensammeln + Nachtrainieren) im "--retraining" Modus
    "retraining_cycles": 3,

    # --- Artefakt-Ladekonfiguration (aus config_general.py -> CONFIG_LOAD_ARTIFACTS) ---
    # Definiert, wie trainierte Modelle und Scaler für die Inferenz geladen werden.
    "inference_mode": "load_artifacts_path", # Mögliche Werte: "load_artifacts_path", "load_artifacts_fast"

    # --- MQTT-Konfiguration (aus config_general.py -> MQTT_CONFIG) ---
    # Diese Werte werden verwendet, wenn loading_strategy = "live_mqtt" ist.
    "MQTT_BROKER_IP": "192.168.0.101", # Beispiel-IP aus Ihren Logs
    "MQTT_PORT": 1883,
    "MQTT_TOPIC": "sim/data/20240341/S6", # Beispiel-Topic aus Ihren Logs

    # --- Pfad-Konfiguration (aus config_general.py -> CONFIG_PATH) ---
    # Die Basispfade für alle Ein- und Ausgaben.
    "paths": {
        "input": "C:/DEV/RevPi_ML/ML_Edge_Device/Input",   # Beispielpfad
        "output": "C:/DEV/RevPi_ML/ML_Edge_Device/Output"  # Beispielpfad
        # Weitere Pfade wie /Models, /Scalers etc. werden zur Laufzeit dynamisch hinzugefügt.
    }
}