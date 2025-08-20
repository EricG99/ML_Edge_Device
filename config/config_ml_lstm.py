# config/config_ml_lstm.py

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
