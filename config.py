from pathlib import Path
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent

# Neue Ordner definieren
INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"
INPUT_EDGE = BASE_DIR / "input_data_edge_device"
OUTPUT_EDGE = BASE_DIR / "output_data_edge_device"
INPUT_SERVER = BASE_DIR / "input_data_server"
OUTPUT_SERVER = BASE_DIR / "output_data_server"

# Ordner erstellen (wenn nicht vorhanden)
for path in [INPUT_DIR, OUTPUT_DIR, INPUT_EDGE, OUTPUT_EDGE, INPUT_SERVER, OUTPUT_SERVER]:
    path.mkdir(parents=True, exist_ok=True)

# Konfiguration
CONFIG_PATH = {
    "paths": {
        "base": BASE_DIR,
        "input": INPUT_DIR,
        "output": OUTPUT_DIR,
        "input_data_edge_device": INPUT_EDGE,
        "output_data_edge_device": OUTPUT_EDGE,
        "input_data_server": INPUT_SERVER,
        "output_data_server": OUTPUT_SERVER,
    }
}

param_rf = {
    # Experimentinformationen
    "time_stamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    "model_name": "random_forest_server_train_small_revpi",
    "dataset": "dataset2_m_v1.csv",

    # Modellparameter (reduziert)
    "n_estimators": 10,             # Weniger Bäume → weniger Speicher
    "max_depth": 5,                 # Begrenzte Tiefe → kürzere Bäume
    "min_samples_split": 5,         # weniger Overfitting
    "min_samples_leaf": 3,          # stabilere Bäume
    "max_features": "sqrt",         # √n Auswahl bei Split → Standard
    "random_state": 42,
    "n_jobs": 1,                    # Kein Multithreading auf RevPi

    # Zeitreihenparameter
    "lags": 4,                      # Weniger Lags → weniger Features
    "horizon": 1,
    "train_fraction": 0.3,          # Weniger Trainingsdaten → weniger RAM
    "rolling_window_size": 5,       # Konsistent mit lags

    # # Feature-Konfiguration
    # "base_features": ['Volume_Flow'],
    # "time_features": [],
    # Feature Engineering
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": [
        'second', "minute", "minute_sin", "minute_cos", "hour", "hour_sin", "hour_cos",
        "day_of_month", "day_of_week", "is_weekend", "month", "month_sin", "month_cos"
    ],
    "include_roll_mean": True,
    "include_roll_std": False,      # Rolling-STD ist speicherintensiv
    "scale_other_features": False,  # Kein Scaler nötig für RF

    # Zielgrößen-Transformation
    "scale_target": False,

}

param_LSTM = {
    # Experiment Setup
    "time_stamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    "model_name": "test_1_lstm",
    "dataset": "dataset2_m_v1.csv",

    # Zeitreihen-Parameter
    "lags": 2,
    "horizon": 2,
    "train_fraction": 0.8,
    "rolling_window_size": 4,

    # Feature Engineering
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": [
        'second', "minute", "minute_sin", "minute_cos", "hour", "hour_sin", "hour_cos",
        "day_of_month", "day_of_week", "is_weekend", "month", "month_sin", "month_cos"
    ],
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": True,
    "scale_target": True,  # Wichtig für LSTM-Vorhersage

    # LSTM-Modell-Parameter
    "num_layers": 1,
    "initial_units": 32,
    "dropout": 0.1,
    "epochs": 1,
    "batch_size": 32,

}


param_LINREG = {
    "time_stamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    "model_name": "LinearModel",
    "model_type": "ridge",  # "linear", "ridge", "lasso"
    "alpha": 0.5,
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": [
        'second', "minute", "minute_sin", "minute_cos", "hour", "hour_sin", "hour_cos",
        "day_of_month", "day_of_week", "is_weekend", "month", "month_sin", "month_cos"
    ],
    "include_roll_mean": True,
    "include_roll_std": False,      # Rolling-STD ist speicherintensiv
    "scale_other_features": True,  # Kein Scaler nötig für RF
    "train_fraction": 0.8,
    "rolling_window_size": 4,
    "lags": 4,
    "horizon": 10,
    "scale_target": True,
    "scaler_type": "standard",
    "dataset": "dataset3_m_v1.csv",
    "run_id": "run01",
    "time_stamp": "2025-06-21"
}


param_rf_test = {
    # Experimentinformationen
    "model_name": "rf_test",
    "dataset": "dataset2_m_v1.csv",
    
    # Modellparameter (stark vereinfacht)
    "n_estimators": 5,              # Sehr wenige Bäume für schnellen Test
    "max_depth": 3,                 # Flache Bäume
    "random_state": 42,
    "n_jobs": 1,                    # Kein Multithreading
    
    # Zeitreihenparameter
    "lags": 2,                      # Nur 2 Verzögerungen
    "horizon": 1,                   # Kurzer Vorhersagehorizont
    "train_fraction": 0.3,
    "rolling_window_size": 4,          # Kleiner Trainingssplit
    
    # Features
    "base_features": ['Volume_Flow'],
    "time_features": ['hour', 'day_of_week'],  # Nur grundlegende Zeitfeatures
    "include_roll_mean": True,
    "include_roll_std": False,      # Rolling-STD ist speicherintensiv
    "scale_other_features": False,  # Kein Scaler nötig für RF

    # Zielgrößen-Transformation
    "scale_target": False,    # Falls Skalierung doch benötigt
}

param_linreg_test = {
    # Experimentinformationen
    "model_name": "linreg_test",
    "dataset": "dataset2_m_v1.csv",
    
    # Modellparameter
    "model_type": "ridge",          # Ridge für Stabilität
    "alpha": 0.1,                   # Geringe Regularisierung
    "train_fraction": 0.3,
    "rolling_window_size": 4,          # Gleicher Split wie RF
    
    # Zeitreihenparameter
    "lags": 2,                      # Konsistent mit RF
    "horizon": 1,
    
    # Features
    "base_features": ['Volume_Flow'],
    "time_features": ['hour', 'day_of_week'],  # Identisch zu RF
    "include_roll_mean": True,
    "include_roll_std": False,      # Rolling-STD ist speicherintensiv
    "scale_other_features": False,  # Kein Scaler nötig für RF

    # Zielgrößen-Transformation
    "scale_target": False,    # Falls Skalierung doch benötigt
}