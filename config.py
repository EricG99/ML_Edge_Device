from pathlib import Path
from datetime import datetime


# Hauptpfade
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"

# Konfiguration
CONFIG_PATH = {
    "paths": {
        # Basis
        "base": BASE_DIR,
        "input": INPUT_DIR,
        "output": OUTPUT_DIR,

        # Input-Unterordner
        "input_data": INPUT_DIR / "Input_Data",
        "input_models": INPUT_DIR / "Input_Models",
        "input_scaler": INPUT_DIR / "Input_Scaler",

        # Fixer Output-Pfad (wird nicht verändert)
        "output_error_metrics": OUTPUT_DIR / "Error_Metrics"
    },
    "model_params": {
        "saved_model_name": "mein_modell.tflite",
        "scaler_name": "mein_scaler.pkl"
    }
}

# Input-Ordner erstellen
for key, path in CONFIG_PATH["paths"].items():
    if key.startswith("input") or key == "output_error_metrics":
        path.mkdir(parents=True, exist_ok=True)


param_LSTM_EDGE = {
    # Experiment Setup
    "time_stamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    "model_name": "EDGE_LSTM",
    "dataset": "dataset2_m_v1.csv",

    # Zeitreihen-Parameter
    "lags": 4,
    "horizon": 4,
    "train_fraction": 0.8,
    "rolling_window_size": 4,

    # Feature Engineering
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": [
        "minute_sin", "minute_cos", "hour_sin", "hour_cos",
        "day_of_month", "day_of_week", "is_weekend"
    ],
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": True,
    "scale_target": True,  # Wichtig für LSTM-Vorhersage

    # LSTM-Modell-Parameter
    "num_layers": 2,
    "initial_units": 32,
    "dropout": 0.1,
    "epochs": 50,
    "batch_size": 32,
}

param_rf = {
    # Experimentinformationen
    "time_stamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    "model_name": "random_forest_server_train_small_revpi",
    "dataset": "dataset3_m_v1.csv",

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
    "horizon": 4,
    "train_fraction": 0.3,          # Weniger Trainingsdaten → weniger RAM
    "rolling_window_size": 5,       # Konsistent mit lags

    # # Feature-Konfiguration
    # "base_features": ['Volume_Flow'],
    "time_features": [],
    # Feature Engineering
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    # "time_features": [
    #     'second', "minute", "minute_sin", "minute_cos", "hour", "hour_sin", "hour_cos",
    #     "day_of_month", "day_of_week", "is_weekend", "month", "month_sin", "month_cos"
    # ],
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

param_LSTM_EDGE = {
    # Experiment Setup
    "time_stamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    "model_name": "EDGE_LSTM",
    "dataset": "dataset2_m_v1.csv",

    # Zeitreihen-Parameter
    "lags": 4,
    "horizon": 4,
    "train_fraction": 0.8,
    "rolling_window_size": 4,

    # Feature Engineering
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": [
        "minute_sin", "minute_cos", "hour_sin", "hour_cos",
        "day_of_month", "day_of_week", "is_weekend"
    ],
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": True,
    "scale_target": True,  # Wichtig für LSTM-Vorhersage

    # LSTM-Modell-Parameter
    "num_layers": 2,
    "initial_units": 32,
    "dropout": 0.1,
    "epochs": 50,
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
    "horizon": 5,                   # Kurzer Vorhersagehorizont
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
    "dataset": "dataset3_m_v1.csv",
    
    # Modellparameter
    "model_type": "ridge",          # Ridge für Stabilität
    "alpha": 0.1,                   # Geringe Regularisierung
    "train_fraction": 0.3,
    "rolling_window_size": 4,          # Gleicher Split wie RF
    
    # Zeitreihenparameter
    "lags": 2,                      # Konsistent mit RF
    "horizon": 5,
    
    # Features
    "base_features": ['Volume_Flow'],
    "time_features": ['hour', 'day_of_week'],  # Identisch zu RF
    "include_roll_mean": True,
    "include_roll_std": False,      # Rolling-STD ist speicherintensiv
    "scale_other_features": False,  # Kein Scaler nötig für RF

    # Zielgrößen-Transformation
    "scale_target": False,    # Falls Skalierung doch benötigt
}

CONFIG_LSTM_ALL = {
    "paths": {
        "base": BASE_DIR,
        "input": INPUT_DIR,
        "output": OUTPUT_DIR,

        # Input Unterordner entsprechend deinem Beispiel
        "input_data": INPUT_DIR / "Input_Data",
        "input_models": INPUT_DIR / "Input_Models",
        "input_scaler": INPUT_DIR / "Input_Scaler",

        # Fixer Output-Pfad für Fehler-Metriken
        "output_error_metrics": OUTPUT_DIR / "Error_Metrics",

        # Falls du weitere Pfade brauchst, z.B. für Modelle / Loss Plots, kannst du sie hier ergänzen:
        "Models": OUTPUT_DIR / "Models",
        "Loss_Plots": OUTPUT_DIR / "Loss_Plots",
        "Model_Structures": OUTPUT_DIR / "Model_Structures",
    },

    "time_stamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    "model_name": "EDGE_LSTM",
    "model_name_edge_device": "Model_2025-06-26_105726_8186_EDGE_LSTM_dataset2_m_v1.csv_2025-06-26_105726_quantized.tflite",
    "scaler_name": "", 
    "dataset": "dataset2_m_v1.csv",

    "lags": 4,
    "horizon": 4,
    "train_fraction": 0.8,
    "rolling_window_size": 4,
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": [

    ],
    "edge_device": True,
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_target": True,
    "num_layers": 1,
    "initial_units": 32,
    "dropout": 0.1,
    "epochs": 1,
    "batch_size": 32,
}
