import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def add_time_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    selected_features = config.get("time_features", [])
    feature_dict = {"time": []}

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")

    if "second" in selected_features:
        df["second"] = df.index.second
        feature_dict["time"].append("second")
    if "minute" in selected_features:
        df["minute"] = df.index.minute
        feature_dict["time"].append("minute")
    if "hour" in selected_features:
        df["hour"] = df.index.hour
        feature_dict["time"].append("hour")
    if "day_of_month" in selected_features:
        df["day_of_month"] = df.index.day
        feature_dict["time"].append("day_of_month")
    if "day_of_week" in selected_features:
        df["day_of_week"] = df.index.dayofweek
        feature_dict["time"].append("day_of_week")
    if "is_weekend" in selected_features:
        df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
        feature_dict["time"].append("is_weekend")
    if "month" in selected_features:
        df["month"] = df.index.month
        feature_dict["time"].append("month")

    # Zyklische Transformationen
    if "minute_sin" in selected_features:
        df["minute_sin"] = np.sin(2 * np.pi * df.index.minute / 60)
        feature_dict["time"].append("minute_sin")
    if "minute_cos" in selected_features:
        df["minute_cos"] = np.cos(2 * np.pi * df.index.minute / 60)
        feature_dict["time"].append("minute_cos")
    if "hour_sin" in selected_features:
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        feature_dict["time"].append("hour_sin")
    if "hour_cos" in selected_features:
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        feature_dict["time"].append("hour_cos")
    if "month_sin" in selected_features:        
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        feature_dict["time"].append("month_sin")
    if "month_cos" in selected_features:
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
        feature_dict["time"].append("month_cos")
    if "dayofweek_sin" in selected_features:
        df["dayofweek_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        feature_dict["time"].append("dayofweek_sin")
    if "dayofweek_cos" in selected_features:
        df["dayofweek_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        feature_dict["time"].append("dayofweek_cos")

    return df, feature_dict


def add_lag_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    base_features = config["base_features"]
    max_lag = config["lags"]
    feature_dict = {"lagged": []}

    for feature in base_features:
        for lag in range(1, max_lag + 1):
            lagged_name = f'{feature}_lag_{lag}'
            df[lagged_name] = df[feature].shift(lag)
            feature_dict["lagged"].append(lagged_name)

    return df, feature_dict


def add_rolling_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    base_features = config["base_features"]
    window_size = config["rolling_window_size"]
    include_roll_mean = config["include_roll_mean"]
    include_roll_std = config["include_roll_std"]

    feature_dict = {"rolling": []}

    for feature in base_features:
        if include_roll_mean:
            mean_name = f'{feature}_roll_mean_{window_size}'
            df[mean_name] = df[feature].rolling(window=window_size).mean().shift(1)
            feature_dict["rolling"].append(mean_name)
        if include_roll_std:
            std_name = f'{feature}_roll_std_{window_size}'
            df[std_name] = df[feature].rolling(window=window_size).std().shift(1)
            feature_dict["rolling"].append(std_name)

    return df, feature_dict


def add_all_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    feature_dict = {
        "base": config["base_features"].copy(),
        "lagged": [],
        "rolling": [],
        "time": [],
    }

    # Zeit-Features
    df, time_dict = add_time_features(df, config)
    feature_dict["time"] = time_dict["time"]

    # Lag-Features
    df, lag_dict = add_lag_features(df, config)
    feature_dict["lagged"] = lag_dict["lagged"]

    # Rolling-Features
    df, roll_dict = add_rolling_features(df, config)
    feature_dict["rolling"] = roll_dict["rolling"]

    # Entferne Zeilen mit NaNs (durch shift/rolling)
    df = df.dropna()

    # Alle Features zusammenführen
    all_features = (
        feature_dict["base"]
        + feature_dict["lagged"]
        + feature_dict["rolling"]
        + feature_dict["time"]
    )
    feature_dict["all"] = all_features

    return df, feature_dict


def create_feature_list_from_dict(feature_dict: dict) -> list:
    return feature_dict["all"]
# LSTM_Utils.py

import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf

import paramiko

from typing import Tuple, List
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import Load_Prepare_Data as LoadPrepareData
import Pipeline_Utils as PipelineUtils

def prepare_data_for_lstm(config: dict):
    """
    Bereitet die 3D- und 2D-Daten für LSTM-Modelle vor.
        Returns:
        X_train_3D, y_train_3D: Eingaben fürs LSTM (3D)
        X_train_2D, y_train_2D: Flache 2D-Version für klassische Modelle / Metriken
        X_test_3D, y_test_3D: Testdaten für LSTM (3D)
        X_test_2D, y_test_2D: Flache Testdaten
        scaler_3D: Featurescaler (für X 3D)
        scaler_2D: Featurescaler (für X 2D)
        y_scaler: Skaler für y (falls separat skaliert)
        train_df, test_df: Originale DataFrames
        train_features_dict: Dictionary mit Feature-Gruppen
        full_feature_list: Finale Featureliste (z. B. für Modell oder Export)
    
    """
    (
        X_train_3D, y_train_3D,
        X_test_3D, y_test_3D,
        scaler_3D, y_scaler,
        train_df, test_df,
        train_features_dict, full_feature_list
    ) = LoadPrepareData._prepare_base_data_3D(config)

    (
        X_train_2D, y_train_2D,
        X_test_2D, y_test_2D,
        scaler_2D, y_scaler_2D, *_
    ) = LoadPrepareData._prepare_base_data_2D(config)

    y_scaler = y_scaler_2D or y_scaler

    return (
        X_train_3D, y_train_3D,
        X_train_2D, y_train_2D,
        X_test_3D, y_test_3D,
        X_test_2D, y_test_2D,
        scaler_3D, scaler_2D, y_scaler,
        train_df, test_df, train_features_dict, full_feature_list
    )

def build_dynamic_lstm(input_shape: Tuple[int, int],
                       num_layers: int = 1,
                       initial_units: int = 64,
                       dropout: float = 0.1,
                       forecast_horizon: int = 1) -> Sequential:
    """
    Dynamisch anpassbares LSTM-Modell für Zeitreihen.

    Args:
        input_shape (tuple): (lags, n_features)
        num_layers (int): Anzahl der LSTM-Schichten
        initial_units (int): Anzahl Units in der ersten Schicht
        dropout (float): Dropout-Rate
        forecast_horizon (int): Ziel-Ausgabeschritte

    Returns:
        tf.keras.Sequential: Keras LSTM Modell
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    units = initial_units

    for i in range(num_layers):
        return_seq = i < num_layers - 1
        model.add(LSTM(units, return_sequences=return_seq))
        model.add(Dropout(dropout))
        model.add(BatchNormalization())
        units = max(units // 2, 4)

    model.add(Dense(forecast_horizon, activation='relu'))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model_LSTM(config: dict, X_train: np.ndarray,
                     y_train: np.ndarray, features: List[str]):
    
    """
    Baut, kompiliert und trainiert ein LSTM-Modell mit zeitbasiertem Validation-Split.

    Args:
        config (dict): Konfigurationsparameter (lags, num_layers, epochs, etc.).
        X_train (np.ndarray): Trainingsdaten (3D: [samples, lags, features]).
        y_train (np.ndarray): Zielwerte (2D: [samples, horizon]).
        features (list): Liste der Feature-Namen.

    Returns:
        tuple: (model, history, train_time)
    """

    input_shape_lstm = (config["lags"], len(features))
    model = build_dynamic_lstm(
        input_shape=input_shape_lstm,
        num_layers=config.get("num_layers", 1),
        initial_units=config.get("initial_units", 64),
        dropout=config.get("dropout", 0.1),
        forecast_horizon=config["horizon"]
    )

    loss_function = config.get("loss", tf.keras.losses.Huber())
    optimizer = config.get("optimizer", "adam")
    model.compile(optimizer=optimizer, loss=loss_function, metrics=config.get("metrics", ["mae"]))

    val_fraction = config.get("validation_fraction_keras", 0.2)
    if val_fraction > 0 and X_train.shape[0] > 10:
        split_index = int((1 - val_fraction) * len(X_train))
        val_data = (X_train[split_index:], y_train[split_index:])
        X_fit, y_fit = X_train[:split_index], y_train[:split_index]
    else:
        X_fit, y_fit = X_train, y_train
        val_data = None

    callbacks = []
    if config.get("use_early_stopping", True):
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=config.get("early_stopping_monitor", "val_loss"),
            patience=config.get("early_stopping_patience", 10),
            restore_best_weights=True
        ))
    if config.get("use_reduce_lr_on_plateau", True):
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor=config.get("lr_plateau_monitor", "val_loss"),
            factor=config.get("lr_factor", 0.5),
            patience=config.get("lr_patience", 3)
        ))

    start = time.time()
    history = model.fit(
        X_fit, y_fit,
        validation_data=val_data,
        epochs=config.get("epochs", 10),
        batch_size=config.get("batch_size", 32),
        callbacks=callbacks,
        verbose=config.get("keras_verbose", 1)
    )
    duration = time.time() - start
    return model, history, duration

from tensorflow.keras.models import Model

def run_inference_lstm(model: Model, X_test: np.ndarray) -> np.ndarray:
    """
    Führt die Inferenz für ein LSTM-Modell durch.
    
    Args:
        model (Model): Keras LSTM-Modell.
        X_test (np.ndarray): Eingabedaten für die Vorhersage, Form: (samples, timesteps, features).
    
    Returns:
        np.ndarray: Vorhersagen des Modells, ggf. flach als 1D-Array.
    """
    print("🔍 Starte LSTM-Inferenz...")
    
    if len(X_test.shape) != 3:
        raise ValueError(f"❌ Erwartete Eingabeform (samples, timesteps, features), aber erhalten: {X_test.shape}")

    try:
        preds = model.predict(X_test, verbose=0)
        preds = np.array(preds)
        print(f"✅ LSTM-Inferenz abgeschlossen – Ausgabeform: {preds.shape}")
    except Exception as e:
        print(f"❌ Fehler bei der LSTM-Inferenz: {e}")
        import traceback
        print(traceback.format_exc())
        raise

    return preds


def save_results_LSTM(config: dict,
                      model: tf.keras.Model,
                      history: dict,
                      pred_orig: np.ndarray,
                      true_orig: np.ndarray,
                      dates: pd.DatetimeIndex,
                      metrics_values: dict,
                      paths: dict,
                      power_time: float,
                      original_features_list: List[str],
                      scaler) -> dict:

    results = PipelineUtils._save_common_results(
        config=config,
        pred_orig=pred_orig,
        true_orig=true_orig,
        dates=dates,
        metrics_values=metrics_values,
        paths=paths,
        power_time=power_time,
        scaler=scaler
    )

    try:
        model_path = PipelineUtils.save_model_with_version(
            model=model,
            directory=paths.get("Models", paths["Base_Output_Path"]),
            model_name=config["model_name"],
            dataset=config["dataset"],
            run_id=config.get("run_id", "run"),
            timestamp=config.get("time_stamp", "ts")
        )
        results["model_path"] = model_path
    except Exception as e:
        print(f"Modellspeicherung fehlgeschlagen: {e}")

    try:
        structure_dir = paths.get("Model_Structures", paths["Base_Output_Path"])
        os.makedirs(structure_dir, exist_ok=True)
        structure_path = os.path.join(structure_dir, f"structure_{config['run_id']}_{config['time_stamp']}.png")
        PipelineUtils.plot_model(model, to_file=structure_path)
        results["model_structure_path"] = structure_path
    except Exception as e:
        print(f"Modellstruktur-Speicherung fehlgeschlagen: {e}")

    try:
        plot_dir = paths.get("Loss_Plots", paths["Base_Output_Path"])
        os.makedirs(plot_dir, exist_ok=True)
        loss_plot_path = PipelineUtils.save_loss_plot(
            history=history,
            model_name=config["model_name"],
            dataset=config["dataset"],
            run_id=config["run_id"],
            timestamp=config["time_stamp"],
            output_dir=plot_dir
        )
        results["loss_plot_path"] = loss_plot_path
    except Exception as e:
        print(f"Loss Plot Speicherung fehlgeschlagen: {e}")

    return results



def quantize_model(model: tf.keras.Model) -> bytes:
    """Konvertiert ein Keras-Modell zu einem quantisierten TFLite-Modell."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quant_model = converter.convert()
    return tflite_quant_model


def save_edge_package(edge_dir: str, tflite_model: bytes, scaler, feature_list: list):
    """Speichert Modell, Scaler und Features für Edge Deployment."""
    os.makedirs(edge_dir, exist_ok=True)

    # Modell speichern
    tflite_path = os.path.join(edge_dir, "model_lstm.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    # Scaler speichern (MinMax oder StandardScaler)
    if hasattr(scaler, 'mean_'):
        np.save(os.path.join(edge_dir, "scaler_mean.npy"), scaler.mean_)
        np.save(os.path.join(edge_dir, "scaler_scale.npy"), scaler.scale_)
    elif hasattr(scaler, 'data_min_'):
        np.save(os.path.join(edge_dir, "scaler_min.npy"), scaler.data_min_)
        np.save(os.path.join(edge_dir, "scaler_max.npy"), scaler.data_max_)

    # Feature-Liste speichern
    with open(os.path.join(edge_dir, "features_lstm_input.json"), "w") as f:
        json.dump(feature_list, f, indent=2)

    return {
        "tflite_model": tflite_path,
        "features": os.path.join(edge_dir, "features_lstm_input.json")
    }


def send_to_edge_device(edge_ip: str, username: str, password: str, local_dir: str, remote_dir: str):
    """Sendet Modellpaket über SSH/SCP an Edge Device."""
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(edge_ip, username=username, password=password)

    sftp = ssh.open_sftp()
    sftp.mkdir(remote_dir) if not remote_dir in sftp.listdir(".") else None

    for file in os.listdir(local_dir):
        sftp.put(os.path.join(local_dir, file), os.path.join(remote_dir, file))

    sftp.close()
    ssh.close()
    print(f"✅ Modellpaket an {edge_ip}:{remote_dir} gesendet.")


def load_quantized_model_from_edge(edge_model_path: str) -> tf.lite.Interpreter:
    """Lädt ein quantisiertes Modell vom Edge Device."""
    interpreter = tf.lite.Interpreter(model_path=edge_model_path)
    interpreter.allocate_tensors()
    return interpreter


def is_edge_training_possible() -> bool:
    """Prüft, ob Edge Device für Training geeignet ist."""
    try:
        import tensorflow as tf
        devices = tf.config.list_physical_devices()
        return len(devices) > 0
    except Exception:
        return False
