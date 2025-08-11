
# LSTM_Utils.py

import os
import json
from pyexpat import model
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import joblib
import traceback


import paramiko

from typing import Tuple, List
from tensorflow.keras.models import Model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils


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

    model.add(Dense(forecast_horizon, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_model_LSTM(config: dict, X_train: np.ndarray,
                     y_train: np.ndarray, features: List[str]):
    
    """
    Baut, kompiliert und trainiert ein LSTM-Modell.

    Args:
        config (dict): Konfigurationsparameter (lags, num_layers, epochs, etc.).
        X_train (np.ndarray): Trainingsdaten (3D: [samples, lags, features]).
        y_train (np.ndarray): Zielwerte (2D: [samples, horizon]).
        features (list): Liste der Feature-Namen.

    Returns:
        tuple: (model, history, train_time)
    """

    # --- Modell bauen und kompilieren (unverändert) ---
    input_shape_lstm = (config["lags"], len(features))
    model = build_dynamic_lstm(
        input_shape=input_shape_lstm,
        num_layers=config.get("num_layers", 1),
        initial_units=config.get("initial_units", 64),
        dropout=config.get("dropout", 0.1),
        forecast_horizon=config["horizon"]
    )

    loss_function = config.get("loss", "huber_loss")
    optimizer = config.get("optimizer", "adam")
    model.compile(optimizer=optimizer, loss=loss_function, metrics=config.get("metrics", ["mae"]))



    X_fit, y_fit, X_val, y_val = PipelineUtils.create_timeseries_validation_split(
        X_train, y_train, config
    )

    # Stelle das validation_data-Tupel für Keras zusammen, falls der Split Daten geliefert hat.
    if X_val is not None and y_val is not None:
        val_data = (X_val, y_val)
        print(f"Keras-Modell wird mit Validierungsdaten der Form X:{X_val.shape}, y:{y_val.shape} trainiert.")
    else:
        val_data = None

    callbacks = PipelineUtils.get_keras_callbacks(config)

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



def save_lstm_metrics_results(config: dict, **kwargs) -> dict:
    """
    Speichert Evaluationsmetriken und Pfad zur Prediction-Datei.
    """
    print("--- Speichere Evaluationsmetriken für LSTM ---")
    
    metrics_results = PipelineUtils._save_metrics_prediction_gerneral(config=config, **kwargs)
    
    return metrics_results

def save_lstm_deployment_artifacts(config: dict,
                                    model: tf.keras.Model,
                                    scaler,
                                    history: dict,
                                    representative_dataset,
                                    **kwargs) -> dict:
    """
    Speichert das trainierte Modell, Scaler, Training-History und weitere Deployment-Artefakte.
    """
    print("--- Speichere LSTM Modell und Scaler für Deployment ---")
    
    saver = PipelineUtils.ModelScalerSaver(config, paths=kwargs.get("paths"))
    
    deployment_results = saver.save_artifacts(
        model=model,
        scaler=scaler,
        history=history,
        representative_dataset=representative_dataset
    )
    
    return deployment_results

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
    try:
        ssh.connect(edge_ip, username=username, password=password)

        sftp = ssh.open_sftp()
        try:
            sftp.mkdir(remote_dir)
        except IOError: # Verzeichnis existiert bereits
            pass

        for file in os.listdir(local_dir):
            sftp.put(os.path.join(local_dir, file), os.path.join(remote_dir, file))

        sftp.close()
        ssh.close()
        logging.info(f"✅ Modellpaket an {edge_ip}:{remote_dir} gesendet.")
    except Exception as e:
        logging.error(f"Fehler beim Senden an Edge Device: {e}", exc_info=True)
        raise


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


def load_model_LSTM(model_path, model_name, model_type="quantized"):
    """
    Lädt ein LSTM-Modell, entweder ein normales Keras-Modell oder ein quantisiertes TFLite-Modell.

    Args:
        model_path (str): Der Pfad zum Verzeichnis, das die Modelldatei enthält.
        model_name (str): Der Name der Modelldatei (ohne Pfad).
        model_type (str): Der Typ des zu ladenden Modells ("normal" für Keras, "quantized" für TFLite).

    Returns:
        tf.keras.Model or tf.lite.Interpreter: Das geladene Modell oder der TFLite Interpreter.

    Raises:
        ValueError: Wenn ein ungültiger model_type angegeben wird.
        RuntimeError: Wenn das Laden des Modells fehlschlägt.
    """
    full_model_path = os.path.join(model_path, model_name)
    logging.info(f"Versuche, {model_type} Modell von {full_model_path} zu laden...")

    if model_type == "normal":
        try:
            model = tf.keras.models.load_model(full_model_path)
            logging.info(f"Normales Keras LSTM-Modell erfolgreich geladen.")
            return model
        except Exception as e:
            logging.error(f"Fehler beim Laden des normalen Keras LSTM-Modells von {full_model_path}: {e}", exc_info=True)
            raise RuntimeError(f"Normales Keras LSTM-Modell konnte nicht geladen werden: {e}")
    elif model_type == "quantized":
        try:
            interpreter = tf.lite.Interpreter(
                model_path=full_model_path,
                # experimental_preserve_all_tensors=True # Kann für Debugging nützlich sein, aber nicht zwingend für Flex
            )
            interpreter.allocate_tensors()

            # Check if Flex delegate is needed (as per suggestion)
            # This check uses a private API and is primarily for informational purposes.
            # In a production environment, you might rely on the runtime error if the delegate isn't linked.
            if hasattr(interpreter, '_get_ops_details'):
                if any('Flex' in op.get('op_name', '') for op in interpreter._get_ops_details()):
                    logging.warning("Warning: Loaded TFLite model requires Flex delegate for deployment.")

            logging.info(f"Quantisiertes TFLite LSTM-Modell erfolgreich geladen.")
            return interpreter
        except Exception as e:
            logging.error(f"Fehler beim Laden des quantisierten TFLite LSTM-Modells von {full_model_path}: {e}", exc_info=True)
            raise RuntimeError(f"Quantisiertes TFLite LSTM-Modell konnte nicht geladen werden: {e}")
    else:
        raise ValueError(f"Ungültiger Modelltyp: {model_type}. Muss 'normal' oder 'quantized' sein.")





