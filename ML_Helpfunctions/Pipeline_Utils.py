import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
from typing import List, Tuple
import json
import logging
import traceback
from pathlib import Path

import psutil
import time


import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

import tensorflow as tf

from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    mean_squared_log_error,
    median_absolute_error
)


# -------------------------------------------
# Hilfsfunktionen: Daten vorbereiten & Skalierung
# -------------------------------------------
def get_keras_callbacks(config: dict) -> list:
    """
    Erstellt eine Liste von Keras-Callbacks basierend auf der Konfiguration.

    Args:
        config (dict): Das Konfigurationsdictionary.

    Returns:
        list: Eine Liste von Keras-Callback-Instanzen.
    """
    callbacks = []
    
    # Early Stopping Callback
    if config.get("use_early_stopping", True):
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=config.get("early_stopping_monitor", "val_loss"),
            patience=config.get("early_stopping_patience", 10),
            restore_best_weights=True
        ))
        print("Callback aktiviert: EarlyStopping")

    # Reduce Learning Rate on Plateau Callback
    if config.get("use_reduce_lr_on_plateau", True):
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor=config.get("lr_plateau_monitor", "val_loss"),
            factor=config.get("lr_factor", 0.5),
            patience=config.get("lr_patience", 3),
            min_lr=config.get("min_lr", 1e-6)
        ))
        print("Callback aktiviert: ReduceLROnPlateau")
        
    return callbacks


def create_timeseries_validation_split(X_train, y_train, config):
    """
    Teilt Trainingsdaten chronologisch in ein Trainings- und ein Validierungsset.
    Die letzten Datenpunkte werden für die Validierung verwendet.

    Args:
        X_train (np.ndarray): Die vollständigen Trainingsdaten (Input).
        y_train (np.ndarray): Die vollständigen Trainingsdaten (Zielwerte).
        config (dict): Konfigurationsdictionary, das 'validation_fraction' enthält.

    Returns:
        tuple: (X_fit, y_fit, X_val, y_val)
               Gibt die aufgeteilten Daten zurück. Wenn keine Validierung stattfindet,
               sind X_val und y_val None.
    """
    val_fraction = config.get("validation_fraction", 0.0)

    # Nur splitten, wenn eine valide Fraktion angegeben ist und genügend Daten vorhanden sind.
    if val_fraction > 0 and X_train.shape[0] > 10:
        print(f"Erstelle chronologischen Validierungs-Split. Validation Fraction: {val_fraction}")
        split_index = int((1 - val_fraction) * len(X_train))
        
        X_fit = X_train[:split_index]
        y_fit = y_train[:split_index]
        
        X_val = X_train[split_index:]
        y_val = y_train[split_index:]
        
        return X_fit, y_fit, X_val, y_val
    else:
        # Wenn keine Validierung stattfinden soll, gib die Originaldaten und None zurück.
        print("Kein Validierungs-Split durchgeführt.")
        return X_train, y_train, None, None

def safe_inverse_transform(scaler, array, target_index=0):
    """
    Sichere inverse Transformation eines skalierten Arrays für ein Ziel-Feature.
    Unterstützt 1D und 2D Arrays.
    Wenn scaler None ist, wird das Array unverändert zurückgegeben.
    """
    if scaler is None:
        # Kein Scaler => keine Transformation nötig
        return array

    if array.ndim == 1:
        full = np.zeros((len(array), scaler.scale_.shape[0]))
        full[:, target_index] = array
        return scaler.inverse_transform(full)[:, target_index]
    elif array.ndim == 2:
        results = []
        for step in range(array.shape[1]):
            temp = np.zeros((array.shape[0], scaler.scale_.shape[0]))
            temp[:, target_index] = array[:, step]
            inverse = scaler.inverse_transform(temp)[:, target_index]
            results.append(inverse)
        return np.stack(results, axis=1)

    
def flatten_config(config: dict, prefix: str = "") -> dict:
    """
    Rekursives Flattening der Config für CSV-Speicherung.
    """
    flat = {}
    for key, value in config.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, prefix=full_key + "_"))
        elif isinstance(value, list):
            flat[full_key] = ", ".join(map(str, value))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            flat[full_key] = value
        else:
            flat[full_key] = str(value)  # Sicherer Fallback
    return flat

def save_metrics_summary(metrics: dict, infer_config: dict, train_config: dict, paths: dict) -> str:
    """
    Fügt die Metriken, Konfigurationen und Pfade eines Inferenzlaufs 
    zu einer zentralen Übersichts-CSV-Datei hinzu.
    """
    summary_path = None
    try:
        # Pfad zur zentralen Metrik-Datei
        summary_dir = paths.get("Error_Metrics")
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, "metrics_summary.csv")
        
        # Konfigurationen für die Speicherung vorbereiten
        flat_infer_cfg = flatten_config(infer_config, "infer_")
        flat_train_cfg = flatten_config(train_config, "train_") if train_config else {}

        # Alle Daten für die neue Zeile kombinieren
        new_row = {
            "run_id": infer_config.get("run_id"),
            "timestamp": infer_config.get("time_stamp"),
            "model_name": infer_config.get("model_name"),
            **metrics,
            **flat_infer_cfg,
            **flat_train_cfg
        }
        
        # Bestehende Datei laden oder neuen DataFrame erstellen
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path)
        else:
            summary_df = pd.DataFrame()
            
        # Neue Zeile hinzufügen und speichern
        # Alte Spalten beibehalten und neue hinzufügen, falls sie nicht existieren
        summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)
        summary_df.to_csv(summary_path, index=False)
        
        print(f"✅ Metriken-Zusammenfassung aktualisiert: {summary_path}")
    except Exception as e:
        print(f"❌ Fehler beim Speichern der Metriken-Zusammenfassung: {e}")
        traceback.print_exc()
    return summary_path


def evaluate_all_metrics(y_true, y_pred, y_train=None, horizon=1, alpha=0.8):
    """
    Berechnet verschiedene Fehlermetriken für ein- oder mehrstufige Vorhersagen.
    
    Args:
        y_true (np.ndarray): Wahre Werte (N, H)
        y_pred (np.ndarray): Vorhersagewerte (N, H)
        y_train (np.ndarray): Trainingsdaten für MASE
        horizon (int): Forecast-Horizont
        alpha (float): Gewichtungsfaktor für weighted MAE bei multi-step Vorhersage
        
    Returns:
        dict: Alle Metriken als Schlüssel-Wert-Paare
    """

    def safe_divide(a, b):
        return a / np.where(b == 0, np.finfo(float).eps, b)

    def smape(y_t, y_p):
        return np.mean(safe_divide(np.abs(y_p - y_t), (np.abs(y_t) + np.abs(y_p)) / 2)) * 100

    def wape(y_t, y_p):
        return np.sum(np.abs(y_t - y_p)) / np.sum(np.abs(y_t)) * 100

    def weighted_mae(y_t, y_p, alpha):
        weights = np.array([alpha ** i for i in range(horizon)])[::-1]
        abs_errors = np.abs(y_t - y_p)
        return np.mean(abs_errors * weights)

    metrics = {}

    if horizon == 1 or len(y_true.shape) == 1:
        # Falls 1D, umformen
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['mape'] = np.mean(np.abs(safe_divide(y_true - y_pred, y_true))) * 100
        metrics['smape'] = smape(y_true, y_pred)
        metrics['wape'] = wape(y_true, y_pred)
        metrics['msle'] = mean_squared_log_error(np.maximum(y_true, 0), np.maximum(y_pred, 0))
        metrics['median_ae'] = median_absolute_error(y_true, y_pred)

        if y_train is not None and len(y_train) > 1:
            naive_forecast = np.abs(np.diff(y_train)).mean()
            metrics['mase'] = metrics['mae'] / naive_forecast if naive_forecast != 0 else np.nan
        else:
            metrics['mase'] = np.nan

    else:
        # Multistep Forecast: Horizon > 1
        metrics['mse'] = []
        metrics['rmse'] = []
        metrics['mae'] = []
        metrics['r2'] = []
        metrics['mape'] = []
        metrics['smape'] = []
        metrics['wape'] = []
        metrics['msle'] = []
        metrics['median_ae'] = []

        for t in range(horizon):
            yt = y_true[:, t]
            yp = y_pred[:, t]
            metrics['mse'].append(mean_squared_error(yt, yp))
            metrics['rmse'].append(np.sqrt(metrics['mse'][-1]))
            metrics['mae'].append(mean_absolute_error(yt, yp))
            metrics['r2'].append(r2_score(yt, yp))
            metrics['mape'].append(np.mean(np.abs(safe_divide(yt - yp, yt))) * 100)
            metrics['smape'].append(smape(yt, yp))
            metrics['wape'].append(wape(yt, yp))
            metrics['msle'].append(mean_squared_log_error(np.maximum(yt, 0), np.maximum(yp, 0)))
            metrics['median_ae'].append(median_absolute_error(yt, yp))

        if y_train is not None and len(y_train) > 1:
            naive_forecast = np.abs(np.diff(y_train)).mean()
            mean_mae = np.mean(metrics['mae'])
            metrics['mase'] = mean_mae / naive_forecast if naive_forecast != 0 else np.nan
        else:
            metrics['mase'] = np.nan

        # Gewichtete Fehler
        metrics['weighted_mae'] = weighted_mae(y_true, y_pred, alpha)

    return metrics


def save_prediction_data(
    config: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray,
    output_path: str = None  
) -> str:
    """
    Speichert Vorhersagedaten mit Zeitstempeln anhand der Konfigurationsdaten.
    y_true und y_pred werden als 1D-Arrays (geflacht, falls Horizon > 1) erwartet.
    
    Optional kann ein vollständiger Dateipfad übergeben werden (output_path),
    andernfalls wird basierend auf config gespeichert.
    """
    print("--- DEBUG: FÜHRE save_prediction_data AUS (mit optionalem output_path) ---")

    model_name = config.get("model_name", "model")
    dataset = config.get("dataset", "data")
    run_id = config.get("run_id", "run")
    timestamp = config.get("time_stamp", "timestamp")
    output_dir = config.get("paths", {}).get("Prediction_Data", ".")
    horizon = config.get("horizon", 1)

    num_samples = len(dates)

    if len(y_true) != num_samples * horizon:
        raise ValueError(f"Länge von y_true ({len(y_true)}) stimmt nicht mit num_samples ({num_samples}) * horizon ({horizon}) überein.")
    if len(y_pred) != num_samples * horizon:
        raise ValueError(f"Länge von y_pred ({len(y_pred)}) stimmt nicht mit num_samples ({num_samples}) * horizon ({horizon}) überein.")

    try:
        y_true_reshaped = y_true.reshape(num_samples, horizon)
        y_pred_reshaped = y_pred.reshape(num_samples, horizon)
    except ValueError as e:
        raise ValueError(f"Fehler beim Reshapen von y_true/y_pred. Originale Exception: {e}")

    df_data = {'date': dates}
    for h in range(horizon):
        df_data[f'true_h{h+1}'] = y_true_reshaped[:, h]
        df_data[f'pred_h{h+1}'] = y_pred_reshaped[:, h]

    df = pd.DataFrame(df_data)

    if output_path is None:
        filename = f"PredictionData_{run_id}_{model_name}_{dataset}_{timestamp}.csv"
        output_path = os.path.join(output_dir, filename)

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✅ Vorhersagedatei gespeichert unter: {output_path}")
    except Exception as e_csv:
        raise IOError(f"Fehler beim Schreiben der CSV-Datei '{output_path}': {e_csv}")

    return output_path


# -------------------------------------------
# Experiment Setup
# -------------------------------------------

def setup_experiment(config: dict, folder_flag: str, run_type: str = None) -> tuple[dict, dict]:
    """
    Initialisiert das Experiment: Erstellt die Ausgabeordnerstruktur basierend auf einem Modell-Flag.
    """
    if not folder_flag or not isinstance(folder_flag, str):
        raise ValueError("Ein gültiger 'folder_flag' als String muss für das Setup übergeben werden.")

    # Zeitstempel & Run-ID erzeugen, falls nicht vorhanden
    if "time_stamp" not in config or config["time_stamp"] is None:
        config["time_stamp"] = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    if "run_id" not in config or config.get("run_id") is None:
        base_run_id = f"{config['time_stamp']}_{random.randint(1000, 9999)}"
        if run_type in ['train', 'inference']:
            config["run_id"] = f"{base_run_id}_{run_type}"
        else:
            config["run_id"] = base_run_id

    # Basispfade
    try:
        paths = config["paths"].copy()
        # Der Haupt-Ausgabepfad (z.B. .../Output)
        base_output_path = Path(paths["output"])
        input_path = Path(paths["input"])
    except KeyError:
        raise ValueError("Fehlender 'input' oder 'output' Pfad in config['paths'].")
        
    # --- NEUE LOGIK: Erstelle den modellspezifischen Ordner ---
    # z.B. .../Output/LSTM oder .../Output/RANDOM_FOREST
    model_base_path = base_output_path / folder_flag

    # Der Ordner für diesen spezifischen Lauf kommt in den Modell-Ordner
    # z.B. .../Output/LSTM/2025-07-20_160000_1234_train
    run_output_path = model_base_path / config["run_id"]

    # Input-Pfade bleiben unverändert
    input_subfolders = {
        "Input_Data": input_path / "Input_Data",
        "Input_Models": input_path / "Input_Models",
        "Input_Scaler": input_path / "Input_Scaler"
    }
    
    # Error_Metrics bleiben im Haupt-Output-Ordner, nicht pro Modell
    persistent_subfolders = {
        "Error_Metrics": base_output_path / "Error_Metrics"
    }

    # Die Unterordner für den Lauf werden relativ zum neuen run_output_path erstellt
    run_subfolders = {
        "Base_Output_Path": run_output_path,
        "Models": run_output_path / "Models",
        "Scalers": run_output_path / "Scalers",
        "Prediction_Data": run_output_path / "Prediction_Data",
        "Model_Structures": run_output_path / "Model_Structures",
        "Model_Summaries": run_output_path / "Model_Summaries",
        "Prediction_Plots": run_output_path / "Prediction_Plots",
        "Loss_Plots": run_output_path / "Loss_Plots"
    }

    # Alle Pfade zusammenführen und Konfiguration aktualisieren
    all_paths = {**paths, **input_subfolders, **persistent_subfolders, **run_subfolders}
    config["paths"] = all_paths

    # Alle benötigten Ordner anlegen
    for path in all_paths.values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)

    print(f"✅ Experiment-Setup für '{folder_flag}' abgeschlossen. Run ID: {config['run_id']}")
    print(f"📁 Ergebnisordner: {run_output_path}")

    return config, config["paths"]

# -------------------------------------------
# Modellbewertung
# -------------------------------------------

def get_cpu_usage() -> float:
    """
    Gibt die aktuelle systemweite CPU-Auslastung als Prozentwert zurück.
    
    Returns:
        float: CPU-Auslastung in Prozent.
    """
    # Der Parameter interval=None macht den Aufruf nicht-blockierend.
    # Er misst die Auslastung seit dem letzten Aufruf.
    return psutil.cpu_percent(interval=None)

def _evaluate_model(
    predictions: np.ndarray,
    y_test: np.ndarray,
    scaler: object,
    test_df: pd.DataFrame,
    config: dict,
    features: List[str],
    y_train: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, dict]:

    base_feature = config.get("base_features", [None])[0]
    if base_feature not in features:
        raise ValueError(f"Base feature '{base_feature}' nicht in features-Liste gefunden.")
    target_index = features.index(base_feature)

    pred_orig = safe_inverse_transform(scaler, predictions, target_index)
    true_orig = safe_inverse_transform(scaler, y_test, target_index)

    dates = test_df.index[config.get("lags", 0):][:len(pred_orig)]

    min_len = min(len(dates), len(true_orig), len(pred_orig))
    dates = dates[:min_len]
    true_orig = true_orig[:min_len]
    pred_orig = pred_orig[:min_len]

    metrics = evaluate_all_metrics(
        y_true=true_orig,
        y_pred=pred_orig,
        y_train=safe_inverse_transform(scaler, y_train, target_index),
        horizon=config.get("horizon", 1)
    )

    return pred_orig, true_orig, dates, metrics


def run_timed_inference(model: object, input_data: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Führt eine einzelne Inferenz durch und misst die exakte Dauer.
    Unterstützt Keras-Modelle, scikit-learn-Modelle und TFLite-Interpreter.
    """
    start_time = time.perf_counter()

    # Prüfe, ob das übergebene Modell ein TFLite-Interpreter ist
    if isinstance(model, tf.lite.Interpreter):
        # TFLite-Inferenz
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        
        # Stelle sicher, dass die Eingabedaten den vom Modell erwarteten Datentyp haben
        input_data_tflite = np.asarray(input_data, dtype=input_details[0]['dtype'])
        
        # Setze den Input-Tensor, führe die Inferenz aus und hole den Output-Tensor
        model.set_tensor(input_details[0]['index'], input_data_tflite)
        model.invoke()
        prediction = model.get_tensor(output_details[0]['index'])
    else:
        # Standard-Inferenz für Keras/scikit-learn-Modelle
        prediction = model.predict(input_data)

    end_time = time.perf_counter()
    duration_ms = (end_time - start_time) * 1000
    
    return prediction, duration_ms


# -------------------------------------------
# Hilfsfunktionen: Visualisierung & Modell speichern
# -------------------------------------------
def save_loss_plot(history: dict, config: dict, paths: dict, output_path: str):
    """
    Erstellt und speichert einen Plot des Trainings- und Validierungsverlusts
    sowie der Metriken aus dem Keras-History-Objekt.

    Args:
        history (dict): Das History-Objekt von model.fit().
        config (dict): Das Konfigurationsdictionary für Titelinformationen etc.
        paths (dict): Das Pfad-Dictionary.
        output_path (str): Der vollständige Pfad zum Speichern des Plots.
    """
    if not hasattr(history, 'history') or not history.history:
        print("⚠️ Kein gültiges History-Objekt zum Plotten vorhanden.")
        return

    history_dict = history.history
    
    # Schlüssel für Loss und die erste Metrik dynamisch finden
    loss_keys = sorted([k for k in history_dict if 'loss' in k])
    metric_keys = sorted([k for k in history_dict if k not in loss_keys])
    
    # Erstelle ein 2x1 Subplot-Grid, falls Metriken vorhanden sind, sonst nur 1x1
    num_subplots = 2 if metric_keys else 1
    fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 6 * num_subplots), sharex=True)
    
    # Sicherstellen, dass 'axes' immer ein Array ist, auch bei nur einem Subplot
    if num_subplots == 1:
        axes = [axes]

    # --- Subplot 1: Loss ---
    for key in loss_keys:
        axes[0].plot(history_dict[key], label=key)
    axes[0].set_title(f'Trainings- & Validierungs-Loss für {config.get("model_name")}')
    # Extrahiert den Loss-Namen aus der Konfiguration
    loss_name = str(config.get("loss")).split('.')[-1].replace("()", "")
    axes[0].set_ylabel(f'Loss ({loss_name})')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_yscale('log') # Log-Skala ist oft hilfreich für Loss-Plots

    # --- Subplot 2: Metriken (falls vorhanden) ---
    if num_subplots > 1:
        primary_metric_name = metric_keys[0].replace('val_', '') # z.B. 'mae'
        for key in metric_keys:
            axes[1].plot(history_dict[key], label=key)
        axes[1].set_title('Trainings- & Validierungs-Metrik')
        axes[1].set_ylabel(primary_metric_name.upper())
        axes[1].set_xlabel('Epoche')
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[0].set_xlabel('Epoche')

    fig.suptitle(f'Trainingsverlauf für Run: {config.get("run_id")}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Platz für den suptitle lassen
    
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        print(f"✅ Loss Plot gespeichert unter: {output_path}")
    except Exception as e:
        print(f"❌ FEHLER beim Speichern des Loss Plots unter '{output_path}': {e}")
    finally:
        plt.close(fig) # Schließt die Figur, um Speicher freizugeben

# def _quantize_and_save_tflite_model(model, directory, model_name, dataset, run_id, representative_dataset, timestamp=None):
#     """
#     Quantisiert ein Keras-Modell in ein TFLite-Modell (INT8) und speichert es.

#     Args:
#         model (tf.keras.Model): Das zu quantisierende Keras-Modell.
#         directory (str): Das Verzeichnis, in dem das Modell gespeichert werden soll.
#         model_name (str): Name des Modells (für den Dateinamen).
#         dataset (str): Name des Datensatzes (für den Dateinamen).
#         run_id (str): Eindeutige ID des Laufs (für den Dateinamen).
#         representative_dataset (tf.data.Dataset oder Generator): Ein kleiner Datensatz zur Kalibrierung der Quantisierung.
#         timestamp (str, optional): Optionaler Zeitstempel für den Dateinamen. Wird generiert, wenn None.

#     Returns:
#         str: Der Pfad zur gespeicherten TFLite-Modell-Datei.

#     Raises:
#         RuntimeError: Wenn das Quantisieren oder Speichern des Modells fehlschlägt.
#     """
#     timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = f"Model_{run_id}_{model_name}_{dataset}_{timestamp}_quantized.tflite"
#     model_path = os.path.join(directory, filename)

#     logging.info(f"Beginne mit der Quantisierung des Modells und Speicherung nach: {model_path}")

#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
#     # Der repräsentative Datensatz ist entscheidend für die vollständige Integer-Quantisierung
#     converter.representative_dataset = lambda: iter(representative_dataset)

#     converter.target_spec.supported_ops = [
#             tf.lite.OpsSet.TFLITE_BUILTINS, # Standard-Operationen
#             tf.lite.OpsSet.SELECT_TF_OPS    # Erlaube zusätzlich TensorFlow-Operationen
#         ]
#     converter._experimental_lower_tensor_list_ops = False

#     try:
#         tflite_model = converter.convert()
#         with open(model_path, "wb") as f:
#             f.write(tflite_model)
#         logging.info(f"Quantisiertes TFLite-Modell erfolgreich gespeichert unter: {model_path}")
#         return model_path
#     except Exception as e:
#         logging.error(f"Fehler beim Quantisieren und Speichern des TFLite-Modells: {e}", exc_info=True)
#         raise RuntimeError(f"TFLite-Modell konnte nicht gespeichert werden: {e}")
    

# # -------------------------------------------
# # Gemeinsame Ergebnisse speichern
# # -------------------------------------------
# def save_keras_artifacts(model: tf.keras.Model, 
#                          history: dict, 
#                          config: dict, 
#                          paths: dict, 
#                          representative_dataset) -> dict:
#     """
#     Speichert Keras-spezifische Artefakte: das Modell selbst (inkl. TFLite),
#     die Modellstruktur als Bild und den Trainingsverlauf (Loss Plot).

#     Args:
#         model: Das trainierte Keras-Modell.
#         history: Das History-Objekt von model.fit().
#         config: Das Konfigurationsdictionary des Laufs.
#         paths: Das Pfad-Dictionary des Laufs.
#         representative_dataset: Dataset für die TFLite-Quantisierung.

#     Returns:
#         dict: Ein Dictionary mit den Pfaden zu den gespeicherten Artefakten.
#     """
#     results = {}

#     # === Modell speichern (Keras .keras + TFLite .tflite) ===
#     try:
#         model_dir = paths.get("Models", os.path.join(paths.get("output"), "Models"))
        
#         # Annahme: save_model_with_version ist eine Ihrer Hilfsfunktionen
#         normal_model_path, quantized_model_path = save_model_with_version(
#             model=model,
#             directory=model_dir,
#             model_name=config["model_name"],
#             dataset=config["dataset"],
#             run_id=config.get("run_id", "run"),
#             timestamp=config.get("time_stamp", "ts"),
#             representative_dataset=representative_dataset
#         )
#         if normal_model_path:
#             results["model_path"] = normal_model_path
#             print(f"✅ Normales Keras-Modell gespeichert unter: {normal_model_path}")
#         if quantized_model_path:
#             results["quantized_model_path"] = quantized_model_path
#             print(f"✅ TFLite-Modell gespeichert unter: {quantized_model_path}")
#     except Exception as e:
#         print(f"❌ Fehler beim Speichern des Modells: {e}")
#         print(traceback.format_exc())

#     # === Modellstruktur speichern ===
#     try:
#         structure_dir = paths.get("Model_Structures", os.path.join(paths.get("output"), "Model_Structures"))
#         os.makedirs(structure_dir, exist_ok=True)
#         structure_path = os.path.join(structure_dir, f"structure_{config['run_id']}_{config['time_stamp']}.png")

#         tf.keras.utils.plot_model(model, to_file=structure_path, show_shapes=True, show_layer_activations=True)
#         print(f"📊 Modellstruktur gespeichert unter: {structure_path}")
#         results["model_structure_path"] = structure_path
#     except Exception as e:
#         print(f"❌ Fehler beim Speichern der Modellstruktur: {e}")
#         print(traceback.format_exc())

#     # === Loss Plot speichern ===

#     try:
#         plot_dir = paths.get("Loss_Plots", os.path.join(paths.get("output"), "Loss_Plots"))
#         os.makedirs(plot_dir, exist_ok=True)
#         loss_plot_path = os.path.join(plot_dir, f"loss_plot_{config['run_id']}_{config['time_stamp']}.png")
        
#         # Dieser Aufruf passt jetzt perfekt zur neuen Funktionsdefinition
#         save_loss_plot(history, config, paths, loss_plot_path)
        
#         print(f"📉 Loss Plot gespeichert unter: {loss_plot_path}")
#         results["loss_plot_path"] = loss_plot_path
#     except Exception as e:
#         print(f"❌ Fehler beim Speichern des Loss Plots: {e}")
#         print(traceback.format_exc())
        
#     return results




# def save_model_with_version(model, directory, model_name, dataset, run_id, timestamp=None, representative_dataset=None, quantized_output_dir=None):
#     """
#     Speichert ein Keras-Modell im normalen Zustand und optional als quantisiertes TFLite-Modell.

#     Args:
#         model (tf.keras.Model): Das zu speichernde Keras-Modell.
#         directory (str): Das Verzeichnis, in dem das normale Keras-Modell gespeichert werden soll.
#         model_name (str): Name des Modells (für den Dateinamen).
#         dataset (str): Name des Datensatzes (für den Dateinamen).
#         run_id (str): Eindeutige ID des Laufs (für den Dateinamen).
#         timestamp (str, optional): Optionaler Zeitstempel für den Dateinamen. Wird generiert, wenn None.
#         representative_dataset (tf.data.Dataset oder Generator, optional): Erforderlich für die Quantisierung.
#         quantized_output_dir (str, optional): Spezifisches Verzeichnis für das quantisierte Modell.
#                                               Verwendet 'directory', wenn None.

#     Returns:
#         tuple: (normal_model_path, quantized_model_path). Pfade zu den gespeicherten Modellen.
#                quantized_model_path ist None, wenn Quantisierung nicht durchgeführt wurde/fehlschlug.
#     """
#     timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # --- Speichern des normalen Keras-Modells ---
#     normal_filename = f"Model_{run_id}_{model_name}_{dataset}_{timestamp}.keras"
#     normal_model_path = os.path.join(directory, normal_filename)
#     try:
#         model.save(normal_model_path)
#         logging.info(f"Normales Keras-Modell erfolgreich gespeichert unter: {normal_model_path}")
#     except Exception as e:
#         logging.error(f"Fehler beim Speichern des normalen Keras-Modells: {e}", exc_info=True)
#         normal_model_path = None # Setze auf None, wenn Speichern fehlschlägt

#     # --- Speichern des quantisierten TFLite-Modells ---
#     quantized_model_path = None
#     if representative_dataset is not None:
#         target_quantized_dir = quantized_output_dir if quantized_output_dir else directory
#         os.makedirs(target_quantized_dir, exist_ok=True) # Stelle sicher, dass das Verzeichnis existiert
#         try:
#             quantized_model_path = _quantize_and_save_tflite_model(
#                 model=model,
#                 directory=target_quantized_dir, # Verwende spezifisches Verzeichnis für quantisiertes Modell
#                 model_name=model_name,
#                 dataset=dataset,
#                 run_id=run_id,
#                 representative_dataset=representative_dataset,
#                 timestamp=timestamp
#             )
#         except Exception as e:
#             logging.error(f"Quantisierung und Speicherung des TFLite-Modells fehlgeschlagen: {e}", exc_info=True)
#             quantized_model_path = None

#     return normal_model_path, quantized_model_path

# -----------------------------------------------------------------------------
# HELPER 1: VERZEICHNISSE SICHERSTELLEN
# -----------------------------------------------------------------------------
def _ensure_output_dirs_exist(paths: dict, dir_keys: List[str]):
    """Stellt sicher, dass alle benötigten Ausgabe-Verzeichnisse existieren."""
    if not paths: return
    try:
        print(f"📁 Stelle sicher, dass Ausgabe-Verzeichnisse existieren...")
        for key in dir_keys:
            dir_path = paths.get(key)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)
    except Exception as e:
        print(f"❌ Fehler beim Erstellen von Verzeichnissen: {e}")
        print(traceback.format_exc())

# -----------------------------------------------------------------------------
# HELPER 2: SKALIERER SPEICHERN (Ihre Funktion, verbessert)
# -----------------------------------------------------------------------------
def save_scaler(scaler, config: dict, paths: dict) -> str:
    """
    Speichert ein Skalierer-Objekt, aber nur, wenn laut Konfiguration
    eine Skalierung stattgefunden hat und ein Skalierer-Objekt existiert.
    """
    scaler_path = None # Wichtig: Initialisieren für den Fehlerfall
    # Prüft beide Skalierungs-Flags in der Konfiguration
    should_scale = config.get("scale_target", False)
    
    if should_scale and scaler is not None:
        try:
            scaler_dir = paths.get("Scalers")
            scaler_filename = f"scaler_{config.get('run_id')}_{config.get('time_stamp')}.joblib"
            scaler_path = os.path.join(scaler_dir, scaler_filename)
            joblib.dump(scaler, scaler_path)
            print(f"✅ Skalierer gespeichert unter: {scaler_path}")
        except Exception as e:
            print(f"⚠️ Fehler beim Speichern des Skalierers: {e}")
            print(traceback.format_exc())
    elif should_scale and scaler is None:
        print("⚠️ Warnung: Skalierung war konfiguriert, aber es wurde kein Skalierer-Objekt übergeben.")
    return scaler_path

# -----------------------------------------------------------------------------
# HELPER 3: VORHERSAGEN SPEICHERN
# -----------------------------------------------------------------------------
def save_predictions_to_csv(pred_orig: np.ndarray, true_orig: np.ndarray,
                            dates: pd.DatetimeIndex, config: dict, paths: dict) -> str:
    """
    Speichert wahre Werte und Vorhersagen in einer CSV-Datei.

    """
    prediction_path = None
    try:
        prediction_dir = paths.get("Prediction_Data")
        prediction_filename = f"predictions_{config.get('run_id')}_{config.get('time_stamp')}.csv"
        prediction_path = os.path.join(prediction_dir, prediction_filename)

        horizon = config.get("horizon", 1)
        
        # Erstelle Spaltennamen für den Horizont
        true_cols = [f'true_t+{i+1}' for i in range(horizon)]
        pred_cols = [f'pred_t+{i+1}' for i in range(horizon)]

        # Erstelle DataFrames aus den 2D-Numpy-Arrays
        true_df = pd.DataFrame(true_orig, columns=true_cols)
        pred_df = pd.DataFrame(pred_orig, columns=pred_cols)
        
        # Kombiniere alles zu einem finalen DataFrame
        # Stelle sicher, dass der 'dates'-Index mit den Daten übereinstimmt
        final_df = pd.concat([
            pd.DataFrame({'date': dates[:len(true_df)]}), 
            true_df, 
            pred_df
        ], axis=1)

        final_df.to_csv(prediction_path, index=False)
        print(f"✅ Vorhersagedaten (Multi-Step) gespeichert unter: {prediction_path}")
    except Exception as e:
        print(f"❌ Fehler beim Speichern der Vorhersagedaten: {e}")
        print(traceback.format_exc())
    return prediction_path
# -----------------------------------------------------------------------------
# HELPER 4: METRIKEN SPEICHERN
# -----------------------------------------------------------------------------
def save_metrics_csv(metrics: dict, config: dict, paths: dict, 
                                 power_time: float, prediction_path: str) -> str:
    """Fügt die Metriken des aktuellen Laufs zu einer zentralen Übersichts-CSV hinzu."""
    summary_path = None
    if not prediction_path:
        print("⚠️ Keine Vorhersagedatei vorhanden – Metriken nicht in Zusammenfassung gespeichert.")
        return summary_path
    try:
        summary_path = os.path.join(paths.get("Error_Metrics"), "metrics_summary.csv")
        
        # Erstelle eine Zeile für die CSV-Datei
        new_row = {
            "timestamp": config.get("time_stamp"),
            "run_id": config.get("run_id"),
            "model_name": config.get("model_name"),
            "dataset": config.get("dataset"),
            "training_time_s": round(power_time, 2),
            **metrics # Füge alle Metriken (MAE, MSE etc.) hinzu
        }
        
        # Lade die bestehende Datei oder erstelle einen neuen DataFrame
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path)
        else:
            summary_df = pd.DataFrame()
            
        # Füge die neue Zeile hinzu und speichere
        summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)
        summary_df.to_csv(summary_path, index=False)
        
        print(f"✅ Metriken-Zusammenfassung aktualisiert: {summary_path}")
    except Exception as e:
        print(f"❌ Fehler beim Speichern der Metriken: {e}")
        print(traceback.format_exc())
    return summary_path




def _save_common_results(
    config: dict,
    pred_orig: np.ndarray,
    true_orig: np.ndarray,
    dates: pd.DatetimeIndex,
    metrics: dict,
    paths: dict,
    power_time: float,
    scaler) -> dict:
    
    print("🟢 Starte Speichern der gemeinsamen Ergebnisse...")
    
    # 1. Stelle sicher, dass die Ordner existieren
    _ensure_output_dirs_exist(paths, ["Prediction_Data", "Error_Metrics", "Scalers"])
    
    # 2. Speichere den Skalierer (falls konfiguriert)
    scaler_path = save_scaler(scaler, config, paths)
    
    # 3. Speichere die Vorhersagen
    prediction_path = save_predictions_to_csv(pred_orig, true_orig, dates, config, paths)
    
    # 4. Speichere die Metriken
    metrics_summary_path = save_metrics_csv(metrics, config, paths, power_time, prediction_path)
    
    # 5. Sammle die Ergebnisse und gib sie zurück
    results = {
        "scaler_path": scaler_path,
        "prediction_file": prediction_path,
        "metrics_summary_path": metrics_summary_path,
    }
    
    print("✅ Gemeinsame Ergebnis-Speicherung abgeschlossen.")
    return results


def _save_metrics_prediction_gerneral(
    config: dict,
    pred_orig: np.ndarray,
    true_orig: np.ndarray,
    dates: pd.DatetimeIndex,
    metrics: dict,
    paths: dict,
    power_time: float,
    **kwargs
    ) -> dict:
    
    print("🟢 Starte Speichern der gemeinsamen Ergebnisse...")
    
    # 1. Stelle sicher, dass die Ordner existieren
    _ensure_output_dirs_exist(paths, ["Prediction_Data", "Error_Metrics"])
    
    # 3. Speichere die Vorhersagen
    prediction_path = save_predictions_to_csv(pred_orig, true_orig, dates, config, paths)
    
    # 4. Speichere die Metriken
    metrics_summary_path = save_metrics_csv(metrics, config, paths, power_time, prediction_path)
    
    # 5. Sammle die Ergebnisse und gib sie zurück
    results_path = {
        "prediction_file": prediction_path,
        "metrics_summary_path": metrics_summary_path,
    }
    
    print("✅ Gemeinsame Ergebnis-Speicherung abgeschlossen.")
    return results_path


def create_representative_dataset_generator(dataset: tf.data.Dataset):
    """
    Erstellt einen Generator für die Representative Dataset Funktion.
    Erwartet ein Dataset mit Eingabe-Tensoren (oder Tupeln mit Eingabe und Label).
    Gibt nur die Eingabe-Tensoren zurück.
    """
    def generator():
        for data_sample in dataset.take(100):  
            if isinstance(data_sample, tuple):
                input_data = data_sample[0]
            else:
                input_data = data_sample
            yield [tf.cast(input_data, tf.float32)]

    return generator


def load_model_artifacts_for_inference(config: dict, folder_flag ) -> tuple:
    """
    Lädt robust die notwendigen Artefakte (Scaler, Features, Modell) für die Inferenz.
    Die Funktion unterstützt zwei verschiedene Modi, die über die Konfiguration
    gesteuert werden. Sie erkennt den Modelltyp automatisch anhand der Dateiendung.

    Args:
        config (dict): Das Konfigurations-Wörterbuch, das den Modus und die Pfade enthält.

    Returns:
        tuple: Ein Tupel mit den geladenen Artefakten in der Reihenfolge (scaler, features, model).

    Raises:
        ValueError: Wenn ein unbekannter 'inference_mode' angegeben wird oder
                    notwendige Konfigurationsschlüssel fehlen.
        FileNotFoundError: Wenn eine der Artefakt-Dateien nicht gefunden wird.

    ---------------------------------------------------------------------------
    Modi:
    ---------------------------------------------------------------------------
    1. mode: 'load_artifacts_fast' (Statischer Modus)
       Lädt Artefakte von fest definierten Pfaden. Ideal für schnelles Testen.
       Benötigte Schlüssel in der config:
       - "model_path_static": "trained_rf_model.joblib" (oder .keras)
       - "scaler_path_static": "trained_rf_scaler.joblib"
       - "features_path_static": "trained_rf_features.joblib"

    2. mode: 'load_artifacts_path' (Dynamischer Modus)
       Lädt Artefakte aus einem versionierten Ordner, der über eine 'load_id'
       in der Konfiguration bestimmt wird. Ideal für den produktiven Einsatz.
       Benötigte Schlüssel in der config:
       - "artifacts_path": "Output/saved_models" (Basis-Verzeichnis)
       - "load_id": "run_20250718_123456" (Spezifische Trainingslauf-ID)
       - "model_filename": "model.keras" (oder .joblib)
    ---------------------------------------------------------------------------
    """
    mode = config.get("inference_mode", "load_artifacts_fast")
    logging.info(f"Lade Artefakte im Modus: '{mode}'...")
    
    training_config = None # NEU: Initialisierung

    # ----- Pfade basierend auf dem Modus bestimmen -----
    if mode == 'load_artifacts_path':
        try:
            # Der Schlüssel 'artifacts_path' wurde in der letzten Korrektur festgelegt
            base_path = config['paths']['artifacts_output']
            load_id = config["load_id"]
            
            # KORREKTUR: Lese den Modell-Dateinamen aus der Konfiguration
            model_filename = config["model_filename"] 
            
        except KeyError as e:
            raise ValueError(f"Für Modus 'load_artifacts_path' fehlt der Schlüssel '{e}' in der Konfiguration.")

        run_dir  = os.path.join(base_path, folder_flag, load_id)

        # Verwende den dynamischen Dateinamen für das Modell
        model_path = os.path.join(run_dir, "Models", model_filename)
        
        # Die Namen für Scaler und Features sind oft konsistent
        scaler_path = os.path.join(run_dir, "Scalers", "scaler.joblib")
        features_path = os.path.join(run_dir, "Models", "features.joblib")

        training_config_path = os.path.join(run_dir, "Models", "training_config.json")

        logging.info(f"Lade versionierte Artefakte aus: {run_dir}")

        if os.path.exists(training_config_path):
            try:
                with open(training_config_path, 'r') as f:
                    training_config = json.load(f)
                logging.info("✅ Trainings-Konfiguration geladen.")
            except Exception as e:
                logging.warning(f"Konnte Trainings-Konfiguration nicht laden: {e}")

    elif mode == 'load_artifacts_fast':
        model_path = config.get("model_path_static", "trained_rf_model.joblib")
        scaler_path = config.get("scaler_path_static", "trained_rf_scaler.joblib")
        features_path = config.get("features_path_static", "trained_rf_features.joblib")
        logging.info("Lade Artefakte von statischen Pfaden.")

    else:
        raise ValueError(f"Unbekannter 'inference_mode': '{mode}'. Gültige Modi sind 'load_artifacts_fast' und 'load_artifacts_path'.")

    for path in [model_path, scaler_path, features_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Benötigte Artefakt-Datei wurde nicht gefunden unter: {path}")

    if model_path.endswith((".keras", ".h5")):
        model = tf.keras.models.load_model(model_path)
    elif model_path.endswith(".joblib"):
        model = joblib.load(model_path)
    elif model_path.endswith(".tflite"):
        logging.info("Lade TFLite-Modell mit TensorFlow Lite Interpreter.")
        # Lade das Modell mit dem TFLite Interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        # Reserviere Speicher für die Tensoren des Modells
        interpreter.allocate_tensors()
        # Gib den Interpreter als "Modell"-Objekt zurück
        model = interpreter
    else:
        raise ValueError(f"Unbekannte Modelldatei-Endung. Unterstützt werden .keras, .h5, .joblib. Pfad: {model_path}")

    # Scaler und Features werden immer mit joblib geladen
    scaler = joblib.load(scaler_path)
    features = joblib.load(features_path)

    logging.info("✅ Alle Artefakte erfolgreich geladen.")

    # Rückgabe in der gewünschten Reihenfolge
    return scaler, features, model, training_config

class ModelScalerSaver:
    """
    Eine Klasse zur Kapselung der gesamten Logik zum Speichern von Modellen
    und zugehörigen Artefakten.
    """
    def __init__(self, config: dict, paths: dict):
        """
        Initialisiert den Saver mit der Konfiguration und den Pfaden für einen Lauf.
        """
        self.config = config
        self.paths = paths
        if not paths:
            raise ValueError("Das 'paths'-Dictionary darf nicht None sein.")
        
        # Stelle sicher, dass die Basis-Verzeichnisse existieren
        self._ensure_output_dirs_exist(["Models", "Scalers", "Model_Structures", "Loss_Plots"])

    def _save_config_as_json(self, output_path: str):
        """Speichert die Konfiguration als JSON-Datei."""
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=4, default=str) # default=str für nicht-serialisierbare Typen
            logging.info(f"✅ Trainings-Konfiguration gespeichert unter: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"❌ Fehler beim Speichern der Konfigurationsdatei: {e}", exc_info=True)
            return None

    def _save_structure_plot(self, model, output_path: str) -> str:
        """
        Erstellt und speichert die Modell-Architektur als Bild.
        
        Args:
            model (tf.keras.Model): Das trainierte Keras-Modell.
            output_path (str): Der vollständige Pfad zum Speichern des Bildes.
            
        Returns:
            str: Der Pfad zum gespeicherten Bild oder None bei einem Fehler.
        """
        try:
            # Stellt sicher, dass das Zielverzeichnis existiert
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            tf.keras.utils.plot_model(
                model,
                to_file=output_path,
                show_shapes=True,
                show_layer_activations=True
            )
            print(f"📊 Modellstruktur gespeichert unter: {output_path}")
            return output_path
        except ImportError:
             print("⚠️ Fehler beim Speichern der Modellstruktur: pydot und graphviz müssen installiert sein.")
             print("Führen Sie aus: pip install pydot graphviz")
             print("Stellen Sie sicher, dass Graphviz auch als Systemprogramm installiert und im PATH ist.")
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Modellstruktur: {e}")
            traceback.print_exc()
        
        return None
    

    def save_artifacts(self, model, scaler, **kwargs) -> dict:
        """
        Hauptmethode zum Speichern. Erkennt den Modelltyp und delegiert.
        """
        print(f"--- 🚀 Starte Speichern der Deployment-Artefakte für Modell: {self.config.get('model_name')} ---")
        results = {}
        results.update(self._save_scaler(scaler))

        config_path = os.path.join(self.paths.get("Models"), "training_config.json")

        saved_config_path = self._save_config_as_json(config_path)
        if saved_config_path:
            results["training_config_path"] = saved_config_path
        
        # Smart Dispatch zu den modellspezifischen Speicher-Methoden
        if isinstance(model, tf.keras.Model):
            results.update(self._save_keras_artifacts(model, **kwargs))
        elif isinstance(model, xgb.XGBRegressor):
            results.update(self._save_xgboost_model(model))
        elif isinstance(model, (RandomForestRegressor, MultiOutputRegressor)):
            results.update(self._save_sklearn_model(model))
        else:
            print(f"⚠️ Warnung: Kein spezifischer Speicherpfad für Modelltyp {type(model).__name__} implementiert.")
            
        results.update(self._save_edge_artifacts())
        return results

    def _ensure_output_dirs_exist(self, dir_keys: list):
        """Stellt sicher, dass alle benötigten Ausgabe-Verzeichnisse existieren."""
        try:
            for key in dir_keys:
                dir_path = self.paths.get(key)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(f"❌ Fehler beim Erstellen von Verzeichnissen: {e}")

    def _save_scaler(self, scaler) -> dict:
        """Speichert das Skalierer-Objekt, falls Skalierung aktiviert ist."""
        if (self.config.get("scale_target", False)) and scaler:
            try:
                path = os.path.join(self.paths.get("Scalers"), "scaler.joblib")
                joblib.dump(scaler, path)
                print(f"✅ Skalierer gespeichert unter: {path}")
                return {"scaler_path": path}
            except Exception as e:
                print(f"⚠️ Fehler beim Speichern des Skalierers: {e}", exc_info=True)
        return {}

    def _save_keras_artifacts(self, model: tf.keras.Model, **kwargs) -> dict:
        """Speichert alle Artefakte für ein Keras-Modell."""
        results = {}
        base_filename = f"{self.config['model_name']}_{self.config['dataset'].split('.')[0]}_{self.config['run_id']}"
        
        try:
            path = os.path.join(self.paths.get("Models"), "model.keras")
            model.save(path)
            results["model_path"] = path
            print(f"✅ Normales Keras-Modell gespeichert unter: {path}")
        except Exception as e:
            print(f"❌ Fehler beim Speichern des Keras-Modells: {e}")
            traceback.print_exc() 

        if self.config.get("edge_device", False):
            quantized_path = self._convert_and_quantize_tflite(model)
            if quantized_path:
                results["quantized_model_path"] = quantized_path
        
        if history := kwargs.get("history"):
            plot_path = os.path.join(self.paths.get("Loss_Plots"), f"loss_plot_{self.config['run_id']}.png")
            if saved_plot_path := self._save_loss_plot(history, plot_path):
                results["loss_plot_path"] = saved_plot_path
        
        struct_path = os.path.join(self.paths.get("Model_Structures"), f"structure_{self.config['run_id']}.png")
        if saved_struct_path := self._save_structure_plot(model, struct_path):
            results["model_structure_path"] = saved_struct_path
            
        return results


    def _convert_and_quantize_tflite(self, model: tf.keras.Model) -> str:
        """
        Konvertiert ein Keras-Modell und wendet standardmäßig eine FLOAT16-Quantisierung an.
        """
        try:
            print("--- 🔬 Starte TFLite-Konvertierung mit FLOAT16-Quantisierung ---")
            
            # Handle LSTM/GRU models differently
            if any(isinstance(layer, (tf.keras.layers.LSTM, tf.keras.layers.GRU)) for layer in model.layers):
                print("INFO: LSTM/GRU-Modell erkannt. Wende speziellen Konvertierungsprozess an.")
                
                # Get the input shape from the original model
                input_shape = model.input_shape
                if not input_shape:
                    raise ValueError("Model has no defined input shape")
                
                # Create concrete function with input signature
                input_spec = tf.TensorSpec(shape=(None,) + input_shape[1:], dtype=tf.float32)
                
                # Create the converter with concrete function
                @tf.function(input_signature=[input_spec])
                def model_func(inputs):
                    return model(inputs)
                
                concrete_func = model_func.get_concrete_function()
                converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
                
                # Set optimization options
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.TFLITE_BUILTINS,
                    tf.lite.OpsSet.SELECT_TF_OPS
                ]
                converter._experimental_lower_tensor_list_ops = False
            else:
                # Standard conversion for non-RNN models
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.float16]

            # Perform conversion
            tflite_model_quant = converter.convert()
            
            # Save the model
            tflite_path = os.path.join(self.paths.get("Models"), "model_quant_float16.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model_quant)
            print(f"✅ TFLite-Modell (FLOAT16) gespeichert unter: {tflite_path}")
            return tflite_path
            
        except Exception as e:
            print(f"❌ Fehler bei der TFLite-Quantisierung (FLOAT16): {e}")
            traceback.print_exc()
            return None

    # def _convert_and_quantize_tflite(self, model, representative_dataset_obj) -> str:
    #     """Konvertiert, quantisiert und speichert ein Keras-Modell als .tflite-Datei."""
    #     if not representative_dataset_obj: return None
    #     try:
    #         print("--- 🔬 Starte TFLite-Konvertierung und Quantisierung ---")
            
    #         # Workaround für Kompatibilitätsprobleme
    #         model_config = model.get_config()
    #         fresh_model = tf.keras.Sequential.from_config(model_config)
    #         fresh_model.set_weights(model.get_weights())
            
    #         converter = tf.lite.TFLiteConverter.from_keras_model(fresh_model)
    #         converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #         converter.representative_dataset = lambda: iter(representative_dataset_obj)
            
    #         if any(isinstance(layer, (tf.keras.layers.LSTM, tf.keras.layers.GRU)) for layer in fresh_model.layers):
    #             converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    #             converter._experimental_lower_tensor_list_ops = False

    #         tflite_model_quant = converter.convert()
            
    #         tflite_path = os.path.join(self.paths.get("Models"), f"{self.config['model_name']}_{self.config['dataset'].split('.')[0]}_{self.config['run_id']}.tflite")
    #         with open(tflite_path, 'wb') as f: f.write(tflite_model_quant)
    #         print(f"✅ TFLite-Modell (quantisiert) gespeichert unter: {tflite_path}")
    #         return tflite_path
    #     except Exception as e:
    #         print(f"❌ Fehler bei der TFLite-Quantisierung: {e}", exc_info=True)
    #         return None
    
    def _save_loss_plot(self, history, output_path: str):
        """
        Erstellt und speichert einen Plot des Trainings- & Validierungsverlusts
        sowie der Metriken aus dem Keras-History-Objekt.
        
        Args:
            history (tf.keras.callbacks.History): Das History-Objekt von model.fit().
            output_path (str): Der vollständige Pfad zum Speichern des Plots.
        """
        if not hasattr(history, 'history') or not history.history:
            print("⚠️ Kein gültiges History-Objekt zum Plotten vorhanden.")
            return

        history_dict = history.history
        
        # Schlüssel für Loss und die erste Metrik dynamisch finden
        loss_keys = sorted([k for k in history_dict if 'loss' in k])
        # Alle anderen Schlüssel werden als Metriken interpretiert
        metric_keys = sorted([k for k in history_dict if 'loss' not in k])
        
        # Erstelle ein 2x1 Subplot-Grid, falls Metriken vorhanden sind, sonst nur 1x1
        num_subplots = 2 if metric_keys else 1
        fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 6 * num_subplots), sharex=True)
        
        # Sicherstellen, dass 'axes' immer ein Array ist, auch bei nur einem Subplot
        if num_subplots == 1:
            axes = [axes]

        # --- Subplot 1: Loss ---
        for key in loss_keys:
            axes[0].plot(history_dict[key], label=key)
        axes[0].set_title(f"Trainings- & Validierungs-Loss für {self.config.get('model_name')}")
        # Extrahiert den Loss-Namen aus der Konfiguration
        loss_name_raw = self.config.get("loss", "loss")
        loss_name = str(loss_name_raw).split('.')[-1].replace("()", "") if not isinstance(loss_name_raw, str) else loss_name_raw
        axes[0].set_ylabel(f'Loss ({loss_name})')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_yscale('log') # Log-Skala ist oft hilfreich für Loss-Plots

        # --- Subplot 2: Metriken (falls vorhanden) ---
        if num_subplots > 1:
            primary_metric_name = metric_keys[0].replace('val_', '') # z.B. 'mae'
            for key in metric_keys:
                axes[1].plot(history_dict[key], label=key)
            axes[1].set_title('Trainings- & Validierungs-Metrik')
            axes[1].set_ylabel(primary_metric_name.upper())
            axes[1].set_xlabel('Epoche')
            axes[1].legend()
            axes[1].grid(True)
        else:
            axes[0].set_xlabel('Epoche')

        fig.suptitle(f"Trainingsverlauf für Run: {self.config.get('run_id')}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Platz für den suptitle lassen
        
        try:
            plt.savefig(output_path)
            print(f"✅ Loss Plot gespeichert unter: {output_path}")
        except Exception as e:
            print(f"❌ FEHLER beim Speichern des Loss Plots unter '{output_path}': {e}")
        finally:
            plt.close(fig) # Schließt die Figur, um Speicher freizugeben und Konflikte zu vermeiden


    # def _save_keras_model(self, model: tf.keras.Model, **kwargs) -> dict:
    #     """
    #     Orchestriert das Speichern eines Keras-Modells und aller zugehörigen Artefakte.
    #     """
    #     results = {}
    #     history = kwargs.get("history")
    #     base_filename = f"{self.config['model_name']}_{self.config['dataset'].split('.')[0]}_{self.config['run_id']}"

    #     # 1. Normales Keras-Modell speichern (unverändert)
    #     try:
    #         model_path = os.path.join(self.paths.get("Models"), f"{base_filename}.keras")
    #         model.save(model_path)
    #         results["model_path"] = model_path
    #         print(f"✅ Normales Keras-Modell gespeichert unter: {model_path}")
    #     except Exception as e:
    #         print(f"❌ Fehler beim Speichern des Keras-Modells: {e}")
    #         traceback.print_exc()

    #     # 2. Quantisiertes TFLite-Modell speichern (NUR wenn Flag gesetzt ist)
    #     if self.config.get("edge_device", False):
    #         # Der Aufruf ist jetzt viel einfacher
    #         quantized_path = self._convert_and_quantize_tflite(
    #             model=model,
    #             representative_dataset_gen=kwargs.get("representative_dataset")
    #         )
    #         if quantized_path:
    #             results["quantized_model_path"] = quantized_path

    #     # 3. Plots speichern (unverändert)
    #     if history:
    #         plot_path = os.path.join(self.paths.get("Loss_Plots"), f"loss_plot_{self.config['run_id']}.png")
    #         saved_plot_path = self._save_loss_plot(history, plot_path)
    #         if saved_plot_path:
    #             results["loss_plot_path"] = saved_plot_path
        
    #     struct_path = os.path.join(self.paths.get("Model_Structures"), f"structure_{self.config['run_id']}.png")
    #     saved_struct_path = self._save_structure_plot(model, struct_path)
    #     if saved_struct_path:
    #         results["model_structure_path"] = saved_struct_path
            
    #     return results
    

    def _save_sklearn_model(self, model) -> dict:
        """Speichert ein Scikit-learn-Modell."""
        try:
            model_dir = self.paths.get("Models")
            model_name = self.config.get("model_name", "sklearn_model")
            dataset = self.config.get("dataset", "data")
            model_filename = "model.joblib"
            model_path = os.path.join(model_dir, "model.joblib")
            joblib.dump(model, model_path, compress=3)
            print(f"📤 Scikit-learn-Modell gespeichert unter: {model_path}")
            return {"model_path": model_path}
        except Exception as e:
            print(f"❌ Fehler beim Speichern des Scikit-learn-Modells: {e}", exc_info=True)
            return {}

    def _save_xgboost_model(self, model: xgb.XGBRegressor) -> dict:
        """Speichert ein XGBoost-Modell."""
        try:
            model_dir = self.paths.get("Models")
            model_name = self.config.get("model_name", "xgb_model")
            dataset = self.config.get("dataset", "data")
            model_filename = f"{model_name}_{dataset}_{self.config['run_id']}_{self.config['time_stamp']}.json"
            model_path = os.path.join(model_dir, "model.json")
            model.save_model(model_path)
            print(f"📤 XGBoost-Modell gespeichert unter: {model_path}")
            return {"model_path": model_path}
        except Exception as e:
            print(f"❌ Fehler beim Speichern des XGBoost-Modells: {e}", exc_info=True)
            return {}
            
    def _save_edge_artifacts(self) -> dict:
        """
        Speichert optionale Artefakte für die Edge-Bereitstellung.
        Diese Methode ist jetzt Teil der Klasse und verwendet self.config und self.paths.
        """
        # Prüfe das Flag direkt aus der Instanz-Konfiguration
        if not self.config.get("enable_edge", False):
            return {} # Gib ein leeres Dictionary zurück, wenn nichts zu tun ist
            
        try:
            print("🧾 Erstelle Edge-Artefakte...")
            # Greife auf die Pfade über die Instanz zu
            model_dir = self.paths.get("Models", os.path.join(self.paths.get("output"), "Models"))
            edge_dir = os.path.join(model_dir, "edge_artifacts")
            os.makedirs(edge_dir, exist_ok=True)

            # Greife auf die Konfiguration über die Instanz zu
            if "scaler_mean" in self.config and "scaler_scale" in self.config:
                np.save(os.path.join(edge_dir, "scaler_mean.npy"), self.config["scaler_mean"])
                np.save(os.path.join(edge_dir, "scaler_scale.npy"), self.config["scaler_scale"])

            if "base_features" in self.config:
                with open(os.path.join(edge_dir, "features.json"), "w") as f:
                    json.dump(self.config["base_features"], f)

            print(f"✅ Edge-Artefakte gespeichert unter: {edge_dir}")
            return {"edge_artifacts": edge_dir}
            
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Edge-Artefakte: {e}")
            print(traceback.format_exc())
            
        return {} 
