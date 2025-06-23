import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import random
from typing import List, Tuple
import json
import traceback

from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    mean_squared_log_error,
    median_absolute_error
)

# -------------------------------------------
# Hilfsfunktionen: Visualisierung & Modell speichern
# -------------------------------------------

def save_loss_plot(history: dict, model_name: str, dataset: str, 
                   run_id: str, timestamp: str, output_dir: str) -> str:
    """Speichert Loss-Kurven mit standardisiertem Dateinamen."""
    plt.figure(figsize=(10, 6))

    if 'loss' in history:
        plt.plot(history['loss'], label='Train Loss')
    else:
        print("WARNUNG: 'loss' nicht im history-Objekt für Loss-Plot gefunden.")

    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Validation Loss')

    plt.title(f'Loss Curve - {model_name} - {dataset} (Run: {run_id})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if 'loss' in history or 'val_loss' in history:
        plt.legend()
    plt.grid(True)

    filename = f"LossPlot_{run_id}_{model_name}_{dataset}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)

    try:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(filepath, bbox_inches='tight')
        print(f"Loss Plot gespeichert unter: {filepath}")
    except Exception as e:
        print(f"FEHLER beim Speichern des Loss Plots unter '{filepath}': {e}")
        filepath = None
    finally:
        plt.close()

    return filepath

def save_model_with_version(model, directory, model_name, dataset, run_id, timestamp=None):
    """Speichert Modell mit standardisiertem Dateinamen"""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"Model_{run_id}_{model_name}_{dataset}_{timestamp}.keras"
    model_path = os.path.join(directory, filename)
    model.save(model_path)
    return model_path




# -------------------------------------------
# Hilfsfunktionen: Daten vorbereiten & Skalierung
# -------------------------------------------

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


def setup_experiment(config: dict) -> tuple[dict, dict]:
    """
    Initialisiert das Experiment, generiert Run-ID und erstellt notwendige Output-Ordnerstrukturen.
    
    Args:
        config (dict): Konfigurationsdictionary mit mindestens 'paths' → 'output'.
    
    Returns:
        tuple: (aktualisierte config, dictionary mit erstellten Pfaden)
    """
    # Zeitstempel und Run-ID setzen
    if "time_stamp" not in config or config["time_stamp"] is None:
        config["time_stamp"] = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    if "run_id" not in config or config["run_id"] is None:
        config["run_id"] = f"{config['time_stamp']}_{random.randint(1000, 9999)}"

    # Output-Basisverzeichnis prüfen
    try:
        base_output_path = config["paths"]["output"]
    except KeyError:
        raise ValueError("Pfad 'paths' → 'output' fehlt in der Konfiguration.")

    # Strukturierte Unterordner definieren
    paths = {
        "Base_Output_Path": base_output_path,
        "Models": os.path.join(base_output_path, "Models"),
        "Model_Structures": os.path.join(base_output_path, "Model Structures"),
        "Model_Summaries": os.path.join(base_output_path, "Model Summaries"),
        "Prediction_Plots": os.path.join(base_output_path, "Prediction Plots"),
        "Loss_Plots": os.path.join(base_output_path, "Loss Plots"),
        "Error_Metrics": os.path.join(base_output_path, "Error Metrics"),
        "Prediction_Data": os.path.join(base_output_path, "Prediction Data"),
        "Scalers": os.path.join(base_output_path, "Scalers")
    }

    # Ordner erstellen, falls nicht vorhanden
    for path in paths.values():
        os.makedirs(path, exist_ok=True)

    print(f"✅ Experiment-Setup abgeschlossen. Run ID: {config['run_id']}")
    print(f"📁 Ausgabepfade initialisiert unter: {base_output_path}")

    return config, paths


def save_metrics_to_summary(
    config: dict,
    metrics: dict,
    prediction_data_path: str = None,
    power_time: float = None
) -> None:
    """
    Speichert die gesamte Config und Metriken in einer CSV-Datei zur Nachvollziehbarkeit.
    """

    # Flattened Config → alle Parameter
    config_flat = flatten_config(config)
    run_id = config_flat.get("run_id", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    start_timestamp = config_flat.get("start_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    row = {
        **config_flat,
        "run_id": run_id,
        "timestamp": start_timestamp,
        "prediction_data_path": prediction_data_path or config_flat.get("prediction_data_path", "N/A"),
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "power_time": power_time,
    }

    for key, value in metrics.items():
        if isinstance(value, (list, np.ndarray)):
            row[key] = ";".join(map(str, value))
        elif isinstance(value, dict):
            row[key] = json.dumps(value)
        else:
            row[key] = value

    # Zielpfad
    output_dir = config.get("output_dir") or config.get("paths", {}).get("output", "./Output")
    metrics_dir = os.path.join(output_dir, "Error_Metrics_1")
    os.makedirs(metrics_dir, exist_ok=True)
    summary_path = os.path.join(metrics_dir, "metrics_summary.csv")

    # Schreiben oder Anhängen
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        if run_id in df["run_id"].values:
            df.loc[df["run_id"] == run_id, row.keys()] = row.values()
        else:
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(summary_path, index=False)

# -------------------------------------------
# Modellbewertung
# -------------------------------------------

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



# -------------------------------------------
# Gemeinsame Ergebnisse speichern
# -------------------------------------------

def _save_common_results(
    config: dict,
    pred_orig: np.ndarray,
    true_orig: np.ndarray,
    dates: pd.DatetimeIndex,
    metrics_values: dict,
    paths: dict,
    power_time: float,
    scaler: object
) -> dict:
    print("🟢 Starte Speichern der gemeinsamen Ergebnisse...")

    results = {
        "scaler_path": None,
        "prediction_file": None,
        "metrics_summary_path": None,
    }

    # === Verzeichnisse sicherstellen ===
    try:
        print(f"📁 Basis-Ausgabeverzeichnis: {paths.get('Base_Output_Path')}")
        os.makedirs(paths["Prediction_Data"], exist_ok=True)
        os.makedirs(paths["Error_Metrics"], exist_ok=True)
    except Exception as e:
        print(f"❌ Fehler beim Erstellen von Verzeichnissen: {e}")
        print(traceback.format_exc())

    # === Skalierer speichern ===
    if scaler is not None:
        try:
            scaler_filename = f"scaler_{config.get('run_id')}_{config.get('time_stamp')}.joblib"
            scaler_path = os.path.join(paths["Scalers"], scaler_filename)
            os.makedirs(paths["Scalers"], exist_ok=True)
            joblib.dump(scaler, scaler_path)
            results["scaler_path"] = scaler_path
            print(f"✅ Skalierer gespeichert unter: {scaler_path}")
        except Exception as e:
            print(f"⚠️ Fehler beim Speichern des Skalierers: {e}")
            print(traceback.format_exc())

    # === Vorhersagedaten speichern ===
    try:
        pred_flat = np.array(pred_orig).flatten()
        true_flat = np.array(true_orig).flatten()

        prediction_filename = f"predictions_{config.get('run_id')}_{config.get('time_stamp')}.csv"
        prediction_path = os.path.join(paths["Prediction_Data"], prediction_filename)

        save_prediction_data(
            config=config,
            y_true=true_flat,
            y_pred=pred_flat,
            dates=dates,
            output_path=prediction_path
        )
        results["prediction_file"] = prediction_path
        print(f"✅ Vorhersagedaten gespeichert unter: {prediction_path}")
    except Exception as e:
        print(f"❌ Fehler beim Speichern der Vorhersagedaten: {e}")
        print(traceback.format_exc())

    # === Metriken speichern ===
    try:
        if results["prediction_file"]:
            save_metrics_to_summary(
                config=config,
                metrics=metrics_values,
                prediction_data_path=results["prediction_file"],
                power_time=power_time
            )
            summary_path = os.path.join(paths["Error_Metrics"], "metrics_summary.csv")
            results["metrics_summary_path"] = summary_path
            print(f"✅ Metriken gespeichert unter: {summary_path}")
        else:
            print("⚠️ Keine Vorhersagedatei vorhanden – Metriken nicht gespeichert.")
    except Exception as e:
        print(f"❌ Fehler beim Speichern der Metriken: {e}")
        print(traceback.format_exc())

    print("✅ Ergebnis-Speicherung abgeschlossen.")
    return results





    


