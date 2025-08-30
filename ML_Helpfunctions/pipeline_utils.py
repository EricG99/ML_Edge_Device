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
import socket


import psutil
import time


import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

import tensorflow as tf

# Der Rest des Codes bleibt gleich
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
    
def get_local_ip():
    """Ermittelt die lokale IP-Adresse des Geräts, um den Zugriffslink anzuzeigen."""
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = "127.0.0.1"
    finally:
        if s:
            s.close()
    return ip_address

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


import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, date

def _to_jsonable(obj):
    """
    Wandelt beliebige Objekte rekursiv in JSON-serialisierbare Typen um.
    Behandelt: pathlib.Path, numpy (inkl. np.generic), datetime/pd.Timestamp,
    Dict/List/Tuple/Set und verschachtelte Strukturen.
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return repr(obj)



def save_metrics_summary(
    metrics: dict,
    run_config: dict,
    training_config: dict,
    paths: dict,
    extra_info: dict | None = None
) -> str:
    """
    Speichert die Metriken als:
      1) Run-spezifische und modell-spezifische JSON (neue Datei je Variante).
      2) Aggregierte CSV 'ErrorMetrics_all_runs.csv' (wird immer ergänzt).
    Rückgabe: absoluter Pfad zur JSON-Datei.
    """
    out_dir = (paths or {}).get("Error_Metrics", (paths or {}).get("Prediction_Data", "."))
    out_dir = os.fspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # --- KORRIGIERTE VERSION START ---
    run_id = run_config.get("run_id", "run")
    model_name = run_config.get("model_name", "model")
    dataset_raw = run_config.get("dataset", "data")
    dataset_clean = os.path.splitext(os.path.basename(dataset_raw))[0]
    model_tag = (extra_info or {}).get("model_tag")

    # Baue den Basis-Dateinamen sauber zusammen
    base_parts = [
        "ErrorMetrics",
        run_id,
        model_name,
        dataset_clean
    ]
    base_name = "_".join(filter(None, base_parts))

    # Füge den Modell-Tag sauber an
    if model_tag:
        final_filename = f"{base_name}__{model_tag}.json"
    else:
        final_filename = f"{base_name}.json"
        
    json_path = os.path.join(out_dir, final_filename)
    # --- KORRIGIERTE VERSION ENDE ---

    # JSON-Payload zusammenbauen (Logik bleibt gleich)
    payload = {
        "run": {
            "run_id": run_id,
            "model_name": model_name,
            "dataset": dataset_raw,
            "time_stamp": run_config.get("time_stamp", "timestamp")
        },
        "metrics": metrics or {},
        "training_config": training_config or {},
        "inference_config": run_config or {}
    }
    if extra_info:
        payload["extra_info"] = extra_info

    payload_jsonable = _to_jsonable(payload)

    # 1) Run-spezifische JSON schreiben
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload_jsonable, f, indent=2, ensure_ascii=False)

    # 2) Aggregierte CSV-Zeile anhängen
    flat = {
        "run_id": run_id,
        "model_name": model_name,
        "dataset": dataset_raw,
        "time_stamp": run_config.get("time_stamp", "timestamp"),
        "model_variant": model_tag, # Spalte für die Aggregation hinzufügen
        "json_path": os.fspath(json_path)
    }
    for k in ["MAE", "MSE", "RMSE", "MAPE", "SMAPE", "R2"]:
        if k in (metrics or {}):
            try:
                flat[k] = float(metrics[k])
            except Exception:
                flat[k] = metrics[k]

    if extra_info and "predictions_file_path" in extra_info:
        flat["predictions_file_path"] = os.fspath(extra_info["predictions_file_path"])

    agg_path = os.path.join(out_dir, "ErrorMetrics_all_runs.csv")
    pd.DataFrame([flat]).to_csv(
        agg_path,
        mode="a",
        header=not os.path.exists(agg_path),
        index=False
    )

    return os.path.abspath(json_path)

def append_prediction_step(
    config: dict,
    date: datetime,
    true_value: float | int | None,
    forecast: list | np.ndarray,
    inference_time_s: float | None,
    total_time_s: float | None,
    cpu_percent: float | None = None,   # CPU in %
    ram_mb: float | None = None,        # RAM in MB (RSS)
    # --- NEUE ZEILE START ---
    ram_percent: float | None = None,   # RAM in %
    # --- NEUE ZEILE ENDE ---
    breakdown: dict | None = None,
    output_path: str | None = None
) -> str:
    """
    Hängt EINEN Inferenz-Schritt als Zeile an eine CSV (oder erzeugt sie, falls neu).
    Spalten:
      date, true_value, pred_h1..pred_h{H}, inference_time_s, total_time_s, cpu_percent, ram_mb, (+ optionale Breakdown-Spalten)
    Rückgabe: absoluter Pfad der Datei.
    """
    model_name = config.get("model_name", "model")
    dataset = config.get("dataset", "data")
    run_id = config.get("run_id", "run")
    timestamp = config.get("time_stamp", "timestamp")
    output_dir = config.get("paths", {}).get("Prediction_Data", ".")
    horizon = int(config.get("horizon", 1))

    # Zielpfad
    if output_path is None:
        filename = f"StepPredictions_{run_id}_{model_name}_{dataset}_{timestamp}.csv"
        output_path = os.path.join(output_dir, filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Forecast normalisieren
    forecast = np.asarray(forecast).reshape(-1).tolist()
    if len(forecast) < horizon:
        forecast += [np.nan] * (horizon - len(forecast))
    elif len(forecast) > horizon:
        forecast = forecast[:horizon]

    row = {
        "date": pd.Timestamp(date).to_pydatetime().isoformat(),
        "true_value": true_value,
        "inference_time_s": float(inference_time_s) if inference_time_s is not None else None,
        "total_time_s": float(total_time_s) if total_time_s is not None else None,
        "cpu_percent": float(cpu_percent) if cpu_percent is not None else None,
        "ram_mb": float(ram_mb) if ram_mb is not None else None,
        # --- NEUE ZEILE START ---
        "ram_percent": float(ram_percent) if ram_percent is not None else None,
        # --- NEUE ZEILE ENDE ---
    }
    for h in range(horizon):
        row[f"pred_h{h+1}"] = forecast[h]

    if breakdown:
        for k, v in breakdown.items():
            row[str(k)] = float(v) if v is not None else None

    write_header = not os.path.exists(output_path)
    pd.DataFrame([row]).to_csv(output_path, index=False, mode="a", header=write_header)
    return os.path.abspath(output_path)


import numpy as np
from typing import Optional, Dict, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_squared_log_error,
    median_absolute_error,
)

def evaluate_all_metrics(
    y_true,
    y_pred,
    y_train: Optional[np.ndarray] = None,
    horizon: Optional[int] = None,
    alpha: float = 0.8,
) -> Dict[str, Any]:
    """
    Berechnet diverse Fehlermetriken für 1- oder Multi-Step-Vorhersagen, NaN-sicher.

    Args:
        y_true: Wahrer Wert(e), Shape (N,) oder (N,H)
        y_pred: Vorhersage(n), Shape (N,) oder (N,H)
        y_train: Optional Trainings-Zeitreihe für MASE (1D)
        horizon: Erwarteter Horizont H; wenn None -> aus Daten abgeleitet (min der Spalten)
        alpha: 0<alpha<=1, Abklingfaktor für weighted MAE (höhere Gewichte für nahe Schritte)

    Returns:
        dict mit Metriken. Bei H==1: Skalare; bei H>1: Listenlängen H.
        Keys: mse, rmse, mae, r2, mape, smape, wape, msle, median_ae, mase, weighted_mae
    """
    eps = 1e-8

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Vereinheitlichen auf 2D (N,H)
    if y_true.ndim == 1:
        y_true = y_true[:, None]
    if y_pred.ndim == 1:
        y_pred = y_pred[:, None]

    # Auf gleiche N und H bringen
    N = min(y_true.shape[0], y_pred.shape[0])
    H_true = y_true.shape[1]
    H_pred = y_pred.shape[1]
    H = min(horizon if horizon is not None else max(H_true, H_pred), H_true, H_pred)

    y_true = y_true[:N, :H]
    y_pred = y_pred[:N, :H]

    # Hilfsfunktionen (numerisch stabil)
    def _smape(yt, yp):
        denom = np.maximum((np.abs(yt) + np.abs(yp)) / 2.0, eps)
        return float(np.mean(np.abs(yp - yt) / denom) * 100.0)

    def _wape(yt, yp):
        denom = np.maximum(np.sum(np.abs(yt)), eps)
        return float(np.sum(np.abs(yt - yp)) / denom * 100.0)

    # Container
    out = {
        "mse": [],
        "rmse": [],
        "mae": [],
        "r2": [],
        "mape": [],
        "smape": [],
        "wape": [],
        "msle": [],
        "median_ae": [],
    }

    # Schrittweise (pro Horizont-Spalte) – mit NaN/Inf-Filter
    for h in range(H):
        yt = y_true[:, h]
        yp = y_pred[:, h]
        mask = np.isfinite(yt) & np.isfinite(yp)
        yt = yt[mask]
        yp = yp[mask]

        if yt.size == 0:
            # Keine gültigen Paare
            out["mse"].append(np.nan)
            out["rmse"].append(np.nan)
            out["mae"].append(np.nan)
            out["r2"].append(np.nan)
            out["mape"].append(np.nan)
            out["smape"].append(np.nan)
            out["wape"].append(np.nan)
            out["msle"].append(np.nan)
            out["median_ae"].append(np.nan)
            continue

        mse_val = mean_squared_error(yt, yp)
        out["mse"].append(float(mse_val))
        out["rmse"].append(float(np.sqrt(mse_val)))
        out["mae"].append(float(mean_absolute_error(yt, yp)))

        # R^2 kann bei konstantem yt fehlschlagen
        try:
            out["r2"].append(float(r2_score(yt, yp)))
        except ValueError:
            out["r2"].append(np.nan)

        # MAPE stabil (Division durch ~0 vermeiden)
        denom = np.maximum(np.abs(yt), eps)
        out["mape"].append(float(np.mean(np.abs((yt - yp) / denom)) * 100.0))

        # SMAPE/WAPE stabil
        out["smape"].append(_smape(yt, yp))
        out["wape"].append(_wape(yt, yp))

        # MSLE braucht Nicht-Negativität
        yt_pos = np.clip(yt, 0.0, None)
        yp_pos = np.clip(yp, 0.0, None)
        try:
            out["msle"].append(float(mean_squared_log_error(yt_pos, yp_pos)))
        except ValueError:
            out["msle"].append(np.nan)

        out["median_ae"].append(float(median_absolute_error(yt, yp)))

    # MASE (auf Basis des mittleren 1-Schritt-Naivfehlers im Train)
    if y_train is not None:
        yt = np.asarray(y_train, dtype=float).ravel()
        yt = yt[np.isfinite(yt)]
        if yt.size > 1:
            diffs = np.diff(yt)
            denom = np.mean(np.abs(diffs[np.isfinite(diffs)])) if diffs.size else np.nan
            if denom is not None and np.isfinite(denom) and denom > eps:
                if H == 1:
                    mase = out["mae"][0] / denom
                else:
                    mase = float(np.nanmean(out["mae"]) / denom)
            else:
                mase = np.nan
        else:
            mase = np.nan
    else:
        mase = np.nan

    # Weighted MAE über den Horizont (größeres Gewicht nahe in der Zukunft)
    if H == 1:
        weighted_mae = out["mae"][0]
    else:
        # Gewichte: alpha^0, alpha^1, ..., alpha^(H-1) (höchstes Gewicht bei h=0)
        w = np.array([alpha ** i for i in range(H)], dtype=float)
        # Nur gültige MAE-Werte berücksichtigen
        mae_arr = np.array(out["mae"], dtype=float)
        valid = np.isfinite(mae_arr)
        if valid.any():
            w_eff = w[valid]
            # normierte gewichtete Summe
            weighted_mae = float(np.sum(w_eff * mae_arr[valid]) / np.sum(w_eff))
        else:
            weighted_mae = np.nan

    # Bei H==1: in Skalare verwandeln (kompatibel zu bisherigem Verhalten)
    if H == 1:
        scalar_out = {k: (v[0] if isinstance(v, list) else v) for k, v in out.items()}
        scalar_out["mase"] = float(mase) if np.isfinite(mase) else np.nan
        scalar_out["weighted_mae"] = float(weighted_mae) if np.isfinite(weighted_mae) else np.nan
        return scalar_out

    # Multi-Step: Listen + aggregierte Kennzahlen ergänzen
    out["mase"] = float(mase) if np.isfinite(mase) else np.nan
    out["weighted_mae"] = float(weighted_mae) if np.isfinite(weighted_mae) else np.nan
    return out

def save_prediction_data(
    config: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray,
    output_path: str = None,
    inference_times: np.ndarray | None = None,   # optional: je Sample
    total_times: np.ndarray | None = None,       # optional: je Sample
    cpu_percents: np.ndarray | None = None,      # optional: je Sample
    ram_mbs: np.ndarray | None = None,           # optional: je Sample
    breakdowns: list[dict] | None = None         # optional: je Sample
) -> str:
    """
    Speichert Vorhersagedaten (Batch) mit Zeitstempeln.
    Erwartet y_true/y_pred flach (num_samples*horizon).
    """
    print("--- DEBUG: FÜHRE save_prediction_data AUS (batch, mit Zeiten & Ressourcen) ---")

    model_name = config.get("model_name", "model")
    dataset = config.get("dataset", "data")
    run_id = config.get("run_id", "run")
    timestamp = config.get("time_stamp", "timestamp")
    output_dir = config.get("paths", {}).get("Prediction_Data", ".")
    horizon = int(config.get("horizon", 1))

    num_samples = len(dates)
    if len(y_true) != num_samples * horizon:
        raise ValueError(f"Länge von y_true ({len(y_true)}) stimmt nicht mit num_samples ({num_samples}) * horizon ({horizon}) überein.")
    if len(y_pred) != num_samples * horizon:
        raise ValueError(f"Länge von y_pred ({len(y_pred)}) stimmt nicht mit num_samples ({num_samples}) * horizon ({horizon}) überein.")

    y_true_reshaped = np.asarray(y_true).reshape(num_samples, horizon)
    y_pred_reshaped = np.asarray(y_pred).reshape(num_samples, horizon)

    if inference_times is not None and len(inference_times) != num_samples:
        raise ValueError("inference_times muss Länge num_samples haben.")
    if total_times is not None and len(total_times) != num_samples:
        raise ValueError("total_times muss Länge num_samples haben.")
    if cpu_percents is not None and len(cpu_percents) != num_samples:
        raise ValueError("cpu_percents muss Länge num_samples haben.")
    if ram_mbs is not None and len(ram_mbs) != num_samples:
        raise ValueError("ram_mbs muss Länge num_samples haben.")
    if breakdowns is not None and len(breakdowns) != num_samples:
        raise ValueError("breakdowns muss Länge num_samples haben.")

    rows = []
    for i in range(num_samples):
        row = {"date": pd.Timestamp(dates[i]).isoformat()}
        row["true_value"] = y_true_reshaped[i, 0] if horizon >= 1 else None
        for h in range(horizon):
            row[f"pred_h{h+1}"] = y_pred_reshaped[i, h]

        row["inference_time_s"] = float(inference_times[i]) if inference_times is not None else None
        row["total_time_s"] = float(total_times[i]) if total_times is not None else None
        row["cpu_percent"] = float(cpu_percents[i]) if cpu_percents is not None else None
        row["ram_mb"] = float(ram_mbs[i]) if ram_mbs is not None else None

        if breakdowns and breakdowns[i]:
            for k, v in breakdowns[i].items():
                row[str(k)] = float(v) if v is not None else None

        rows.append(row)

    df = pd.DataFrame(rows)
    if output_path is None:
        filename = f"PredictionData_{run_id}_{model_name}_{dataset}_{timestamp}.csv"
        output_path = os.path.join(output_dir, filename)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Vorhersagedatei gespeichert unter: {output_path}")
    return os.path.abspath(output_path)


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
    


    # Die Unterordner für den Lauf werden relativ zum neuen run_output_path erstellt
    run_subfolders = {
        "Base_Output_Path": run_output_path,
        "Models": run_output_path / "Models",
        "Scalers": run_output_path / "Scalers",
        "Prediction_Data": run_output_path / "Prediction_Data",
        "Model_Structures": run_output_path / "Model_Structures",
        "Error_Metrics": run_output_path / "Error_Metrics",
        "Model_Summaries": run_output_path / "Model_Summaries",
        "Prediction_Plots": run_output_path / "Prediction_Plots",
        "Loss_Plots": run_output_path / "Loss_Plots"
    }

    # Alle Pfade zusammenführen und Konfiguration aktualisieren
    all_paths = {**paths, **input_subfolders, **run_subfolders}
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

# In Pipeline_Utils.py
import psutil
import logging

# Kleiner Priming-Status für Windows/Threads
_CPU_PRIMED = False

def get_cpu_usage() -> float:
    """
    Liefert die systemweite CPU-Last in Prozent (0..100).
    Vermeidet das bekannte '0.0' beim ersten Aufruf durch kurzes Priming.
    """
    global _CPU_PRIMED
    try:
        # Beim ersten Aufruf einmal kurz blockierend messen (Priming)
        if not _CPU_PRIMED:
            try:
                psutil.cpu_percent(interval=0.15)  # ~150 ms
            finally:
                _CPU_PRIMED = True

        # Danach non-blocking; wenn 0.0 zurückkommt, einmal kurz nachmessen
        val = psutil.cpu_percent(interval=None)
        if val == 0.0:
            val = psutil.cpu_percent(interval=0.15)
        return float(val)
    except Exception:
        logging.exception("get_cpu_usage() fehlgeschlagen.")
        return 0.0


def get_memory_usage():
    """
    Ermittelt die aktuelle System-RAM-Auslastung mithilfe von psutil.

    Returns:
        dict: Ein Dictionary mit den Werten für Gesamt-RAM (GB),
              genutzten RAM (GB) und die prozentuale Auslastung.
              Gibt 'N/A' zurück, falls psutil nicht verfügbar ist.
    """
    try:
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "percent": mem.percent
        }
    except (ImportError, AttributeError):
        # Fallback, falls psutil nicht installiert ist oder ein Fehler auftritt
        logging.warning("psutil nicht gefunden oder Fehler beim Auslesen. RAM-Nutzung kann nicht ermittelt werden.")
        return {
            "total_gb": "N/A",
            "used_gb": "N/A",
            "percent": 0
        }

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
    Achtung: Für INT8-TFLite wird korrekt mit scale/zero_point quantisiert.
    """
    start_time = time.perf_counter()

    if isinstance(model, tf.lite.Interpreter):
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        tensor = np.asarray(input_data)

        in_det = input_details[0]
        if in_det['dtype'] == np.int8:
            scale, zero_point = in_det.get('quantization', (0.0, 0))
            if scale == 0.0:
                tensor = tensor.astype(np.int8)
            else:
                tensor = np.round(tensor / scale + zero_point).astype(np.int8)
        else:
            tensor = tensor.astype(in_det['dtype'])

        model.set_tensor(in_det['index'], tensor)
        model.invoke()
        prediction = model.get_tensor(output_details[0]['index'])
    else:
        prediction = model.predict(input_data)

    duration_ms = (time.perf_counter() - start_time) * 1000.0
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


def create_representative_dataset_generator(dataset, config):
    """
    Erstellt einen korrekten Generator für die TFLite INT8-Quantisierung.
    """
    def generator():
        for data_sample in dataset.take(100):  
            if isinstance(data_sample, tuple):
                input_data = data_sample[0]
            else:
                input_data = data_sample
            yield [tf.cast(input_data, tf.float32)]

    return generator


def load_model_artifacts_for_inference(config: dict, folder_flag: str) -> tuple:
    """
    Lädt robust die notwendigen Artefakte (Scaler, Features, Modell) für die Inferenz.
    Unterstützte Modelle: .keras/.h5 (TF/Keras), .joblib (sklearn), .tflite (TFLite), .json (XGBoost).
    Rückgabe: (scaler, features, model, training_config, y_scaler)
    """
    import os, json, logging, joblib

    # <<< Wichtig: TensorFlow anfangs importieren (oder sauber abfangen) >>>
    try:
        import tensorflow as tf  # benötigt für .keras/.h5/.tflite
    except Exception:
        tf = None  # wir prüfen vor jeder Nutzung und geben eine klare Fehlermeldung

    mode = config.get("inference_mode", "load_artifacts_fast")
    logging.info(f"Lade Artefakte im Modus: '{mode}'...")

    scaler = None
    y_scaler = None
    features = None
    model = None
    training_config = None

    # ----- Pfade bestimmen -----
    if mode == 'load_artifacts_path':
        try:
            base_path = config['paths'].get('output') or config.get('artifacts_path')
            load_id = config["load_id"]
            model_filename = config.get("model_filename", "model.joblib")
        except KeyError as e:
            raise ValueError(f"Für Modus 'load_artifacts_path' fehlt der Schlüssel '{e}' in der Konfiguration.")

    base_path = Path(config["paths"]["output"]) / folder_flag / run_id
    models_path = base_path / "Models"
    scalers_path = base_path / "Scalers"

    # Trainings-Konfiguration laden
    training_config_path = models_path / "training_config.json"
    if not training_config_path.exists():
        raise FileNotFoundError(f"training_config.json nicht gefunden unter: {training_config_path}")
    with open(training_config_path, "r", encoding="utf-8") as f:
        training_config = json.load(f)
    logging.info("✅ Trainings-Konfiguration geladen.")

    # Feature-Liste laden
    features_path = models_path / "features.joblib"
    if not features_path.exists():
        raise FileNotFoundError(f"features.joblib nicht gefunden unter: {features_path}")
    feature_list = joblib.load(features_path)

    # Modell laden (mit Logik für verschiedene Dateitypen)
    model_filename = config.get("model_filename")
    if not model_filename:
        raise ValueError("Kein 'model_filename' in der Konfiguration für die Inferenz gefunden.")
    
    model_path = models_path / model_filename
    if not model_path.exists():
        raise FileNotFoundError(f"Modell-Datei '{model_filename}' nicht gefunden unter: {model_path}")

    model = None
    if model_filename.endswith(('.keras', '.h5')):
        model = tf.keras.models.load_model(model_path)
    elif model_filename.endswith('.tflite'):
        model = tf.lite.Interpreter(model_path=str(model_path))
        model.allocate_tensors()
    elif model_filename.endswith(('.joblib', '.pkl', '.json')):
        model = joblib.load(model_path)
    else:
        raise ValueError(f"Unbekanntes Modellformat für Datei: {model_filename}")
    
    # Scaler laden (mit Prüfung der Konfiguration)
    scaler = None
    y_scaler = None
    if training_config.get('scale_other_features', True):
        scaler_path = scalers_path / "scaler.joblib"
        if not scaler_path.exists():
            raise FileNotFoundError(f"scaler.joblib wurde erwartet (scale_other_features=true), aber nicht gefunden unter: {scaler_path}")
        scaler = joblib.load(scaler_path)

    if training_config.get('scale_target', False):
        y_scaler_path = scalers_path / "y_scaler.joblib"
        if y_scaler_path.exists():
            y_scaler = joblib.load(y_scaler_path)

    return scaler, feature_list, model, training_config, y_scaler


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



# In Pipeline_Utils.py -> Klasse ModelScalerSaver

    def _export_tflite_variants(self, model, representative_dataset=None) -> dict:
        """
        Exportiert TFLite-Varianten basierend auf den 'quant_modes' in der Konfiguration.
        'no-quant' blockiert NICHT länger andere Modi.
        """
        import os, logging
        import tensorflow as tf

        results = {}

        q_modes_raw = self.config.get('quant_modes', [])
        q_modes = set(str(m).lower() for m in (q_modes_raw or []))

        if not q_modes:
            return results

        # Effektive Modi: alles außer 'no-quant'
        effective = q_modes - {'no-quant'}
        # Nur wenn wirklich NUR 'no-quant' da ist -> keine Quantisierung
        if not effective:
            return results

        models_dir = self.paths.get("Models")
        os.makedirs(models_dir, exist_ok=True)

        is_recurrent_model = any(isinstance(l, (tf.keras.layers.LSTM, tf.keras.layers.GRU)) for l in model.layers)

        # --- FLOAT16 ---
        if 'quant-16' in effective:
            try:
                conv = tf.lite.TFLiteConverter.from_keras_model(model)
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
                conv.target_spec.supported_types = [tf.float16]
                if is_recurrent_model:
                    logging.info("LSTM/GRU erkannt – SELECT_TF_OPS für Float16.")
                    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                    conv._experimental_lower_tensor_list_ops = False
                tflite_model = conv.convert()
                path = os.path.join(models_dir, "model_quant_float16.tflite")
                with open(path, "wb") as f:
                    f.write(tflite_model)
                results["tflite_float16"] = path
                logging.info(f"✅ TFLite FLOAT16 gespeichert: {path}")
            except Exception as e:
                logging.error(f"❌ Float16-Konvertierung fehlgeschlagen: {e}", exc_info=True)

        # --- INT8 (dynamic + optional full) ---
        if 'quant-8' in effective:
            # dynamic INT8
            try:
                conv = tf.lite.TFLiteConverter.from_keras_model(model)
                conv.optimizations = [tf.lite.Optimize.DEFAULT]
                if is_recurrent_model:
                    logging.info("LSTM/GRU erkannt – SELECT_TF_OPS für INT8 (dynamic).")
                    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                    conv._experimental_lower_tensor_list_ops = False
                tflite_model_dynamic = conv.convert()
                path_dyn = os.path.join(models_dir, "model_quant_int8.tflite")
                with open(path_dyn, "wb") as f:
                    f.write(tflite_model_dynamic)
                results["tflite_int8_dynamic"] = path_dyn
                logging.info(f"✅ TFLite INT8 (dynamic) gespeichert: {path_dyn}")
            except Exception as e:
                logging.error(f"❌ INT8 (dynamic) fehlgeschlagen: {e}", exc_info=True)

            # full-INT8 nur mit representative dataset
            if representative_dataset is not None:
                try:
                    conv = tf.lite.TFLiteConverter.from_keras_model(model)
                    conv.optimizations = [tf.lite.Optimize.DEFAULT]
                    conv.representative_dataset = representative_dataset
                    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    if is_recurrent_model:
                        logging.info("LSTM/GRU erkannt – zusätzlich SELECT_TF_OPS für INT8 (full).")
                        conv.target_spec.supported_ops.append(tf.lite.OpsSet.SELECT_TF_OPS)
                        conv._experimental_lower_tensor_list_ops = False
                    conv.inference_input_type = tf.int8
                    conv.inference_output_type = tf.int8
                    tflite_model_full = conv.convert()
                    path_full = os.path.join(models_dir, "model_quant_int8_full.tflite")
                    with open(path_full, "wb") as f:
                        f.write(tflite_model_full)
                    results["tflite_int8_full"] = path_full
                    logging.info(f"✅ TFLite INT8 (full) gespeichert: {path_full}")
                except Exception as e:
                    logging.error(f"❌ INT8 (full) fehlgeschlagen: {e}", exc_info=True)
            else:
                logging.info("Kein representative_dataset → INT8 (full) wird übersprungen.")

        return results



    def _save_config_as_json(self, output_path: str):
        """Speichert die Konfiguration als JSON-Datei."""
        try:
            # Stelle sicher, dass das Verzeichnis existiert
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                # default=str wird verwendet, um nicht-serialisierbare Typen (wie Pfad-Objekte) zu konvertieren
                json.dump(self.config, f, indent=4, default=str) 
            logging.info(f"✅ Trainings-Konfiguration gespeichert unter: {output_path}")
            return output_path
        except Exception as e:
            logging.error(f"❌ Fehler beim Speichern der Konfigurationsdatei: {e}", exc_info=True)
            return None

    def _save_structure_plot(self, model, output_path: str) -> str:
        """
        Erstellt und speichert die Modell-Architektur als Bild.
        """
        try:
            # Stellt sicher, dass das Zielverzeichnis existiert
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # tf.keras.utils.plot_model(
            #     model,
            #     to_file=output_path,
            #     show_shapes=True,
            #     show_layer_activations=True
            # )
            #print(f"📊 Modellstruktur gespeichert unter: {output_path}")
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
        Hauptmethode zum Speichern. Speichert zuerst das Modell, ermittelt dessen Größe,
        fügt sie zur Konfiguration hinzu und speichert dann die Konfigurations-JSON.
        """
        print(f"--- 🚀 Starte Speichern der Deployment-Artefakte für Modell: {self.config.get('model_name')} ---")
        results = {}
        
        # Schritt 1: Scaler speichern
        results.update(self._save_scaler(scaler))

        # Schritt 2: Modell-spezifische Artefakte speichern
        model_artifacts = {}
        if isinstance(model, tf.keras.Model):
            model_artifacts = self._save_keras_artifacts(model, **kwargs)
        elif isinstance(model, xgb.XGBRegressor):
            model_artifacts = self._save_xgboost_model(model)
        elif isinstance(model, (RandomForestRegressor, MultiOutputRegressor)):
            model_artifacts = self._save_sklearn_model(model)
        elif type(model).__name__ in ("LGBMRegressor", "LGBMClassifier"):
            model_artifacts = self._save_lightgbm_model(model)
        else:
            print(f"⚠️ Warnung: Kein spezifischer Speicherpfad für Modelltyp {type(model).__name__} implementiert.")
        
        results.update(model_artifacts)

        # Schritt 3: Modellgröße ermitteln und zur Konfiguration hinzufügen
        model_path = results.get("model_path")
        if model_path and os.path.exists(model_path):
            try:
                model_size_bytes = os.path.getsize(model_path)
                model_size_mb = round(model_size_bytes / (1024 * 1024), 4)
                self.config['model_size_MB'] = model_size_mb
                logging.info(f"✅ Modellgröße ermittelt: {model_size_mb} MB")
            except Exception as e:
                logging.error(f"❌ Fehler beim Ermitteln der Modellgröße: {e}")
        else:
            logging.warning("⚠️ Modellpfad nach dem Speichern nicht gefunden. Größe wird nicht in die Konfiguration geschrieben.")

        # Schritt 4: Konfiguration als JSON speichern (jetzt mit der Modellgröße)
        config_path = os.path.join(self.paths.get("Models"), "training_config.json")
        saved_config_path = self._save_config_as_json(config_path)
        if saved_config_path:
            results["training_config_path"] = saved_config_path
        
        # Schritt 5: Weitere Artefakte speichern
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
        """Speichert alle Artefakte für ein Keras-Modell (inkl. TFLite-Varianten)."""
        results = {}
        base_filename = f"{self.config['model_name']}_{self.config['dataset'].split('.')[0]}_{self.config['run_id']}"

        # 1) Keras .keras speichern
        try:
            path = os.path.join(self.paths.get("Models"), "model.keras")
            model.save(path)
            results["model_path"] = path
            print(f"✅ Normales Keras-Modell gespeichert unter: {path}")
        except Exception as e:
            print(f"❌ Fehler beim Speichern des Keras-Modells: {e}")
            traceback.print_exc()

        # --- ANGEPASSTE LOGIK HIER ---
        # 2) Optional: TFLite-Varianten, wenn Edge-Flag UND Quantisierung aktiv sind
        edge_flag = bool(self.config.get("edge_device", False) or self.config.get("enable_edge", False))
        quant_enabled = self.config.get('quantization_enabled', True)

        if edge_flag and quant_enabled:
            logging.info("TFLite conversion is enabled for edge device run.")
            rep_ds = kwargs.get("representative_dataset") or kwargs.get("train_dataset") or None
            tflite_paths = self._export_tflite_variants(model, representative_dataset=rep_ds)
            results.update(tflite_paths)
        elif edge_flag and not quant_enabled:
            logging.info("TFLite conversion is disabled via config, skipping .tflite creation for this edge run.")
        # --- ENDE DER ANPASSUNG ---

        # 3) Optional: Loss-Plot & Struktur
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
        # --- HIER EBENFALLS PRÜFEN, FALLS DIESE METHODE GENUTZT WIRD ---
        if not self.config.get('quantization_enabled', True):
            logging.warning("TFLite conversion is disabled via config. Call to _convert_and_quantize_tflite skipped.")
            return None
        # --- ENDE DER PRÜFUNG ---
            
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

    def _save_lightgbm_model(self, model) -> dict:
        try:
            import joblib
            model_dir = self.paths.get("Models")
            model_path = os.path.join(model_dir, "model.joblib")
            joblib.dump(model, model_path, compress=3)
            print(f"📤 LightGBM-Modell gespeichert unter: {model_path}")
            return {"model_path": model_path}
        except Exception as e:
            print(f"❌ Fehler beim Speichern des LightGBM-Modells: {e}", exc_info=True)
            return {}

    
    def _save_loss_plot(self, history, output_path: str):
        """
        Erstellt und speichert einen Plot des Trainings- & Validierungsverlusts
        sowie der Metriken aus dem Keras-History-Objekt.
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