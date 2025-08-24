import os
import joblib
import json
import time
import numpy as np
import sys
import pandas as pd
from typing import List, Tuple
import traceback

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import ML_Helpfunctions.Load_Prepare_Data as LoadPrepareData
import ML_Helpfunctions.pipeline_utils as PipelineUtils


def train_random_forest_pipeline(config: dict, mode: str = "server", ssh_config: dict = None) -> dict:
    """
    Random Forest Pipeline mit drei Modi:
    1. 'server': Training & Inferenz auf Server
    2. 'edge': Training auf Server, Inferenz auf Edge
    3. 'edge_train': Training & Inferenz auf Edge (simuliert lokal)
    """
    # Daten vorbereiten
    X_train, y_train, X_test, y_test, feature_list, scaler, test_df, base_features = \
        LoadPrepareData._prepare_base_data_2D(config)

    if mode == "edge_train":
        model, _ = train_random_forest_edge(config, X_train, y_train, base_features)
    else:
        model, _ = train_random_forest_model(config, X_train, y_train, base_features)

    # Modell evaluieren
    y_pred, y_true, dates, metrics = evaluate_model_random_forest(
        config, model, X_test, y_test, scaler, test_df, y_train, base_features
    )

    # Ergebnisse speichern
    paths = config["paths"] if "paths" in config else {"Models": "./models"}
    results = save_results_random_forest(config, model, y_pred, y_true, dates, metrics, paths, None, base_features)

    # Optional: Edge-Deployment via SSH
    if mode == "edge" and ssh_config:
        _upload_to_revpi(results["model_path"], ssh_config)

    return {
        "model_path": results.get("model_path"),
        "scaler": os.path.join(paths["Models"], "scaler.pkl"),
        "metrics": metrics,
        "edge_artifacts": results.get("edge_artifacts", None)
    }

def train_random_forest_model(config: dict, X_train: np.ndarray,
                              y_train: np.ndarray, features: list | None = None):
    """
    Trainiert ein Random Forest-Modell.
    - Bei horizon>1 wird IMMER MultiOutputRegressor genutzt.
    - Gibt (model, train_time_seconds) zurück.
    """
    import time
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor

    print("Starte Training für Random Forest-Modell...")
    t0 = time.time()

    rf_base = RandomForestRegressor(
        n_estimators=config.get("n_estimators", 100),
        max_depth=config.get("max_depth", None),
        min_samples_split=config.get("min_samples_split", 2),
        min_samples_leaf=config.get("min_samples_leaf", 1),
        max_features=config.get("max_features", 1.0),
        random_state=config.get("random_state", None),
        n_jobs=config.get("n_jobs", -1),
    )

    H = int(config.get("horizon", 1))
    y_train_fit = y_train

    if H > 1:
        # y muss 2D sein: (n_samples, H)
        y_train_fit = np.asarray(y_train, dtype=float)
        if y_train_fit.ndim != 2 or y_train_fit.shape[1] != H:
            raise ValueError(f"RF-Training: Erwartet y in Form (n, {H}), bekam {y_train_fit.shape}.")
        model_to_train = MultiOutputRegressor(rf_base)
        print(f"Random Forest: MultiOutputRegressor wird für horizon={H} verwendet.")
    else:
        # Single-Output → 1D
        y_train_fit = np.asarray(y_train, dtype=float).ravel()
        model_to_train = rf_base

    print(f"Starte Scikit-learn model.fit() auf Daten mit Shape X: {X_train.shape}, Y: {y_train_fit.shape}...")
    model_to_train.fit(X_train, y_train_fit)
    print("Random Forest-Modell Training abgeschlossen.")

    dt = time.time() - t0
    print(f"Trainingszeit für Random Forest: {dt:.2f} Sekunden.")
    print(f"Model type after training: {type(model_to_train)}")

    return model_to_train, dt


def train_random_forest_edge(config, X_train, y_train, features):
    X_train, X_val, y_train, y_val = LoadPrepareData._create_train_val_split(X_train, y_train, 0.2)
    rf = RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_split=2,
        max_samples=0.5,
        random_state=config["random_state"],
        n_jobs=-1
    )

    return rf


def run_inference_random_forest(model, X_test: np.ndarray) -> np.ndarray:
    if len(X_test.shape) == 3:
        X_test = X_test.reshape(X_test.shape[0], -1)
    preds = model.predict(X_test)
    return np.clip(preds, 0, None)


def evaluate_model_random_forest(config: dict,
                                 predictions: np.ndarray,
                                 y_test: np.ndarray,
                                 scaler,
                                 test_df,
                                 y_train: np.ndarray,
                                 features: list):
    return PipelineUtils._evaluate_model(predictions, y_test, scaler, test_df, config, features, y_train)

def save_sklearn_model(model, config: dict, paths: dict) -> str:
    """Speichert ein Scikit-learn-Modell mit joblib."""
    model_path = None
    try:
        model_dir = paths.get("Models", os.path.join(paths.get("output"), "Models"))
        os.makedirs(model_dir, exist_ok=True)

        model_name = config.get("model_name", "sklearn_model").replace(" ", "_")
        dataset_name = config.get("dataset", "data").replace(".csv", "").replace(" ", "_")
        
        model_filename = f"{model_name}_{dataset_name}_{config['run_id']}_{config['time_stamp']}.joblib"
        model_path = os.path.join(model_dir, model_filename)

        joblib.dump(model, model_path, compress=3)
        print(f"📤 Scikit-learn-Modell gespeichert unter: {model_path}")
    except Exception as e:
        print(f"❌ Fehler beim Speichern des Scikit-learn-Modells: {e}")
        print(traceback.format_exc())
    return model_path



def save_results_random_forest(
    config,
    model,
    scaler,
    pred_orig,
    true_orig,
    dates,
    metrics,
    paths,
    power_time
):
    """
    Orchestriert das Speichern aller Artefakte für ein trainiertes RandomForest-Modell
    durch Aufruf der generischen Hilfsfunktionen in Pipeline_Utils.
    """
    print(f"--- Speichere detaillierte {config.get('model_name')} Ergebnisse via PipelineUtils ---")

    # 1. Speichere gemeinsame Ergebnisse (Skalierer, Metriken, Vorhersagen-CSV)
    common_results = PipelineUtils._save_common_results(
        config=config,
        pred_orig=pred_orig,
        true_orig=true_orig,
        dates=dates,
        metrics=metrics,
        paths=paths,
        power_time=power_time,
        scaler=scaler
    )

    # 2. Speichere das Scikit-learn-Modell
    model_path = save_sklearn_model(model, config, paths)

    # 3. Speichere die Edge-Artefakte
    edge_artifacts_path = PipelineUtils.save_edge_artifacts(config, paths)
    
    # 4. Sammle alle Ergebnisse und gib sie zurück
    final_results = {**common_results}
    if model_path:
        final_results["model_path"] = model_path
    if edge_artifacts_path:
        final_results["edge_artifacts"] = edge_artifacts_path
        
    return final_results



def _upload_to_revpi(file_path: str, ssh_config: dict):
    import paramiko
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=ssh_config["host"],
        username=ssh_config["user"],
        password=ssh_config["password"]
    )
    sftp = ssh.open_sftp()
    remote_path = ssh_config.get("remote_path", "/home/pi/model.joblib")
    sftp.put(file_path, remote_path)
    sftp.close()
    ssh.close()