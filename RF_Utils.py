import os
import joblib
import json
import time
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import Load_Prepare_Data as LoadPrepareData
import Pipeline_Utils as PipelineUtils
import traceback

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
                              y_train: np.ndarray, features: list):
    """
    Trainiert ein Random Forest-Modell, ggf. mit MultiOutputRegressor.
    Der Train-Validation-Split für die `fit`-Methode ist hier nicht enthalten.

    Args:
        config (dict): Konfigurationsparameter (n_estimators, max_depth, etc.).
        X_train (np.ndarray): Trainingsdaten (Input, 2D: [samples, flat_features]).
        y_train (np.ndarray): Trainingsdaten (Zielwerte). Die Form wird intern für
                              single-output (1D) oder multi-output (2D) angepasst.
        features (list): Liste der (geflachten) Feature-Namen (nicht direkt für das
                         Training verwendet, aber Teil der üblichen Signatur).

    Returns:
        tuple: (model, history, train_time)
            - model: Das trainierte Random Forest-Modell.
            - history: Ein Dummy-Verlaufsobjekt.
            - train_time: Die Trainingszeit in Sekunden.
    """
    print("Starte Training für Random Forest-Modell...")
    start_time = time.time()

    # 1. Modell initialisieren
    rf_base = RandomForestRegressor(
        n_estimators=config.get("n_estimators", 100),
        max_depth=config.get("max_depth", None),
        min_samples_split=config.get("min_samples_split", 2),
        min_samples_leaf=config.get("min_samples_leaf", 1),
        max_features=config.get("max_features", 1.0), 
        random_state=config.get("random_state", None),
        n_jobs=config.get("n_jobs", -1)
    )

    # 2. Multi-Output-Strategie und y_train-Anpassung
    current_horizon = config.get("horizon", 1)
    model_to_train = rf_base
    y_train_for_fit = y_train # Wird ggf. angepasst

    if current_horizon > 1:
        # y_train sollte für MultiOutputRegressor die Form (n_samples, n_outputs/horizon) haben
        model_to_train = MultiOutputRegressor(rf_base)
        # y_train_for_fit bleibt y_train (erwartet 2D)
        print(f"Random Forest: MultiOutputRegressor wird für horizon={current_horizon} verwendet.")
    else: # horizon == 1
        # RandomForestRegressor erwartet y als 1D-Array (n_samples,).
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_train_for_fit = y_train.ravel()
            print("Random Forest: y_train wurde für single-output von 2D zu 1D (ravel) angepasst.")
        # Falls y_train bereits 1D ist, ist keine Anpassung nötig.
        # model_to_train bleibt rf_base

    # 3. Modell trainieren
    # Scikit-learn Modelle werden auf den gesamten übergebenen Trainingsdaten trainiert.
    # Ein separater Validierungssplit für die fit-Methode ist hier nicht üblich.
    print(f"Starte Scikit-learn model.fit() für RandomForest auf Daten mit Shape X: {X_train.shape}, Y: {y_train_for_fit.shape}...")
    model_to_train.fit(X_train, y_train_for_fit)
    print("Random Forest-Modell Training abgeschlossen.")


    training_duration_seconds = time.time() - start_time
    print(f"Trainingszeit für Random Forest: {training_duration_seconds:.2f} Sekunden.")
    print(f"Model type after training: {type(model_to_train)}")


    return model_to_train, training_duration_seconds

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


def save_results_random_forest(
    config,
    model,
    pred_orig,
    true_orig,
    dates,
    metrics,
    paths,
    power_time
):
    # === Skalierer extrahieren ===
    scaler = model if not isinstance(model, dict) else model.get("scaler")

    # === Gemeinsame Ergebnisse speichern ===
    results = PipelineUtils._save_common_results(
        config=config,
        pred_orig=pred_orig,
        true_orig=true_orig,
        dates=dates,
        metrics_values=metrics,
        paths=paths,
        power_time=power_time,
        scaler=scaler
    )

    # === Modell speichern ===
    try:
        model_dir = paths.get("Models", os.path.join(paths.get("Base_Output_Path", "."), "Models"))
        os.makedirs(model_dir, exist_ok=True)

        model_name = config.get("model_name", "rf_model").replace(".csv", "").replace(" ", "_")
        dataset_name = config.get("dataset", "data").replace(".csv", "").replace(" ", "_")
        model_filename = f"{model_name}_{dataset_name}_{config['run_id']}_{config['time_stamp']}.joblib"
        model_path = os.path.join(model_dir, model_filename)

        joblib.dump(model, model_path, compress=3)
        results["model_path"] = model_path
        print(f"📤 Modell gespeichert unter: {model_path}")
    except Exception as e:
        print(f"❌ Fehler beim Speichern des Modells: {e}")
        print(traceback.format_exc())

    # === Edge-Artefakte speichern (optional) ===
    try:
        if config.get("enable_edge", False):
            edge_dir = os.path.join(model_dir, "edge_artifacts")
            os.makedirs(edge_dir, exist_ok=True)

            if "scaler_mean" in config:
                np.save(os.path.join(edge_dir, "scaler_mean.npy"), config["scaler_mean"])
                np.save(os.path.join(edge_dir, "scaler_scale.npy"), config["scaler_scale"])

            with open(os.path.join(edge_dir, "features.json"), "w") as f:
                json.dump(config["base_features"], f)

            results["edge_artifacts"] = edge_dir
            print(f"🧾 Edge-Artefakte gespeichert unter: {edge_dir}")
    except Exception as e:
        print(f"❌ Fehler beim Speichern der Edge-Artefakte: {e}")
        print(traceback.format_exc())

    return results



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