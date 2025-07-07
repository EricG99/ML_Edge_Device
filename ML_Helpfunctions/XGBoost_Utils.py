import os
import time
import json
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import traceback


import ML_Helpfunctions.Pipeline_Utils as PipelineUtils

def train_xgboost_model(config: dict, X_train: np.ndarray,
                        y_train: np.ndarray, features: list):
    """
    Trainiert ein XGBoost-Modell. Verwendet einen chronologischen Split
    für das Validierungsset, wenn Early Stopping aktiviert ist.
    """
    print("--- Starte Training für XGBoost-Modell ---")
    start_time = time.time()

    # 1. Modellparameter (unverändert)
    model_params = {
        'n_estimators': config.get('n_estimators', 1000),
        'max_depth': config.get('max_depth', 5),
        # ... weitere Parameter
        'random_state': config.get('random_state', 42),
        'n_jobs': config.get('n_jobs', -1)
    }
    model_to_train = xgb.XGBRegressor(**model_params)

    # 2. Multi-Output-Strategie (unverändert)
    current_horizon = config.get("horizon", 1)
    y_train_for_fit = y_train
    if current_horizon == 1 and y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train_for_fit = y_train.ravel()

    # 3. Modell trainieren mit chronologischem Split für Early Stopping
    fit_params = {}
    
    # NEU: Verwende unsere Time-Series-Split-Funktion
    if config.get('early_stopping_rounds'):
        # Führe den chronologischen Split durch
        X_fit, y_fit, X_val, y_val = PipelineUtils.create_timeseries_validation_split(
            X_train, y_train_for_fit, config
        )

        # Stelle die Parameter für .fit() zusammen, nur wenn der Split erfolgreich war
        if X_val is not None and y_val is not None:
            fit_params['early_stopping_rounds'] = config.get('early_stopping_rounds')
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False
            print(f"Verwende Early Stopping mit chronologischem Validierungsset.")
    else:
        # Wenn kein Early Stopping, nutze die vollen Daten
        X_fit, y_fit = X_train, y_train_for_fit

    print(f"Starte XGBoost model.fit() auf Daten mit Shape X: {X_fit.shape}, Y: {y_fit.shape}...")
    model_to_train.fit(X_fit, y_fit, **fit_params)
    print("XGBoost-Modell Training abgeschlossen.")

    training_duration_seconds = time.time() - start_time
    print(f"Trainingszeit für XGBoost: {training_duration_seconds:.2f} Sekunden.")

    return model_to_train, training_duration_seconds

def run_inference_xgboost(model, X_test):
    """Führt die Inferenz mit dem trainierten XGBoost-Modell durch."""
    print("--- Starte Inferenz auf Testdaten ---")
    predictions = model.predict(X_test)
    return predictions


# -----------------------------------------------------------------------------
# NEUER HELPER: XGBOOST-MODELL SPEICHERN
# -----------------------------------------------------------------------------
def save_xgboost_model(model: xgb.XGBRegressor, config: dict, paths: dict) -> str:
    """
    Speichert ein XGBoost-Modell mit seiner nativen .save_model()-Methode.

    Args:
        model: Das trainierte XGBoost-Modell.
        config: Das Konfigurationsdictionary des Laufs.
        paths: Das Pfad-Dictionary des Laufs.

    Returns:
        str: Der Pfad zur gespeicherten Modelldatei oder None bei einem Fehler.
    """
    model_path = None
    try:
        model_dir = paths.get("Models", os.path.join(paths.get("output"), "Models"))
        os.makedirs(model_dir, exist_ok=True)

        model_name = config.get("model_name", "xgb_model").replace(" ", "_")
        dataset_name = config.get("dataset", "data").replace(".csv", "").replace(" ", "_")
        
        # XGBoost-Modelle werden oft als .json oder .ubj gespeichert
        model_filename = f"{model_name}_{dataset_name}_{config['run_id']}_{config['time_stamp']}.json"
        model_path = os.path.join(model_dir, model_filename)

        # Native, robustere Speichermethode von XGBoost verwenden
        model.save_model(model_path) 
        
        print(f"📤 XGBoost-Modell (als .json) gespeichert unter: {model_path}")
    except Exception as e:
        print(f"❌ Fehler beim Speichern des XGBoost-Modells: {e}")
        print(traceback.format_exc())
    return model_path

def save_results_xgboost(
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
    Orchestriert das Speichern aller Artefakte für ein trainiertes XGBoost-Modell
    durch Aufruf der generischen Hilfsfunktionen in Pipeline_Utils.
    """
    print(f"--- Speichere detaillierte {config.get('model_name')} Ergebnisse via PipelineUtils ---")

    # Guard-Clause zur Typprüfung
    if not isinstance(model, xgb.XGBRegressor):
        raise TypeError(f"Das übergebene Modell ist kein XGBoost Regressor, sondern {type(model)}")

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

    # 2. Speichere das XGBoost-Modell mit der neuen, spezialisierten Funktion
    model_path = save_xgboost_model(model, config, paths)

    # 3. Speichere die Edge-Artefakte (falls konfiguriert)
    edge_artifacts_path = PipelineUtils.save_edge_artifacts(config, paths)
    
    # 4. Sammle alle Ergebnisse und gib sie zurück
    final_results = {**common_results}
    if model_path:
        final_results["model_path"] = model_path
    if edge_artifacts_path:
        final_results["edge_artifacts"] = edge_artifacts_path
        
    return final_results