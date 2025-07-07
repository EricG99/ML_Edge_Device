# --- Imports ---

import os
import sys
import datetime
import numpy as np
import pandas as pd

# Sklearn (wird intern von Hilfsfunktionen genutzt)
from sklearn.preprocessing import MinMaxScaler, RobustScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import der Hilfsfunktionen
from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils 
from ML_Helpfunctions import XGBoost_Utils as XGBUtils
from ML_Helpfunctions import RF_Utils as RFUtils

# # Import der Konfigurationen
# from config import CONFIG_PATH
# from config import param_xgb_test

# CONFIG_XGB_ALL = {**CONFIG_PATH, **param_xgb_test}
from pathlib import Path
from datetime import datetime


# Hauptpfade
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"

CONFIG_XGB_ALL = {
    # ---- Pfade bleiben erhalten (lokales Setup) ----
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

    # ---- Modellname und aktueller Timestamp ----
    "model_name": "XGBoost_Model",
    "time_stamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),

    # ---- Datensatz ----
    "dataset": "filtered_wanda_dataset.csv",

    # ---- Modellparameter (für RF) ----
    "n_estimators": 100,
    "max_depth": 25,
    "min_samples_split": 5,
    "min_samples_leaf": 3,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": 1,

    # ---- Modell-Dateinamen ----
    "model_params": {
        "saved_model_name": "random_forest_model.pkl",
        "scaler_name": "scaler.pkl"
    },

    # ---- Zeitreihenparameter ----
    "lags": 2,
    "horizon": 2,
    "train_fraction": 0.3,
    "rolling_window_size": 5,

    # ---- Feature-Konfiguration ----
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    # "time_features": [
    #     'second', "minute", "minute_sin", "minute_cos", "hour", "hour_sin", "hour_cos",
    #     "day_of_month", "day_of_week", "is_weekend", "month", "month_sin", "month_cos"
    # ],
    "time_features": [],

    # ---- Feature Engineering ----
    "include_roll_mean": True,
    "include_roll_std": True,
    "scale_other_features": False,
    "scale_target": False,
}


def setup_and_train_xgb_model(param_xgb):
    """Bereitet die Daten vor und trainiert das XGBoost-Modell."""
    # 1. Setup
    param_xgb, paths = PipelineUtils.setup_experiment(param_xgb)

    # 2. Daten vorbereiten (diese Funktion wird wiederverwendet)
    # Annahme: Diese Funktion liefert auch Validierungsdaten für Early Stopping
    X_train_2D, y_train_2D, X_test_2D, y_test_2D, scaler_2D, y_scaler, train_df, test_df, train_features_dict, full_feature_list = LoadPrepareData._prepare_base_data_2D(param_xgb)


    print(f"[DEBUG] Shape y_train_2D: {y_train_2D.shape}, Shape y_test_2D: {y_test_2D.shape}")
    print(f"[DEBUG] Horizon aus config: {param_xgb.get('horizon')}")

    # 3. Modell trainieren
    model, train_time = XGBUtils.train_xgboost_model(
        config=param_xgb,
        X_train=X_train_2D,
        y_train=y_train_2D,
        features=full_feature_list

    )

    return model, train_time, param_xgb, paths, X_train_2D, y_train_2D, X_test_2D, y_test_2D, scaler_2D, test_df, full_feature_list


def run_inference_and_save_results_xgb(model, train_time, param_xgb, paths,
                                       X_test_2D, y_test_2D, y_train_2D,
                                       full_feature_list, scaler_2D, test_df):
    """Führt Inferenz durch, evaluiert das Modell und speichert Ergebnisse."""
    
    # 4. Modell-Inferenz 
    preds_test = XGBUtils.run_inference_xgboost(
        model=model,
        X_test=X_test_2D
    )

    # 5. Modell evaluieren 
    pred_orig, true_orig, dates, metrics = RFUtils.evaluate_model_random_forest(
        config=param_xgb,
        predictions=preds_test,  
        y_test=y_test_2D,
        scaler=scaler_2D,
        test_df=test_df,
        y_train=y_train_2D,  
        features=full_feature_list
    )

    # 6. Ergebnisse speichern
    results = XGBUtils.save_results_xgboost(
        config=param_xgb, 
        model=model,
        scaler =scaler_2D,
        pred_orig=pred_orig, 
        true_orig=true_orig, 
        dates=dates, 
        metrics=metrics, 
        paths=paths,
        power_time=train_time,
    )

    # 7. Bewertung auf Testdaten (R-Quadrat)
    r2_score_val = model.score(X_test_2D, y_test_2D)
    print(f"Modellbewertung (R^2) auf skalierten Testdaten: {r2_score_val:.4f}")

    return metrics, results


def run_full_pipeline_xgb(param_xgb):
    """Führt den vollständigen XGBoost-Pipeline-Prozess aus."""
    model, train_time, param_xgb, paths, X_train_2D, y_train_2D, X_test_2D, y_test_2D, scaler_2D, test_df, full_feature_list = setup_and_train_xgb_model(param_xgb)

    metrics, results = run_inference_and_save_results_xgb(
        model=model,
        train_time=train_time,
        param_xgb=param_xgb,
        paths=paths,
        X_test_2D=X_test_2D,
        y_test_2D=y_test_2D,
        y_train_2D=y_train_2D,
        full_feature_list=full_feature_list,
        scaler_2D=scaler_2D,
        test_df=test_df
    )

    return model, metrics, results, full_feature_list

def run_full_pipeline_XGBoost_without_save(param_xgb):
    """Führt die vollständige Pipeline mit allen Konfigurationen aus."""
    # 4. Modell-Inferenz 
    model, train_time, param_xgb, paths, X_train_2D, y_train_2D, X_test_2D, y_test_2D, scaler_2D, test_df, full_feature_list = setup_and_train_xgb_model(param_xgb)
    preds_test = XGBUtils.run_inference_xgboost(model=model, X_test=X_test_2D)
    
    # R² berechnen mit den Vorhersagen (statt model.score)
    from sklearn.metrics import r2_score
    r2_test = r2_score(y_test_2D, preds_test)
    print(f"R² (Test): {r2_test:.4f}")
    return model, full_feature_list

# --- Main Ausführung ---
if __name__ == "__main__":
    print("Starte XGBoost-Pipeline...")
    model, metrics, results, full_feature_list = run_full_pipeline_xgb(CONFIG_XGB_ALL)
    print("\nXGBoost-Pipeline abgeschlossen.")
    print("\nFinale Metriken:", metrics)