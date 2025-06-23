# --- Imports ---

import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime

# Sklearn
from sklearn.preprocessing import MinMaxScaler, RobustScaler


# Modulimporte aus dem Projekt
import Feature_Engeneering as fe
import Load_Prepare_Data as LoadPrepareData
import Pipeline_Utils as PipelineUtils # Changed to PipelineUtils
import RF_Utils as RFUtils # Changed to RFUtils

from config import CONFIG_PATH
from config import param_rf_test

CONFIG_RF_ALL = {**CONFIG_PATH, **param_rf_test}


def setup_and_train_rf_model(param_rf):
    """Bereitet die Daten vor und trainiert das Random-Forest-Modell."""
    # 1. Setup
    param_rf, paths = PipelineUtils.setup_experiment(param_rf)

    # 2. Daten vorbereiten mit erweiterten 2D-Features
    X_train_2D, y_train_2D, X_test_2D, y_test_2D, scaler_2D, y_scaler, train_df, test_df, train_features_dict, full_feature_list = LoadPrepareData._prepare_base_data_2D(param_rf)

    print(f"[DEBUG] Shape y_train_2D: {y_train_2D.shape}, Shape y_test_2D: {y_test_2D.shape}")
    print(f"[DEBUG] Horizon aus config: {param_rf.get('horizon')}")
    # 3. Modell trainieren
    model, train_time = RFUtils.train_random_forest_model(
        config=param_rf,
        X_train=X_train_2D,
        y_train=y_train_2D,
        features=full_feature_list
    )

    return model, train_time, param_rf, paths, X_train_2D, y_train_2D, X_test_2D, y_test_2D, scaler_2D, test_df, full_feature_list


def run_inference_and_save_results_rf(model, train_time, param_rf, paths,
                                      X_test_2D, y_test_2D, y_train_2D,
                                      full_feature_list, scaler_2D, test_df):
    """Führt Inferenz durch, evaluiert das Modell und speichert Ergebnisse."""
    
    # 4. Modell-Inferenz 
    preds_test = RFUtils.run_inference_random_forest(
        model=model,
        X_test=X_test_2D
    )

    # 5. Modell evaluieren 
    pred_orig, true_orig, dates, metrics = RFUtils.evaluate_model_random_forest(
        config=param_rf,
        predictions=preds_test,  
        y_test=y_test_2D,
        scaler=scaler_2D,
        test_df=test_df,
        y_train=y_train_2D,   
        features=full_feature_list
    )

    # 6. Ergebnisse speichern
    results = RFUtils.save_results_random_forest(
        config=param_rf, 
        model=model, 
        pred_orig=pred_orig, 
        true_orig=true_orig, 
        dates=dates, 
        metrics=metrics, 
        paths=paths,
        power_time=train_time,
    )

    # 7. Bewertung auf Testdaten
    r2_score = model.score(X_test_2D, y_test_2D)
    print(f"Modellbewertung (R^2) auf Testdaten: {r2_score}")

    return metrics, results


def run_full_pipeline_rf(param_rf):
    """Führt den vollständigen Random-Forest-Pipeline-Prozess aus."""
    model, train_time, param_rf, paths, X_train_2D, y_train_2D, X_test_2D, y_test_2D, scaler_2D, test_df, full_feature_list = setup_and_train_rf_model(param_rf)

    metrics, results = run_inference_and_save_results_rf(
        model=model,
        train_time=train_time,
        param_rf=param_rf,
        paths=paths,
        X_test_2D=X_test_2D,
        y_test_2D=y_test_2D,
        y_train_2D=y_train_2D,
        full_feature_list=full_feature_list,
        scaler_2D=scaler_2D,
        test_df=test_df
    )

    return model, metrics, results

def run_full_pipeline_rf_without_save(param_rf):
    """Führt die vollständige Pipeline mit allen Konfigurationen aus."""
    # 4. Modell-Inferenz 
    model, train_time, param_rf, paths, X_train_2D, y_train_2D, X_test_2D, y_test_2D, scaler_2D, test_df, full_feature_list = setup_and_train_rf_model(param_rf)
    preds_test = RFUtils.run_inference_random_forest(model=model, X_test=X_test_2D)
    
    # R² berechnen mit den Vorhersagen (statt model.score)
    from sklearn.metrics import r2_score
    r2_test = r2_score(y_test_2D, preds_test)
    print(f"R² (Test): {r2_test:.4f}")

#model, metrics, results = run_full_pipeline_rf(CONFIG_RF_ALL)

run_full_pipeline_rf_without_save(CONFIG_RF_ALL)

if __name__ == "__main__":
    model, metrics, results = run_full_pipeline_rf(CONFIG_RF_ALL)
    print("Pipeline abgeschlossen.")
    print("Metriken:", metrics)

