# --- Imports ---

import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Eigene Module
import Feature_Engeneering as fe
import Load_Prepare_Data as LoadPrepareData
import Pipeline_Utils as PipelineUtils
from config import CONFIG_PATH, param_linreg_test

# Kombinierte Konfiguration
CONFIG_LINREG_ALL = {**CONFIG_PATH, **param_linreg_test}


def setup_and_train_linear_model(param_lin):
    """Daten vorbereiten und lineares Modell trainieren."""
    # 1. Setup
    param_lin, paths = PipelineUtils.setup_experiment(param_lin)

    # 2. Daten vorbereiten (2D-Format)
    X_train_2D, y_train_2D, X_test_2D, y_test_2D, scaler_2D, y_scaler, train_df, test_df, train_features_dict, full_feature_list = LoadPrepareData._prepare_base_data_2D(param_lin)

    print(f"[DEBUG] Shape X_train: {X_train_2D.shape}, y_train: {y_train_2D.shape}")

    # 3. Modellwahl
    model_type = param_lin.get("model_type", "linear").lower()
    if model_type == "ridge":
        model = Ridge(alpha=param_lin.get("alpha", 1.0))
    elif model_type == "lasso":
        model = Lasso(alpha=param_lin.get("alpha", 1.0))
    else:
        model = LinearRegression()

    # 4. Training
    model.fit(X_train_2D, y_train_2D)

    return model, param_lin, paths, X_train_2D, y_train_2D, X_test_2D, y_test_2D, scaler_2D, test_df, full_feature_list


def run_inference_and_save_results_lin(model, param_lin, paths,
                                       X_test_2D, y_test_2D, y_train_2D,
                                       full_feature_list, scaler_2D, test_df):
    """Inferenz, Evaluation und Ergebnis-Speicherung."""
    # 1. Inferenz
    preds_test = model.predict(X_test_2D)
        # 4. R²-Score berechnen und ausgeben
    r2 = r2_score(y_test_2D, preds_test)
    print(f"[INFO] R²-Score auf Testdaten: {r2:.4f}")

    # 2. Evaluation
    pred_orig, true_orig, dates, metrics = PipelineUtils._evaluate_model(
        config=param_lin,
        predictions=preds_test,
        y_test=y_test_2D,
        scaler=scaler_2D,
        test_df=test_df,
        y_train=y_train_2D,
        features=full_feature_list
    )

    # 3. Speichern
    results = PipelineUtils._save_common_results(
        config=param_lin,
        pred_orig=pred_orig,
        true_orig=true_orig,
        dates=dates,
        metrics_values=metrics,
        paths=paths,
        power_time=None,
        scaler=scaler_2D
    )

    return metrics, results


def run_full_pipeline_lin(param_lin):
    """Führt die vollständige Pipeline mit einem linearen Modell aus."""
    model, param_lin, paths, X_train_2D, y_train_2D, X_test_2D, y_test_2D, scaler_2D, test_df, full_feature_list = setup_and_train_linear_model(param_lin)
    
    metrics, results = run_inference_and_save_results_lin(
        model=model,
        param_lin=param_lin,
        paths=paths,
        X_test_2D=X_test_2D,
        y_test_2D=y_test_2D,
        y_train_2D=y_train_2D,
        full_feature_list=full_feature_list,
        scaler_2D=scaler_2D,
        test_df=test_df
    )

    return model, metrics, results


def run_full_pipeline_lin_without_save(param_lin):
    """Führt die vollständige Pipeline mit einem linearen Modell ohne Speicherung aus."""
    model, param_lin, paths, X_train_2D, y_train_2D, X_test_2D, y_test_2D, scaler_2D, test_df, full_feature_list = setup_and_train_linear_model(param_lin)
        # 1. Inferenz

    preds_test = model.predict(X_test_2D)
    
    # 4. R²-Score berechnen und ausgeben
    r2 = r2_score(y_test_2D, preds_test)
    print(f"[INFO] R²-Score auf Testdaten: {r2:.4f}")

    return model, r2

# Pipeline ausführen
#model, metrics, results = run_full_pipeline_lin(CONFIG_LINREG_ALL)

#run_full_pipeline_lin_without_save(CONFIG_LINREG_ALL)

if __name__ == "__main__":
    #model, r2 = run_full_pipeline_lin_without_save(CONFIG_LINREG_ALL)
    model , mtrics, results = run_full_pipeline_lin(CONFIG_LINREG_ALL)
    #print(f"Final R²-Score: {r2:.4f}")