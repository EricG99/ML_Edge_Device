# --- Imports ---
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Sklearn
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# Modulimporte aus dem Projekt
import Feature_Engeneering as fe
import Load_Prepare_Data as LoadPrepareData
import Pipeline_Utils as PipelineUtils
import LSTM_Utils as LSTMUtils

from config import CONFIG_PATH
from config import param_LSTM

CONFIG_LSTM_ALL = {**CONFIG_PATH, **param_LSTM}


def setup_and_train_lstm_model(param_LSTM):
    """Bereitet die Daten vor und trainiert das LSTM-Modell."""

    # 1. Setup
    param_LSTM, paths = PipelineUtils.setup_experiment(param_LSTM)

    # 2. Daten vorbereiten mit erweiterten 2D-Features
    (
        X_train_3D, y_train_3D,
        X_test_3D, y_test_3D,
        scaler_3D, y_scaler,
        train_df, test_df,
        train_features_dict, full_feature_list
    ) = LoadPrepareData._prepare_base_data_3D(param_LSTM)

    print(f"[DEBUG] Shape y_train_3D: {y_train_3D.shape}, Shape y_test_3D: {y_test_3D.shape}")
    print(f"[DEBUG] Horizon aus config: {param_LSTM.get('horizon')}")

    # 3. Modell trainieren
    model, history, duration = LSTMUtils.train_model_LSTM(
        config=param_LSTM,
        X_train=X_train_3D,
        y_train=y_train_3D,
        features=full_feature_list
    )

    return model, duration, param_LSTM, paths, X_train_3D, y_train_3D, X_test_3D, y_test_3D, scaler_3D, test_df, full_feature_list, history


def run_inference_and_save_results_lstm(model, train_time, param_LSTM, paths,
                                        X_test_3D, y_test_3D, y_train_3D,
                                        full_feature_list, scaler_3D, test_df, history):
    """Führt Inferenz durch, evaluiert das Modell und speichert Ergebnisse."""

    # 4. Modell-Inferenz
    preds_test = LSTMUtils.run_inference_lstm(
        model=model,
        X_test=X_test_3D
    )

    # 5. Modell evaluieren
    pred_orig, true_orig, dates, metrics = PipelineUtils._evaluate_model(
        config=param_LSTM,
        predictions=preds_test,
        y_test=y_test_3D,
        scaler=scaler_3D,
        test_df=test_df,
        y_train=y_train_3D,
        features=full_feature_list
    )

    # 6. Ergebnisse speichern
    results = LSTMUtils.save_results_LSTM(
        config=param_LSTM,
        model=model,
        history=history,
        pred_orig=pred_orig,
        true_orig=true_orig,
        dates=dates,
        metrics_values=metrics,
        paths=paths,
        power_time=train_time,
        original_features_list=full_feature_list,
        scaler=scaler_3D
    )

    return metrics, results

from tensorflow.keras.models import load_model

def load_model_and_run_inference(model_path: str,
                                  X_test_3D: np.ndarray,
                                  scaler_3D,
                                  y_test_3D: np.ndarray,
                                  test_df: pd.DataFrame,
                                  param_LSTM: dict,
                                  y_train_3D: np.ndarray,
                                  full_feature_list: list) -> dict:
    """
    Lädt ein gespeichertes LSTM-Modell und führt Inferenz & Evaluation aus.

    Args:
        model_path (str): Pfad zum gespeicherten Modell.
        X_test_3D (np.ndarray): Testdaten für Inferenz.
        scaler_3D: Skalierer für Rücktransformation.
        y_test_3D (np.ndarray): Wahre Testwerte.
        test_df (pd.DataFrame): Test DataFrame mit Zeitstempeln.
        param_LSTM (dict): Konfiguration.
        y_train_3D (np.ndarray): Trainingszielwerte (für Eval).
        full_feature_list (list): Liste der verwendeten Features.

    Returns:
        dict: Ergebnisse inkl. Vorhersage, Metriken etc.
    """

    print(f"📦 Lade Modell von: {model_path}")
    try:
        model = load_model(model_path)
        print("✅ Modell erfolgreich geladen.")
    except Exception as e:
        print(f"❌ Fehler beim Laden des Modells: {e}")
        raise

    # Inferenz
    preds_test = LSTMUtils.run_inference_lstm(model=model, X_test=X_test_3D)

    # Evaluation
    pred_orig, true_orig, dates, metrics = PipelineUtils._evaluate_model(
        config=param_LSTM,
        predictions=preds_test,
        y_test=y_test_3D,
        scaler=scaler_3D,
        test_df=test_df,
        y_train=y_train_3D,
        features=full_feature_list
    )

    # Zusammenfassen der Ergebnisse
    results = {
        "pred_orig": pred_orig,
        "true_orig": true_orig,
        "dates": dates,
        "metrics": metrics
    }

    return results



def run_full_pipeline_LSTM(param_LSTM):
    """Führt den vollständigen LSTM-Pipeline-Prozess aus."""

    model, train_time, param_LSTM, paths, \
        X_train_3D, y_train_3D, X_test_3D, y_test_3D, \
        scaler_3D, test_df, full_feature_list, history = setup_and_train_lstm_model(param_LSTM)

    metrics, results = run_inference_and_save_results_lstm(
        model=model,
        train_time=train_time,
        param_LSTM=param_LSTM,
        paths=paths,
        X_test_3D=X_test_3D,
        y_test_3D=y_test_3D,
        y_train_3D=y_train_3D,
        full_feature_list=full_feature_list,
        scaler_3D=scaler_3D,
        test_df=test_df,
        history=history
    )

    return model, metrics, results


# --- Ausführung der Pipeline ---
if __name__ == "__main__":
    model, metrics, results = run_full_pipeline_LSTM(CONFIG_LSTM_ALL)
