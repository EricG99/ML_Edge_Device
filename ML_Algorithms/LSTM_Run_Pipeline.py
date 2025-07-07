# --- Imports ---
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import tensorflow as tf


# Sklearn
from sklearn.preprocessing import MinMaxScaler, RobustScaler

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ML_Helpfunctions.Pipeline_Utils import ModelScalerSaver 

from ML_Helpfunctions import Feature_Engeneering as fe
from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils 
from ML_Helpfunctions import LSTM_Utils as LSTMUtils

#import tf # TensorFlow-Import für LSTM-Modelle Lite


from config import CONFIG_PATH, CONFIG_LSTM_ALL
from config import param_LSTM

# CONFIG_LSTM_ALL = {**CONFIG_PATH, **param_LSTM}


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



def setup_and_load_lstm_model(param_lstm_config):
    """Lädt ein vortrainiertes LSTM-Modell und bereitet die Daten vor."""

    # 1. Setup
    param_lstm_config, paths = PipelineUtils.setup_experiment(param_lstm_config)

    # 2. Daten vorbereiten mit erweiterten 3D-Features
    (
        X_train_3D, y_train_3D,
        X_test_3D, y_test_3D,
        scaler_3D, y_scaler,
        train_df, test_df,
        train_features_dict, full_feature_list
    ) = LoadPrepareData._prepare_base_data_3D(param_lstm_config)

    print(f"[DEBUG] Shape y_train_3D: {y_train_3D.shape}, Shape y_test_3D: {y_test_3D.shape}")
    print(f"[DEBUG] Horizon aus config: {param_lstm_config.get('horizon')}")

    # 3. Modell laden
    model = LSTMUtils.load_model_LSTM(
        model_path=param_lstm_config.get("input_data_edge_device"),
        model_name=param_lstm_config.get("model_name"),
    )

    return model, param_lstm_config, paths, X_train_3D, y_train_3D, X_test_3D, y_test_3D, scaler_3D, test_df, full_feature_list



def run_inference_and_save_results_lstm(
    model,
    train_time,
    param_LSTM,
    paths,
    X_train_3D,
    X_test_3D,
    y_test_3D,
    y_train_3D,
    full_feature_list,
    scaler_3D,
    test_df,
    history
):
    """
    Führt Inferenz durch, evaluiert die Vorhersagen und speichert alle Artefakte
    durch direkten Aufruf der zentralen Hilfsfunktionen und der ModelSaver-Klasse.
    """
    # 1. Modell-Inferenz (unverändert)
    print("--- 🧠 Starte Inferenz auf Testdaten ---")
    preds_test = model.predict(X_test_3D)

    # 2. Modell-Evaluation (unverändert)
    print("--- 📈 Evaluiere Modellvorhersagen ---")
    pred_orig, true_orig, dates, metrics = PipelineUtils._evaluate_model(
        predictions=preds_test,
        y_test=y_test_3D,
        scaler=scaler_3D,
        test_df=test_df,
        config=param_LSTM,
        features=full_feature_list,
        y_train=y_train_3D
    )

    # 3. Speichere die Evaluationsmetriken UND Vorhersagen in CSVs
    # Dieser Aufruf speichert die Metriken in der Summary-Datei und die Vorhersagen
    # in einer eigenen CSV. Wir benötigen den Rückgabewert hier nicht direkt.
    # Annahme: _save_common_results ist Ihre Funktion zum Speichern von Metriken/Vorhersagen.
    common_results = PipelineUtils._save_common_results(
        config=param_LSTM,
        pred_orig=pred_orig,
        true_orig=true_orig,
        dates=dates,
        metrics=metrics,
        paths=paths,
        power_time=train_time,
        scaler=scaler_3D # Wichtig, falls _save_common_results den Skalierer auch speichert
    )

    # 4. Speichere die Deployment-Artefakte (Modell, TFLite, Plots)
    # Initialisiere und rufe den zentralen Saver auf
    saver = ModelScalerSaver(config=param_LSTM, paths=paths)
    deployment_results = saver.save_artifacts(
        model=model,
        scaler=scaler_3D,
        history=history,
    )
    
    # 5. Kombiniere alle Ergebnis-Pfade für die finale Rückgabe
    results = {**common_results, **deployment_results}

    return metrics, results

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
        X_train_3D=X_train_3D, 
        X_test_3D=X_test_3D,
        y_test_3D=y_test_3D,
        y_train_3D=y_train_3D,
        full_feature_list=full_feature_list,
        scaler_3D=scaler_3D,
        test_df=test_df,
        history=history
    )

    return model, metrics, results


def run_full_pipeline_LSTM_without_save(param_LSTM):
    """Führt die vollständige Pipeline mit allen Konfigurationen aus."""
    model, train_time, param_LSTM, paths, \
        X_train_3D, y_train_3D, X_test_3D, y_test_3D, \
        scaler_3D, test_df, full_feature_list, history = setup_and_train_lstm_model(param_LSTM)
    
    # 4. Modell-Inferenz
    preds_test = LSTMUtils.run_inference_lstm(
        model=model,
        X_test=X_test_3D
    )
    return model, full_feature_list
    
if __name__ == "__main__":
    model, metrics, results = run_full_pipeline_LSTM(CONFIG_LSTM_ALL)
