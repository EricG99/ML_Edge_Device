
# --- Imports ---
import os
import sys
import datetime
import numpy as np
import pandas as pd

import tensorflow as tf



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ML_Helpfunctions.Pipeline_Utils import ModelScalerSaver 


from ML_Helpfunctions import CNN_1D_Utils as CNNUtils 
from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils 

# from config import CONFIG_PATH, param_CNN_1D
# CONFIG_CNN_1D_ALL = {**CONFIG_PATH, **param_CNN_1D}

from pathlib import Path
from datetime import datetime


# Hauptpfade
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"



CONFIG_CNN_1D_ALL = {
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
    "model_name": "1D_CNN_Model",
    "time_stamp": datetime.now().strftime("%Y-%m-%d_%H%M%S"),

    # ---- Datensatz ----
    "dataset": "filtered_wanda_dataset.csv",

    # --- Zeitreihenparameter (können gleich bleiben) ---
    "zielvariable": "Pressure_Hall",
    "lags": 4, # Wichtig für die Sequenzlänge
    "horizon": 2,
    "train_fraction": 0.3,
    
    # --- Feature-Konfiguration (kann gleich bleiben) ---
    "base_features": ['Volume_Flow', 'Pressure_Hall'],
    "time_features": [],
    "include_roll_mean": True,
    "include_roll_std": True,
    "rolling_window_size": 5,
    "scale_target": False,
    "edge_device": True, 

    # --- NEU: 1D-CNN-spezifische Hyperparameter ---
    "num_conv_blocks": 1,     # Anzahl der Conv/MaxPool-Blöcke
    "filters": 64,            # Anzahl der Filter in der ersten Conv-Schicht
    "kernel_size": 3,         # Größe des Faltungskerns
    "pool_size": 2,           # Größe für das MaxPooling
    "dense_units": 50,        # Anzahl der Neuronen in der Dense-Schicht
    "dropout": 0.1,

    # --- Trainingsparameter (können ähnlich sein) ---
    "epochs": 1,
    "batch_size": 32,
    "loss": tf.keras.losses.Huber(), # Empfohlene, robustere Methode
    "optimizer": "adam",
    "use_early_stopping": True,
    "early_stopping_patience": 10,
    "validation_fraction": 0.15,
}

def setup_and_train_cnn_model(param_cnn):
    """Bereitet die Daten vor und trainiert das 1D-CNN-Modell."""
    param_cnn, paths = PipelineUtils.setup_experiment(param_cnn)

    # 2. Daten vorbereiten (Wiederverwendung der 3D-Datenvorbereitung)
    (
        X_train_3D, y_train_3D, X_test_3D, y_test_3D,
        scaler_3D, y_scaler, train_df, test_df,
        train_features_dict, full_feature_list
    ) = LoadPrepareData._prepare_base_data_3D(param_cnn)

    print(f"[DEBUG] Shape y_train_3D: {y_train_3D.shape}, Shape y_test_3D: {y_test_3D.shape}")

    # 3. Modell trainieren
    model, history, duration = CNNUtils.train_model_1D_CNN(
        config=param_cnn,
        X_train=X_train_3D,
        y_train=y_train_3D,
        features=full_feature_list
    )

    return model, duration, param_cnn, paths, X_train_3D, y_train_3D, X_test_3D, y_test_3D, scaler_3D, test_df, full_feature_list, history


def run_inference_and_save_results_cnn(
    model, train_time, param_cnn, paths,
    X_train_3D, X_test_3D, y_test_3D, y_train_3D,
    full_feature_list, scaler_3D, test_df, history
):
    """
    Führt Inferenz mit einem trainierten 1D-CNN-Modell durch, evaluiert die Vorhersagen
    und speichert Metriken sowie Modell-Artefakte getrennt.

    Rückgabe:
        metrics (dict): Evaluationsmetriken
        results (dict): Kombinierte Ergebnisse von Metrik- und Modell-Speicherung
    """
    # 1. Inferenz
    preds_test = CNNUtils.run_inference_cnn(model=model, X_test=X_test_3D)

    # 2. Evaluation
    pred_orig, true_orig, dates, metrics = PipelineUtils._evaluate_model(
        config=param_cnn,
        predictions=preds_test,
        y_test=y_test_3D,
        scaler=scaler_3D,
        test_df=test_df,
        y_train=y_train_3D,
        features=full_feature_list
    )


    # 4. Speichern der Metriken und Vorhersagen
    metrics_results = CNNUtils.save_cnn_metrics_prediction(
        config=param_cnn,
        pred_orig=pred_orig,
        true_orig=true_orig,
        dates=dates,
        metrics=metrics,
        paths=paths,
        power_time=train_time,
    )

    # 5. Speichern des Modells und weiterer Artefakte
    # Initialisiere und rufe den zentralen Saver auf
    saver = ModelScalerSaver(config=param_cnn, paths=paths)
    deployment_results = saver.save_artifacts(
        model=model,
        scaler=scaler_3D,
        history=history,
    )

    # 6. Ergebnisse zusammenführen
    results = {**metrics_results, **deployment_results}

    return metrics, results


def run_full_pipeline_CNN(param_cnn):
    """Führt den vollständigen 1D-CNN-Pipeline-Prozess aus."""
    model, train_time, param_cnn, paths, \
        X_train_3D, y_train_3D, X_test_3D, y_test_3D, \
        scaler_3D, test_df, full_feature_list, history = setup_and_train_cnn_model(param_cnn)

    metrics, results = run_inference_and_save_results_cnn(
        model=model, train_time=train_time, param_cnn=param_cnn, paths=paths,
        X_train_3D=X_train_3D, X_test_3D=X_test_3D, y_test_3D=y_test_3D,
        y_train_3D=y_train_3D, full_feature_list=full_feature_list,
        scaler_3D=scaler_3D, test_df=test_df, history=history
    )
    return model, metrics, results


if __name__ == "__main__":
    model, metrics, results = run_full_pipeline_CNN(CONFIG_CNN_1D_ALL)