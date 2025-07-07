import numpy as np
import tensorflow as tf
import os
import sys
import joblib
import pandas as pd
from datetime import datetime
from typing import List, Tuple



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ML_Helpfunctions.Pipeline_Utils import ModelScalerSaver 

from ML_Helpfunctions import Feature_Engeneering as fe
from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils 
from ML_Helpfunctions import LSTM_Utils as LSTMUtils
# Util-Imports
from ML_Helpfunctions import (
    Pipeline_Utils,
    Load_Prepare_Data as LoadPrepareData,
    LSTM_Utils,
    CNN_1D_Utils,
    RF_Utils,
    XGB_Utils
)

# Konfigurations-Import
from config import param_LSTM # Annahme: Sie haben diese Funktion

# =============================================================================
# == GEMEINSAME HILFSFUNKTION FÜR ALLE PIPELINES
# =============================================================================
def _run_inference_and_saving(model, config, paths, data_tuple, history=None, train_time=0.0):
    """
    Diese private Funktion führt die Inferenz, Evaluation und das Speichern durch.
    Sie wird von allen drei Haupt-Pipelines aufgerufen.
    """
    print("\n--- SCHRITT 3: INFERENZ, EVALUATION & SPEICHERN ---")
    
    # Daten aus dem Tupel entpacken
    X_train, y_train, X_test, y_test, scaler, _, _, test_df, _, features = data_tuple

    # 3a. Inferenz mit Zeitmessung
    preds, timings = PipelineUtils.run_inference_with_timing(model, X_test)
    
    # 3b. Evaluation
    pred_orig, true_orig, dates, metrics = PipelineUtils._evaluate_model(
        preds, y_test, scaler, test_df, config, features, y_train
    )
    metrics["avg_inference_time_ms"] = np.mean(timings)
    metrics["total_train_time_s"] = train_time

    # 3c. Speichern der Metriken und Vorhersagen
    prediction_path = PipelineUtils.save_predictions_to_csv(pred_orig, true_orig, dates, config, paths)
    metrics_results = PipelineUtils.save_evaluation_metrics(config, metrics, paths, train_time, prediction_path)

    # 3d. Speichern der Deployment-Artefakte
    scaler_to_save = scaler if config.get("save_scaler_on_retrain", True) else None

    representative_dataset_obj = None
    if config.get("edge_device", False) and config.get("model_name") in ['EDGE_LSTM', '1D_CNN_Model']:
        num_samples = min(100, X_train.shape[0])
        representative_dataset_obj = tf.data.Dataset.from_tensor_slices(X_train[:num_samples])
    
    saver = ModelScalerSaver(config=config, paths=paths)
    deployment_results = saver.save_artifacts(
        model=model, scaler=scaler_to_save, history=history, representative_dataset=representative_dataset_obj
    )
    
    return metrics, {**metrics_results, **deployment_results}

# =============================================================================
# == DIE DREI GEWÜNSCHTEN HAUPT-PIPELINES
# =============================================================================

def run_full_training_pipeline(config: dict):
    """Pipeline 1: Setup, Training von Grund auf, Inferenz, Speichern (alles)."""
    print(f"\n===== STARTE PIPELINE 1: VOLLSTÄNDIGES TRAINING FÜR {config.get('model_name')} =====")
    
    config, paths = PipelineUtils.setup_experiment(config)
    
    model_name = config.get("model_name")
    if model_name in ["EDGE_LSTM", "1D_CNN_Model"]:
        data_tuple = LoadPrepareData._prepare_base_data_3D(config)
    else:
        data_tuple = LoadPrepareData._prepare_base_data_2D(config)
    
    X_train, y_train, _, _, _, _, _, _, _, features = data_tuple
    
    history, train_time = None, 0.0
    if model_name == 'EDGE_LSTM':
        model, history, train_time = LSTM_Utils.train_model_LSTM(config, X_train, y_train, features)
    elif model_name == '1D_CNN_Model':
        model, history, train_time = CNN_1D_Utils.train_model_1D_CNN(config, X_train, y_train, features)
    # ... hier elif für RF, XGB etc. einfügen ...
    
    return _run_inference_and_saving(model, config, paths, data_tuple, history, train_time)


def run_inference_only_pipeline(config: dict):
    """Pipeline 2: Setup, Laden eines Modells, Inferenz, Speichern (nur Metriken)."""
    print(f"\n===== STARTE PIPELINE 2: INFERENZ FÜR {config.get('model_name')} =====")

    config, paths = PipelineUtils.setup_experiment(config)
    model_name = config.get("model_name")
    
    # Laden, aber keine Artefakte speichern (deshalb rufen wir die Haupt-Save-Funktion nicht auf)
    model = RF_Utils.load_rf_model(config.get("model_load_path")) # Beispiel für RF
    # ... hier if/elif für die anderen load_...-Funktionen ...
    
    # Daten nur für Inferenz vorbereiten
    if model_name in ["EDGE_LSTM", "1D_CNN_Model"]:
        _, _, X_test, y_test, scaler, _, _, test_df, _, features = LoadPrepareData._prepare_base_data_3D(config)
    # ...
    
    preds, timings = PipelineUtils.run_inference_with_timing(model, X_test)
    _, _, _, metrics = PipelineUtils._evaluate_model(...) # Evaluation durchführen
    metrics["avg_inference_time_ms"] = np.mean(timings)
    
    # NUR Metriken speichern
    PipelineUtils.save_evaluation_metrics(config, metrics, paths, 0.0, None)
    print(f"===== PIPELINE 2 FÜR {config.get('model_name')} ABGESCHLOSSEN =====")


def run_retraining_pipeline(config: dict):
    """Pipeline 3: Setup, Laden, Weitertrainieren, Inferenz, Speichern (ohne Skalierer)."""
    print(f"\n===== STARTE PIPELINE 3: NACHTRAINIEREN FÜR {config.get('model_name')} =====")
    
    config, paths = PipelineUtils.setup_experiment(config)
    
    model_name = config.get("model_name")
    if model_name in ["EDGE_LSTM", "1D_CNN_Model"]:
        data_tuple = LoadPrepareData._prepare_base_data_3D(config)
    else:
        data_tuple = LoadPrepareData._prepare_base_data_2D(config)
    
    X_train, y_train, _, _, _, _, _, _, _, features = data_tuple
    
    # Modell laden
    model_to_retrain = RF_Utils.load_rf_model(config.get("model_load_path")) # Beispiel
    # ...
    
    # Weitertrainieren (hier muss ggf. die train_func angepasst werden, um ein 'initial_model' zu akzeptieren)
    config['epochs'] = config.get("retrain_epochs", 5) # Weniger Epochen für Retraining
    retrained_model, history, train_time = LSTM_Utils.train_model_LSTM(config, X_train, y_train, features, initial_model=model_to_retrain)
    # ...

    # Setze ein Flag, damit der Skalierer nicht erneut gespeichert wird
    config['save_scaler_on_retrain'] = False
    return _run_inference_and_saving(retrained_model, config, paths, data_tuple, history, train_time)


# =============================================================================
# == HAUPTAUSFÜHRUNG
# =============================================================================
if __name__ == "__main__":
    
    # 1. MODELL AUSWÄHLEN
    MODEL_CHOICE = "LSTM"  # Optionen: "LSTM", "CNN", "RF", "XGB"
    config = get_config_by_name(MODEL_CHOICE)
    
    # Fügen Sie für Pipeline 2 & 3 den Pfad zum zu ladenden Modell hinzu
    # config["model_load_path"] = "path/to/your/model.keras"

    # 2. PIPELINE AUSWÄHLEN UND AUSFÜHREN
    run_full_training_pipeline(config)
    # run_inference_only_pipeline(config)
    # run_retraining_pipeline(config)