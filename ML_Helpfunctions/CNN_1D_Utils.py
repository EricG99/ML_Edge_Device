import time
from typing import List
import numpy as np
import tensorflow as tf
import os
import joblib
import traceback
import pandas as pd

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from ML_Helpfunctions import Pipeline_Utils as PipelineUtils


def build_dynamic_1D_CNN(input_shape, num_conv_blocks, filters, kernel_size, pool_size, dense_units, dropout, forecast_horizon):
    """Baut ein dynamisches 1D-CNN-Modell."""
    model = tf.keras.Sequential(name="Sequential_1D_CNN")
    
    # Input-Schicht / Erster Convolutional Block
    model.add(tf.keras.layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        activation='relu',
        input_shape=input_shape,
        padding='same'  # <-- HINZUGEFÜGT: Verhindert die Verkürzung der Sequenz
    ))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size))
    model.add(tf.keras.layers.Dropout(dropout))

    # Weitere optionale Convolutional Blöcke
    for i in range(num_conv_blocks - 1):
        filters *= 2
        model.add(tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same'  # <-- HINZUGEFÜGT: Verhindert die Verkürzung der Sequenz
        ))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size))
        model.add(tf.keras.layers.Dropout(dropout))

    # Flatten- und Dense-Schichten zur Interpretation
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=dense_units, activation='relu'))
    model.add(tf.keras.layers.Dropout(dropout))
    
    # Output-Schicht
    model.add(tf.keras.layers.Dense(units=forecast_horizon))
    
    model.summary()
    return model


def train_model_1D_CNN(config: dict, X_train: np.ndarray,
                       y_train: np.ndarray, features: List[str]):
    """
    Baut, kompiliert und trainiert ein 1D-CNN-Modell.
    Verwendet zentrale Hilfsfunktionen für den Validation-Split und Callbacks.
    """
    print("--- Starte Training für 1D-CNN-Modell ---")
    
    # 1. Modell bauen (unverändert)
    input_shape_cnn = (config["lags"], len(features))
    model = build_dynamic_1D_CNN(
        input_shape=input_shape_cnn,
        num_conv_blocks=config.get("num_conv_blocks", 1),
        filters=config.get("filters", 64),
        kernel_size=config.get("kernel_size", 3),
        pool_size=config.get("pool_size", 2),
        dense_units=config.get("dense_units", 50),
        dropout=config.get("dropout", 0.1),
        forecast_horizon=config["horizon"]
    )
    
    # 2. Modell kompilieren (unverändert)
    loss_function = config.get("loss", "huber_loss")
    optimizer = config.get("optimizer", "adam")
    model.compile(optimizer=optimizer, loss=loss_function, metrics=["mae"])
    
    # 3. Daten splitten und Callbacks vorbereiten
    X_fit, y_fit, X_val, y_val = PipelineUtils.create_timeseries_validation_split(
        X_train, y_train, config
    )
    val_data = (X_val, y_val) if X_val is not None else None
    
    callbacks = PipelineUtils.get_keras_callbacks(config)

    # 4. Modell trainieren
    start = time.time()
    history = model.fit(
        X_fit, y_fit,
        validation_data=val_data,
        epochs=config.get("epochs", 10),
        batch_size=config.get("batch_size", 32),
        callbacks=callbacks,
        verbose=config.get("keras_verbose", 1)
    )
    duration = time.time() - start
    print(f"1D-CNN-Modell Training abgeschlossen in {duration:.2f} Sekunden.")

    return model, history, duration



def load_model_CNN(model_path, model_name):
    """Lädt ein gespeichertes Keras-Modell."""
    # Diese Funktion ist identisch zu load_model_LSTM
    full_path = os.path.join(model_path, model_name)
    print(f"Lade Modell von: {full_path}")
    return tf.keras.models.load_model(full_path)

def run_inference_cnn(model, X_test):
    """Führt Inferenz mit dem CNN-Modell durch."""
    # Diese Funktion ist identisch zu run_inference_lstm
    return model.predict(X_test)



def save_cnn_metrics_prediction(config: dict, **kwargs) -> dict:
    """
    Speichert Evaluationsmetriken und Vorhersagedatei für ein 1D-CNN-Modell.
    """
    print("--- Speichere Evaluationsmetriken für CNN ---")
    metrics_results = PipelineUtils._save_metrics_prediction_gerneral(config=config, **kwargs)
    return metrics_results


def save_cnn_deployment_artifacts(
    config: dict,
    model: tf.keras.Model,
    history: dict,
    scaler,
    paths: dict, 
    **kwargs
) -> dict:
    """
    Orchestriert das Speichern aller CNN-Deployment-Artefakte
    durch Instanziierung und Verwendung der zentralen ModelSaver-Klasse.
    """
    print(f"--- Initialisiere Speicherprozess für {config.get('model_name')} via ModelSaver ---")
    
    saver = PipelineUtils.ModelScalerSaver(config, paths=paths)
    
    # Der Rest der Funktion ist korrekt
    deployment_results = saver.save_artifacts(
        model=model, 
        scaler=scaler,
        history=history,
    )
    
    return deployment_results