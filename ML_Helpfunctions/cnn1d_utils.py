# cnn1d_utils.py
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from typing import List, Tuple
import sys
import os

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ML_Helpfunctions import Pipeline_Utils

def build_dynamic_cnn1d(input_shape: Tuple[int, int], 
                        config: dict) -> Model:
    """
    Baut ein dynamisches 1D-CNN-Modell basierend auf der Konfiguration.

    Args:
        input_shape (tuple): Die Form der Eingabedaten (lags, n_features).
        config (dict): Konfigurationsdictionary mit Modellparametern.

    Returns:
        tf.keras.Model: Das kompilierte Keras-Modell.
    """
    num_conv_layers = config.get("num_conv_layers", 2)
    filters = config.get("filters", 64)
    kernel_size = config.get("kernel_size", 3)
    pool_size = config.get("pool_size", 2)
    dense_units = config.get("dense_units", 100)
    dropout_rate = config.get("dropout", 0.2)
    forecast_horizon = config.get("horizon", 1)

    inputs = Input(shape=input_shape)
    x = inputs

    # Convolutional Blöcke
    for i in range(num_conv_layers):
        x = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', padding='causal')(x)
        x = MaxPooling1D(pool_size=pool_size)(x)
        filters = max(filters // 2, 16) # Filter für tiefere Schichten reduzieren

    # Flatten und Dense Schichten
    x = Flatten()(x)
    x = Dense(units=dense_units, activation='relu')(x)
    x = Dropout(rate=dropout_rate)(x)
    
    # Output Schicht
    outputs = Dense(units=forecast_horizon, activation='relu')(x) # ReLU um negative Vorhersagen zu vermeiden

    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def train_model_cnn1d(config: dict, X_train: np.ndarray, y_train: np.ndarray, features: List[str]):
    """
    Baut, kompiliert und trainiert ein 1D-CNN-Modell.

    Args:
        config (dict): Konfigurationsparameter.
        X_train (np.ndarray): Trainingsdaten (3D: [samples, lags, features]).
        y_train (np.ndarray): Zielwerte (2D: [samples, horizon]).
        features (list): Liste der Feature-Namen.

    Returns:
        tuple: (model, history, train_time)
    """
    print("🚀 Starte Training für 1D-CNN-Modell...")
    
    # --- Modell bauen ---
    input_shape_cnn = (config["lags"], len(features))
    model = build_dynamic_cnn1d(input_shape=input_shape_cnn, config=config)
    
    # --- Modell kompilieren ---
    loss_function = config.get("loss", "huber_loss")
    optimizer = config.get("optimizer", "adam")
    model.compile(optimizer=optimizer, loss=loss_function, metrics=config.get("metrics", ["mae"]))
    model.summary()

    # --- Trainings- und Validierungsdaten vorbereiten ---
    X_fit, y_fit, X_val, y_val = Pipeline_Utils.create_timeseries_validation_split(X_train, y_train, config)
    val_data = (X_val, y_val) if X_val is not None and y_val is not None else None

    # --- Callbacks holen ---
    callbacks = Pipeline_Utils.get_keras_callbacks(config)

    # --- Modell trainieren ---
    start_time = time.time()
    history = model.fit(
        X_fit, y_fit,
        validation_data=val_data,
        epochs=config.get("epochs", 50),
        batch_size=config.get("batch_size", 32),
        callbacks=callbacks,
        verbose=config.get("keras_verbose", 1)
    )
    duration = time.time() - start_time
    print(f"✅ 1D-CNN Training abgeschlossen. Dauer: {duration:.2f} Sekunden.")

    return model, history, duration