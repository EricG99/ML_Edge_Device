
import time
import logging
from typing import Tuple, List, Optional
from xml.parsers.expat import model

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, Dropout,
    GlobalAveragePooling1D, Dense
)
from tensorflow.keras.optimizers import Adam

# Project utils (callbacks, validation split, artifact saving are handled elsewhere)
try:
    from ML_Helpfunctions import Pipeline_Utils
except Exception as e:
    logging.warning("Pipeline_Utils nicht gefunden (dev env). Training läuft trotzdem: %s", e)

logger = logging.getLogger("CNN1DUtils")
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def build_cnn1d_model(
    input_shape: Tuple[int, int],
    blocks: int = 2,
    base_filters: int = 64,
    kernel_size: int = 5,
    dropout: float = 0.1,
    horizon: int = 1,
    activation: str = "relu",
) -> Sequential:
    """
    Baut ein leichtgewichtiges 1D-CNN für Zeitreihen‑Forecasting.

    Args:
        input_shape: (lags, n_features)
        blocks: Anzahl der Conv-Bausteine (Conv1D->BN->Act->Dropout)
        base_filters: Anzahl Filter im ersten Block (wird pro Block halbiert)
        kernel_size: Kernelgröße der Convs
        dropout: Dropoutrate zwischen den Blöcken
        horizon: Anzahl der Vorhersageschritte (Ausgabedimension)
        activation: Aktivierungsfunktion (z.B. 'relu', 'gelu')

    Returns:
        Kompiliertes tf.keras.Sequential Modell
    """
    model = Sequential([Input(shape=input_shape)])
    filters = int(base_filters)

    for i in range(blocks):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding="same"))
        model.add(BatchNormalization())
        model.add(Activation(activation))
        if dropout and dropout > 0:
            model.add(Dropout(dropout))
        # leichte "Feature-Verjüngung" pro Block
        filters = max(filters // 2, 4)

    model.add(GlobalAveragePooling1D())
    model.add(Dense(max(filters, 8)))
    model.add(Activation(activation))
    model.add(Dense(horizon, activation="linear"))

    model.compile(
        optimizer=Adam(),
        loss=tf.keras.losses.Huber(),  # oder einfach: "huber"
        metrics=["mae"]
    )
    return model


def train_model_CNN1D(
    config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    features: List[str]
):
    """
    Trainiert das CNN1D‑Modell. Nutzt die gleichen Callback/Validierungs-Helfer
    wie LSTM, falls verfügbar.

    Returns:
        (model, history, train_time_seconds)
    """
    # Eingabeform: (samples, lags, features)
    input_shape = (int(config.get("lags", X_train.shape[1])), X_train.shape[2])
    horizon = int(config.get("horizon", y_train.shape[1] if y_train.ndim > 1 else 1))

    model = build_cnn1d_model(
        input_shape=input_shape,
        blocks=int(config.get("cnn_blocks", 2)),
        base_filters=int(config.get("cnn_base_filters", 64)),
        kernel_size=int(config.get("cnn_kernel_size", 5)),
        dropout=float(config.get("cnn_dropout", 0.1)),
        horizon=horizon,
        activation=str(config.get("cnn_activation", "relu"))
    )

    # Validation‑Split (zeitlich) & Callbacks
    if "Pipeline_Utils" in globals():
        X_fit, y_fit, X_val, y_val = Pipeline_Utils.create_timeseries_validation_split(
            X_train, y_train, config
        )
        callbacks = Pipeline_Utils.get_keras_callbacks(config)
    else:
        X_fit, y_fit, X_val, y_val = X_train, y_train, None, None
        callbacks = []

    val_data = (X_val, y_val) if X_val is not None and y_val is not None else None

    start = time.perf_counter()
    history = model.fit(
        X_fit, y_fit,
        validation_data=val_data,
        epochs=int(config.get("epochs", 10)),
        batch_size=int(config.get("batch_size", 32)),
        callbacks=callbacks,
        verbose=int(config.get("keras_verbose", 1)),
    )
    train_time = time.perf_counter() - start

    return model, history, train_time
