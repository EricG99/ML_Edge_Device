import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from pathlib import Path

# Feature Engineering-Modul (optional)
import Feature_Engeneering as fe

from pathlib import Path



# ---------------------------------------------------
# HILFSFUNKTIONEN
# ---------------------------------------------------
def _get_file_path(config: dict) -> Path:
    """
    Gibt den vollständigen Pfad zur Datei zurück, basierend auf dem Input-Pfad
    und einem Dateinamen, der in config["filenames"] unter filename_key gespeichert ist.

    Args:
        config (dict): Konfigurationsdictionary mit 'paths' und 'filenames'.
        filename_key (str): Schlüssel für den Dateinamen in config["filenames"].

    Returns:
        Path: Pfad zur Datei.

    Raises:
        KeyError: Wenn 'input' oder der Dateiname nicht im config enthalten ist.
        FileNotFoundError: Wenn die Datei nicht existiert.
    """
    try:
        input_dir = Path(config["paths"]["input"])
        filename = config["dataset"]  
    except KeyError as e:
        raise KeyError(f"Fehlender Schlüssel in config: {e}")

    file_path = input_dir / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {file_path}")

    return file_path


# ---------------------------------------------------
# LADEN TRAIN/TEST
# ---------------------------------------------------
def load_train_data_with_datetime(train_period_start: str,
                                  train_period_end: str,
                                  config: dict,
                                  make_date_as_index: bool = True) -> pd.DataFrame:
    start_date = pd.to_datetime(train_period_start)
    end_date = pd.to_datetime(train_period_end)

    file_path = _get_file_path(config)
    df = pd.read_csv(file_path, sep=";")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[(df["Datetime"] >= start_date) & (df["Datetime"] <= end_date)].copy()

    if make_date_as_index:
        df = df.set_index("Datetime")
    idx = df.index if make_date_as_index else pd.to_datetime(df["Datetime"])
    df["hour"] = idx.hour
    df["weekday"] = idx.dayofweek

    print(f"Loaded {len(df)} rows from '{file_path}'")
    return df


def load_test_data_with_datetime(test_period_start: str,
                                 test_period_end: str,
                                 config: dict,
                                 make_date_as_index: bool = True) -> pd.DataFrame:
    start_date = pd.to_datetime(test_period_start)
    end_date = pd.to_datetime(test_period_end)

    file_path = _get_file_path(config)
    df = pd.read_csv(file_path, sep=";")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[(df["Datetime"] >= start_date) & (df["Datetime"] <= end_date)].copy()

    if make_date_as_index:
        df = df.set_index("Datetime")
    idx = df.index if make_date_as_index else pd.to_datetime(df["Datetime"])
    df["hour"] = idx.hour
    df["weekday"] = idx.dayofweek

    print(f"Loaded {len(df)} rows for test from '{file_path}'")
    return df


def _load_full_timeseries(config: dict,
                          make_date_as_index: bool = True) -> pd.DataFrame:
    file_path = _get_file_path(config)
    df = pd.read_csv(file_path, sep=";")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime")

    if make_date_as_index:
        df = df.set_index("Datetime")
        idx = df.index
    else:
        idx = pd.to_datetime(df["Datetime"])

    df["millisecond"] = idx.microsecond // 1000
    df["second"] = idx.second
    df["minute"] = idx.minute
    df["hour"] = idx.hour
    df["day"] = idx.day
    df["weekday"] = idx.dayofweek
    df["week"] = idx.isocalendar().week.astype(int)
    df["month"] = idx.month
    df["year"] = idx.year

    print(f"Geladen: {len(df)} Zeilen aus '{file_path}' mit Zeitfeatures")
    return df


# ---------------------------------------------------
# TRAIN / TEST SPLIT NACH FRAKTION
# ---------------------------------------------------
def load_train_data_by_fraction(config: dict,
                                train_fraction: float = 0.75,
                                make_date_as_index: bool = True) -> pd.DataFrame:
    df = _load_full_timeseries(config, make_date_as_index)
    split_idx = int(len(df) * train_fraction)
    return df.iloc[:split_idx].copy()


def load_test_data_by_fraction(config: dict,
                               train_fraction: float = 0.75,
                               make_date_as_index: bool = True) -> pd.DataFrame:
    df = _load_full_timeseries(config, make_date_as_index)
    split_idx = int(len(df) * train_fraction)
    return df.iloc[split_idx:].copy()


# ---------------------------------------------------
# SCALING UND SLIDING WINDOWS
# ---------------------------------------------------
def load_and_scale_data(train_data: np.ndarray,
                        test_data: np.ndarray,
                        scaler_type: str = 'minmax',
                        scale_other: bool = True):
    if scaler_type == 'robust':
        scaler_main = RobustScaler()
    else:
        scaler_main = MinMaxScaler(feature_range=(0, 1))

    train_main_scaled = scaler_main.fit_transform(train_data[:, :2])
    test_main_scaled = scaler_main.transform(test_data[:, :2])

    if scale_other:
        train_other_scaled = np.zeros_like(train_data[:, 2:])
        test_other_scaled = np.zeros_like(test_data[:, 2:])
        for i in range(train_data[:, 2:].shape[1]):
            scaler_other = MinMaxScaler()
            train_other_scaled[:, i] = scaler_other.fit_transform(train_data[:, [i + 2]]).flatten()
            test_other_scaled[:, i] = scaler_other.transform(test_data[:, [i + 2]]).flatten()
    else:
        train_other_scaled = train_data[:, 2:]
        test_other_scaled = test_data[:, 2:]

    train_scaled = np.hstack([train_main_scaled, train_other_scaled])
    test_scaled = np.hstack([test_main_scaled, test_other_scaled])

    print(f"[load_and_scale_data] Train: {train_scaled.shape}, Test: {test_scaled.shape}")
    return train_scaled, test_scaled, scaler_main

def convert_data_to_sliding_window(data_array: np.ndarray,
                                   lag_horizon: int,
                                   forecast_horizon: int = 1,
                                   shift: int = 1):
    X, y = [], []
    for i in range(0, len(data_array) - lag_horizon - forecast_horizon + 1, shift):
        X.append(data_array[i:i + lag_horizon])
        y.append(data_array[i + lag_horizon:i + lag_horizon + forecast_horizon, 0])
    return np.array(X), np.array(y)

def create_flat_windows(data: np.ndarray,
                        lag_horizon: int,
                        forecast_horizon: int = 1,
                        shift: int = 1):
    X, y = [], []
    for i in range(0, len(data) - lag_horizon - forecast_horizon + 1, shift):
        X.append(data[i:i + lag_horizon].flatten())
        y.append(data[i + lag_horizon:i + lag_horizon + forecast_horizon, 0])
    return np.array(X), np.array(y)

def create_sliding_windows(data: np.ndarray,
                           lag_horizon: int,
                           forecast_horizon: int = 1,
                           shift: int = 1):
    X, y = [], []
    for i in range(0, len(data) - lag_horizon - forecast_horizon + 1, shift):
        X.append(data[i:i + lag_horizon])
        y.append(data[i + lag_horizon:i + lag_horizon + forecast_horizon, 0])
    return np.array(X), np.array(y)


def prepare_3d_train_data(
    base_train_data,
    base_test_data,
    feature_list,
    used_lags=2,
    forecast_horizon=1,
    scaler_type="minmax",
    scale_target=False
):
    from sklearn.preprocessing import MinMaxScaler, RobustScaler

    scaler_class = RobustScaler if scaler_type == "robust" else MinMaxScaler
    target_column = base_train_data.columns[0]  # Achtung: ggf. explizit übergeben

    # Nur die gewünschten Features verwenden
    combined_3D = pd.concat([base_train_data[feature_list], base_test_data[feature_list]])
    scaler_3D = scaler_class()
    train_scaled_3D = scaler_3D.fit_transform(base_train_data[feature_list])
    test_scaled_3D = scaler_3D.transform(combined_3D)

    X_3D, y_3D = convert_data_to_sliding_window(
        train_scaled_3D,
        lag_horizon=used_lags,
        forecast_horizon=forecast_horizon,
        shift=1
    )

    scaler_y_3D = None
    if scale_target:
        scaler_y_3D = scaler_class()
        y_3D = scaler_y_3D.fit_transform(y_3D)

    return X_3D, y_3D, scaler_3D, scaler_y_3D, train_scaled_3D, test_scaled_3D


def prepare_2d_train_data(
    full_feature_train_data,
    full_feature_test_data,
    used_lags=12,
    scaler_type="minmax",
    scale_target=False,
    scale_features=False  # NEU: Steuerung, ob skaliert wird
):
    from sklearn.preprocessing import MinMaxScaler, RobustScaler

    scaler_class = RobustScaler if scaler_type == "robust" else MinMaxScaler
    target_column = full_feature_train_data.columns[0]

    scaler_2D = None
    X_2D = full_feature_train_data.values

    if scale_features:
        combined_2D = pd.concat([full_feature_train_data, full_feature_test_data])
        scaler_2D = scaler_class()
        scaler_2D.fit(combined_2D)
        X_2D = scaler_2D.transform(full_feature_train_data)

    y_2D_raw = full_feature_train_data.iloc[used_lags:][target_column].values
    X_2D = X_2D[used_lags : used_lags + len(y_2D_raw)]

    scaler_y_2D = None
    if scale_target:
        scaler_y_2D = scaler_class()
        y_2D = scaler_y_2D.fit_transform(y_2D_raw.reshape(-1, 1)).flatten()
    else:
        y_2D = y_2D_raw

    return X_2D, y_2D, scaler_2D, scaler_y_2D, None  # letzter Wert (X_2D_full_scaled) wird nicht gebraucht




def prepare_test_data_3D(
    base_test_data: pd.DataFrame,
    feature_list: list,
    scaler_3D,
    scaler_y=None,
    used_lags: int = 1,
    forecast_horizon: int = 1,
    scale_target: bool = False,
):
    test_scaled_3D = scaler_3D.transform(base_test_data[feature_list].values)

    X_3D, y_3D = convert_data_to_sliding_window(
        test_scaled_3D,
        lag_horizon=used_lags,
        forecast_horizon=forecast_horizon,
        shift=1
    )
    if scale_target and scaler_y is not None:
        y_3D = scaler_y.transform(y_3D.reshape(-1, y_3D.shape[-1]))
    return X_3D, y_3D


def prepare_test_data_2D(
    full_feature_test_data: pd.DataFrame,
    scaler_2D,
    scaler_y=None,
    used_lags: int = 1,
    scale_target: bool = False,
    target_column: str = None,
):
    X_2D = full_feature_test_data.values
    if scaler_2D is not None:
        X_2D = scaler_2D.transform(X_2D)
    X_2D = X_2D[used_lags:]
    if target_column is None:
        target_column = full_feature_test_data.columns[0]
    y_2D_raw = full_feature_test_data.iloc[used_lags:][target_column].values
    y_2D = y_2D_raw[:len(X_2D)]
    if scale_target and scaler_y is not None:
        y_2D = scaler_y.transform(y_2D.reshape(-1, 1)).flatten()
    return X_2D, y_2D



def create_multi_step_target(y, horizon):
    """Convert 1D y into 2D array with horizon columns"""
    y = np.asarray(y)
    return np.column_stack([y[i:i-horizon or None] for i in range(horizon)])

def _prepare_base_data_shared(config: dict) -> tuple:
    """
    Gemeinsame Vorverarbeitungsschritte für 2D und 3D:
    - Läd Trainings- und Testdaten
    - Führt Feature Engineering durch
    - Gibt vorbereitete DataFrames und Featureinformationen zurück
    """
    print("\nSchritt 1: Lade Trainings- und Testdaten...")

    train_df = load_train_data_by_fraction(
        config=config,
        train_fraction=config["train_fraction"],
        make_date_as_index=True
    )
    test_df = load_test_data_by_fraction(
        config=config,
        train_fraction=config["train_fraction"],
        make_date_as_index=True
    )

    print("\nSchritt 2: Feature Engineering...")
    train_df, train_features_dict = fe.add_all_features(
        train_df,
        config
    )
    test_df, _ = fe.add_all_features(
        test_df,
        config
    )

    print(f"Trainingsdaten: {train_df.shape}, Testdaten: {test_df.shape}")
    print("Verfügbare Features (Train):", train_features_dict["all"])

    full_feature_list = train_features_dict["all"]

    return train_df, test_df, train_features_dict, full_feature_list


def _prepare_base_data_2D(config: dict) -> tuple:
    train_df, test_df, train_features_dict, full_feature_list = _prepare_base_data_shared(config)
    base_features = config["base_features"]

    X_train_2D, y_train_2D, scaler_2D, y_scaler, _ = prepare_2d_train_data(
        full_feature_train_data=train_df[full_feature_list],
        full_feature_test_data=test_df[full_feature_list],
        used_lags=config["lags"],
        scale_target=config.get("scale_target", False),
        scaler_type=config.get("scaler_type", "minmax"),
        scale_features=config.get("scale_other_features", False)  
    )

    X_test_2D, y_test_2D = prepare_test_data_2D(
        full_feature_test_data=test_df[full_feature_list],
        scaler_2D=scaler_2D,
        scaler_y=y_scaler,
        used_lags=config["lags"],
        scale_target=config.get("scale_target", False),
        target_column=base_features[0]
    )

    if config["horizon"] > 1:
        y_train_2D = create_multi_step_target(y_train_2D, config["horizon"])
        y_test_2D = create_multi_step_target(y_test_2D, config["horizon"])
        X_train_2D = X_train_2D[:len(y_train_2D)]
        X_test_2D = X_test_2D[:len(y_test_2D)]

    print(f"RF Full-Feature Datenformate - X_train_2D: {X_train_2D.shape}, y_train_2D: {y_train_2D.shape}")

    return (
        X_train_2D,
        y_train_2D,
        X_test_2D,
        y_test_2D,
        scaler_2D,
        y_scaler,
        train_df,
        test_df,
        train_features_dict,
        full_feature_list
    )


def _prepare_base_data_3D(config: dict) -> tuple:
    train_df, test_df, train_features_dict, full_feature_list = _prepare_base_data_shared(config)

    X_train_3D, y_train_3D, scaler_3D, y_scaler, _, _ = prepare_3d_train_data(
        base_train_data=train_df,
        base_test_data=test_df,
        feature_list=full_feature_list,
        used_lags=config["lags"],
        forecast_horizon=config["horizon"],
        scaler_type=config.get("scaler_type", "minmax"),
        scale_target=config.get("scale_target", False)
    )

    X_test_3D, y_test_3D = prepare_test_data_3D(
        base_test_data=test_df,
        feature_list=full_feature_list,
        scaler_3D=scaler_3D,
        scaler_y=y_scaler,
        used_lags=config["lags"],
        forecast_horizon=config["horizon"],
        scale_target=config.get("scale_target", False)
    )

    return (
        X_train_3D,
        y_train_3D,
        X_test_3D,
        y_test_3D,
        scaler_3D,
        y_scaler,
        train_df,
        test_df,
        train_features_dict,
        full_feature_list
    )


def _create_train_val_split(X_train, y_train, validation_fraction=0.2):
    """
    Teilt die Trainingsdaten in Trainings- und Validierungssets auf.

    Args:
        X_train (np.ndarray): Die Eingabedaten für das Training.
        y_train (np.ndarray): Die Zielwerte für das Training.
        validation_fraction (float, optional): Der Anteil der Daten, der für die Validierung verwendet werden soll.
            Defaults to 0.2.

    Returns:
        tuple: (X_train, X_val, y_train, y_val) - Die aufgeteilten Daten.
    """
    train_idx = int(len(X_train) * (1 - validation_fraction))  # Korrigierte Berechnung des Trainingsindex
    X_train_split, X_val = X_train[:train_idx], X_train[train_idx:]
    y_train_split, y_val = y_train[:train_idx], y_train[train_idx:]
    return X_train_split, X_val, y_train_split, y_val