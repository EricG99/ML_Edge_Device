import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from pathlib import Path
import sys
import logging

# Feature Engineering-Modul (optional)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import ML_Helpfunctions.Feature_Engeneering as fe

from pathlib import Path

# ---------------------------------------------------
# HILFSFUNKTIONEN
# ---------------------------------------------------
def _get_file_path(config: dict) -> Path:
    """
    Ermittelt den vollständigen Pfad zur Datendatei robust aus config['paths'] und 'dataset'.
    Akzeptiert mehrere mögliche Keys für den Eingabeordner.
    """
    try:
        paths = config["paths"]
    except KeyError:
        raise KeyError("Fehlender Schlüssel: config['paths']")

    # akzeptiere mehrere Namensvarianten
    for key in ("input_data", "Input_Data", "input"):
        data_dir = paths.get(key)
        if data_dir:
            break
    else:
        raise KeyError(
            "Kein Eingabeordner in config['paths'] gefunden. Erwartet einer von: "
            "'input_data', 'Input_Data', 'input'."
        )

    filename = config.get("dataset")
    if not filename:
        raise KeyError("Fehlender Schlüssel: config['dataset'] (Dateiname)")

    file_path = Path(data_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Datendatei nicht gefunden: {file_path}")

    print(f"✔️ Datendatei gefunden: {file_path}")
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
    df = pd.read_csv(file_path, sep=",")
    df.columns = df.columns.str.lower()
    

    # anstelle der fehleranfälligen 'datetime'-String-Spalte.
    if "time" not in df.columns:
        raise ValueError(f"Die CSV-Datei '{file_path}' muss eine 'time'-Spalte mit Unix-Timestamps enthalten.")
    
    # Wandle den Unix-Timestamp in ein Datetime-Objekt um und weise es der 'datetime'-Spalte zu.
    df["datetime"] = pd.to_datetime(df["time"], unit='ms')
    
    df = df.sort_values("datetime")

    # Der restliche Code, der Zeit-Features aus der 'datetime'-Spalte extrahiert,
    # kann nun unverändert bleiben, da die Spalte jetzt korrekte Werte enthält.
    idx = df["datetime"]
    df["millisecond"] = idx.dt.microsecond // 1000
    df["second"] = idx.dt.second
    df["minute"] = idx.dt.minute
    df["hour"] = idx.dt.hour
    df["day"] = idx.dt.day
    df["weekday"] = idx.dt.dayofweek
    df["week"] = idx.dt.isocalendar().week.astype(int)
    df["month"] = idx.dt.month
    df["year"] = idx.dt.year

    if make_date_as_index:
        df = df.set_index("datetime")
    
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

    return X_2D, y_2D, scaler_2D, scaler_y_2D




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

    # --- HIER IST DIE GEWÜNSCHTE DEBUG-AUSGABE ---
    print("\n[DEBUG] Spaltennamen im Trainings-DataFrame NACH Feature Engineering:")
    # Wir lassen uns die komplette Liste aller Spalten ausgeben
    column_list = train_df.columns.tolist()
    print(column_list)
    print(f"[DEBUG] Anzahl der Spalten: {len(column_list)}")
    # --- ENDE DEBUG-AUSGABE ---

    print(f"\nTrainingsdaten (Shape): {train_df.shape}, Testdaten (Shape): {test_df.shape}")
    print("Verfügbare Features (Train):", train_features_dict["all"])

    full_feature_list = train_features_dict["all"]

    return train_df, test_df, train_features_dict, full_feature_list


def _prepare_base_data_2D(config: dict) -> tuple:
    train_df, test_df, train_features_dict, full_feature_list = _prepare_base_data_shared(config)
    base_features = config["base_features"]

    X_train_2D, y_train_2D, scaler_2D, y_scaler = prepare_2d_train_data(
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

# In Ihrem Hilfs-Modul (z.B. Pipeline_Utils.py)

def convert_data_to_multi_output_window(data: np.ndarray, config: dict):
    """
    Konvertiert einen 2D-Array in 3D-Fenster (für X) und 2D-Fenster (für y).
    y enthält nun 'horizon' zukünftige Schritte.
    """
    lags = config.get('lags', 1)
    horizon = config.get('horizon', 1)
    
    X, y = [], []
    # Die Schleife muss früher enden, um Platz für den Horizont zu haben
    for i in range(len(data) - lags - horizon + 1):
        X.append(data[i:(i + lags)])
        y.append(data[i + lags:(i + lags + horizon)])
    
    return np.array(X), np.array(y)

class DataPipelineBase:
    """
    Eine abstrakte Basisklasse, die die gemeinsame Logik für Datenpipelines kapselt.
    Verwaltet das Laden von Daten, Feature Engineering und den Zustand (Scaler, Features).
    """
    def __init__(self, config: dict):
        """
        Initialisiert die Pipeline mit der Konfiguration.

        Args:
            config (dict): Ein Konfigurationswörterbuch, das alle Einstellungen enthält.
        """
        self.config = config
        
        # Zustandsobjekte, die während des Trainings initialisiert werden
        self.scaler = None
        self.y_scaler = None
        self.full_feature_list = None
        self.train_features_dict = None
        
        # Puffer für Live-Daten (z.B. für MQTT)
        self._live_data_buffer = pd.DataFrame()

    def _load_data(self, mode: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        """
        Interne Methode zum Laden von Daten basierend auf der Strategie in der Konfiguration.
        
        Args:
            mode (str): Entweder 'train' oder 'test'.
            
        Returns:
            Ein Tuple mit (train_df, test_df). test_df kann None sein.
        """
        strategy = self.config.get("loading_strategy", "split")
        print(f"\nLade Daten im Modus '{mode}' mit Strategie '{strategy}'...")

        if strategy == "split":            
            train_df = load_train_data_by_fraction(config=self.config, make_date_as_index=True)
            test_df = load_test_data_by_fraction(config=self.config, make_date_as_index=True)
            return train_df, test_df

        elif strategy == "separate_csv":
            if mode == 'train':
                # Im Trainingsmodus wird die Test-CSV nur für eine robustere Skalierung benötigt
                train_df = _load_full_timeseries(config=self.config, make_date_as_index=True)
                test_df = pd.read_csv(self.config['test_csv_path'], index_col='datetime', parse_dates=True) if 'test_csv_path' in self.config else None
                return train_df, test_df
            else: # mode == 'test'
                test_df = pd.read_csv(self.config['test_csv_path'], index_col='datetime', parse_dates=True)
                return None, test_df
        
        elif strategy == "live_mqtt":
             if mode == 'train':
                train_df = _load_full_timeseries(config=self.config, make_date_as_index=True)
                return train_df, None
             else: # mode == 'test'
                print("Im MQTT-Modus werden Testdaten live verarbeitet, kein initiales Laden.")
                return None, None
        
        else:
            raise ValueError(f"Unbekannte Lade-Strategie: {strategy}")

    def prepare_training_data(self):
        """Muss von den Unterklassen implementiert werden."""
        raise NotImplementedError("Diese Methode muss in der Unterklasse implementiert werden.")

    def prepare_testing_data(self):
        """Muss von den Unterklassen implementiert werden."""
        raise NotImplementedError("Diese Methode muss in der Unterklasse implementiert werden.")
        
    def prepare_live_data_point(self, mqtt_payload: dict):
        """Muss von den Unterklassen implementiert werden."""
        raise NotImplementedError("Diese Methode muss in der Unterklasse implementiert werden.")


class DataPipeline2D(DataPipelineBase):
    """
    Eine Klasse zur Kapselung des gesamten Datenvorbereitungsprozesses für 2D-Modelle.
    Der Scaler wird hier nur mit Trainingsdaten trainiert.
    """
    def __init__(self, config: dict):
        super().__init__(config)


    
        
    def _prepare_and_scale_training_data(self, train_df_featured: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialisiert und trainiert die Scaler ausschließlich auf den Trainingsdaten
        und gibt die skalierten 2D-Arrays zurück.
        """
        scaler_type = self.config.get("scaler_type", "minmax")
        scale_target = self.config.get("scale_target", False)
        scale_features = self.config.get("scale_other_features", False)
        target_column = self.config["base_features"][0]

        scaler_class = RobustScaler if scaler_type == "robust" else MinMaxScaler

        # 1. Feature-Skalierung (X)
        X_train = train_df_featured.values
        if scale_features:
            print("Passe Feature-Scaler (scaler) auf Trainingsdaten an...")
            self.scaler = scaler_class()
            # WICHTIG: .fit_transform() wird NUR auf den Trainingsdaten ausgeführt.
            X_train = self.scaler.fit_transform(X_train)
        
        # 2. Target-Skalierung (y)
        y_train_raw = train_df_featured[target_column].values
        y_train = y_train_raw

        if scale_target:
            print("Passe Target-Scaler (y_scaler) auf Trainingsdaten an...")
            self.y_scaler = scaler_class()
            y_train = self.y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
            
        return X_train, y_train

    def prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Führt die komplette Pipeline für die Trainingsdaten aus.
        """
        print("--- Starte 2D Trainings-Pipeline ---")
        
        # Schritt 1: Lade Trainings-CSV (Test-CSV wird hier ignoriert)
        train_df, _ = self._load_data(mode='train')

        # Schritt 2: Feature Engineering
        print("\nSchritt 2: Feature Engineering...")
        train_df_featured, self.train_features_dict = fe.add_all_features(train_df, self.config)
        self.full_feature_list = self.train_features_dict["all"]
        
        original_rows = len(train_df_featured)
        train_df_featured.dropna(inplace=True)
        print(f"Entferne Zeilen mit NaNs nach FE. Shape: {original_rows} -> {len(train_df_featured)}")

        # Schritt 3: Skalierung und finale Vorbereitung (nur auf Trainingsdaten)
        print("\nSchritt 3: Skalierung und finale 2D-Formatierung...")
        X_train_2D, y_train_2D = self._prepare_and_scale_training_data(
            train_df_featured=train_df_featured[self.full_feature_list]
        )

        # Schritt 4: Multi-Step Target (optional)
        if self.config.get("horizon", 1) > 1:
            print(f"\nSchritt 4: Erstelle Multi-Step Target für Horizont={self.config['horizon']}...")
            y_train_2D = create_multi_step_target(y_train_2D, self.config["horizon"])
            X_train_2D = X_train_2D[:len(y_train_2D)]

        print(f"\nTrainings-Pipeline abgeschlossen. Shapes: X_train: {X_train_2D.shape}, y_train: {y_train_2D.shape}")
        return X_train_2D, y_train_2D
        
    # Die Methode prepare_testing_data bleibt unverändert.

class DataPipeline3D(DataPipelineBase):
    """
    Eine Klasse zur Kapselung des gesamten Datenvorbereitungsprozesses für 3D-Modelle.
    Der Scaler wird hier nur mit Trainingsdaten trainiert.
    """
    def __init__(self, config: dict):
        super().__init__(config)

    def _create_windows(self, data, config):
        """Erstellt 3D-Fenster für LSTM-Modelle mit Multi-Output-y."""
        logging.info("Erstelle 3D-Fenster und Multi-Output-Zielvektoren...")
        
        X, y = convert_data_to_multi_output_window(data, config)
    
        # Die Form von y ist jetzt (samples, horizon), wir brauchen aber nur die Zielvariable
        # Wir nehmen an, dass die Zielvariable die ERSTE Spalte im ursprünglichen Datensatz war.
        target_col_index = 0 
        y = y[:, :, target_col_index]

        # Stellen Sie sicher, dass X 3D ist, auch wenn es nur einen Lag gibt.
        if X.ndim == 2:
            X = np.expand_dims(X, axis=2)
            
        return X, y

    def prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Führt die komplette Pipeline für die Trainingsdaten aus und formt sie in 3D.
        Der Scaler wird ausschließlich auf den Trainingsdaten angepasst.
        """
        print("--- Starte 3D Trainings-Pipeline ---")
        
        # Schritt 1: Lade Daten (Testdaten werden nicht mehr für den Scaler verwendet)
        train_df, _ = self._load_data(mode='train') 

        # Schritt 2: Feature Engineering
        print("\nSchritt 2: Feature Engineering...")
        train_df_featured, self.train_features_dict = fe.add_all_features(train_df, self.config)
        self.full_feature_list = self.train_features_dict["all"]
        print(f"Verfügbare Features: {self.full_feature_list}")

        # WICHTIG: NaN-Zeilen entfernen, die durch rollierende Features entstehen,
        # BEVOR der Scaler angepasst wird.
        train_df_featured.dropna(inplace=True)
        print(f"Nach FE und dropna verbleiben {len(train_df_featured)} Zeilen für das Training.")
        
        # Schritt 3: Scaler ausschließlich auf Trainingsdaten anpassen
        print("\nSchritt 3: Scaler anpassen (nur auf Trainingsdaten)...")
        scaler_class = RobustScaler if self.config.get("scaler_type", "minmax") == "robust" else MinMaxScaler
        self.scaler = scaler_class()
        self.scaler.fit(train_df_featured[self.full_feature_list])
        print("Feature-Scaler wurde ausschließlich auf den Trainingsdaten angepasst.")
        
        if self.config.get("scale_target", False):
            self.y_scaler = scaler_class()
            target_col = self.config["base_features"][0]
            # y_scaler wird ebenfalls nur auf den Target-Werten der Trainingsdaten angepasst
            self.y_scaler.fit(train_df_featured[[target_col]])
            print("Target-Scaler wurde ausschließlich auf den Trainingsdaten angepasst.")

        # Schritt 4: Trainingsdaten transformieren und in Sliding Windows umwandeln
        print("\nSchritt 4: Skalierung und 3D-Windowing für Trainingsdaten...")
        train_scaled = self.scaler.transform(train_df_featured[self.full_feature_list])
        
        X_train_3D, y_train_3D_raw = convert_data_to_sliding_window(
            train_scaled,
            lag_horizon=self.config["lags"],
            forecast_horizon=self.config["horizon"],
            shift=1
        )
        
        y_train_3D = y_train_3D_raw
        if self.config.get("scale_target", False):
            # Transformieren mit dem bereits angepassten y_scaler
            # Form für den Scaler anpassen -> (N, 1)
            y_train_3D_reshaped = y_train_3D_raw.reshape(-1, 1)
            # Skalieren
            y_train_3D_scaled = self.y_scaler.transform(y_train_3D_reshaped)
            # Zurück in die ursprüngliche Form (Anzahl_Fenster, forecast_horizon)
            y_train_3D = y_train_3D_scaled.reshape(y_train_3D_raw.shape)

        print(f"\nTrainings-Pipeline abgeschlossen. Shapes: X_train: {X_train_3D.shape}, y_train: {y_train_3D.shape}")
        return X_train_3D, y_train_3D

    def prepare_testing_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Führt die komplette Pipeline für die (Batch-)Testdaten aus.
        """
        if not self.scaler or not self.full_feature_list:
            raise RuntimeError("Die Methode `prepare_training_data` muss zuerst aufgerufen werden.")
        
        print("\n--- Starte 3D Batch-Testing-Pipeline ---")

        _, test_df = self._load_data(mode='test')
        
        # Feature Engineering & NaN-Entfernung
        test_df_featured, _ = fe.add_all_features(test_df, self.config)
        test_df_featured.dropna(inplace=True)
        
        # Skalierung mit trainiertem Scaler
        test_scaled = self.scaler.transform(test_df_featured[self.full_feature_list])

        # 3D-Fenster erstellen
        X_test_3D, y_test_3D_raw = convert_data_to_sliding_window(
            test_scaled,
            lag_horizon=self.config["lags"],
            forecast_horizon=self.config["horizon"],
            shift=1
        )
        
        # Target skalieren, falls konfiguriert
        y_test_3D = y_test_3D_raw
        if self.config.get("scale_target", False):
            y_test_3D = self.y_scaler.transform(y_test_3D_raw)

        print(f"\nTest-Pipeline abgeschlossen. Shapes: X_test: {X_test_3D.shape}, y_test: {y_test_3D.shape}")
        return X_test_3D, y_test_3D