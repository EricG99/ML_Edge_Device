import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from pathlib import Path
import sys

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
    Gibt den vollständigen Pfad zur Zieldatendatei zurück.
    
    Diese Funktion sucht im "input_data"-Verzeichnis (definiert in der Konfiguration)
    nach dem Dateinamen, der unter dem Schlüssel "dataset" in der Konfiguration
    gespeichert ist.

    Args:
        config (dict): Konfigurationsdictionary, das unter "paths" den Pfad 
                       "input_data" und den Dateinamen unter "dataset" enthält.

    Returns:
        Path: Ein Path-Objekt, das auf die existierende Datendatei verweist.

    Raises:
        KeyError: Wenn notwendige Schlüssel in der Konfiguration fehlen.
        FileNotFoundError: Wenn die zusammengesetzte Datei nicht gefunden wird.
    """
    try:
        # === KORREKTUR: Hier auf den spezifischen "input_data"-Pfad verweisen ===
        data_dir = config["paths"]["input_data"] 
        filename = config["dataset"]
    except KeyError as e:
        raise KeyError(f"Fehlender oder falscher Schlüssel in der Konfiguration: {e}. "
                     f"Stellen Sie sicher, dass config['paths']['Input_Data'] und config['dataset'] existieren.")

    # Stelle sicher, dass data_dir ein Path-Objekt ist (falls es als String geladen wurde)
    file_path = Path(data_dir) / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Die angegebene Datendatei wurde nicht gefunden unter: {file_path}")
        
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

def create_timeseries_validation_split(X_train, y_train, config):
    """
    Teilt Trainingsdaten chronologisch in ein Trainings- und ein Validierungsset.
    Die letzten Datenpunkte werden für die Validierung verwendet.

    Args:
        X_train (np.ndarray): Die vollständigen Trainingsdaten (Input).
        y_train (np.ndarray): Die vollständigen Trainingsdaten (Zielwerte).
        config (dict): Konfigurationsdictionary, das 'validation_fraction' enthält.

    Returns:
        tuple: (X_fit, y_fit, X_val, y_val)
               Gibt die aufgeteilten Daten zurück. Wenn keine Validierung stattfindet,
               sind X_val und y_val None.
    """
    val_fraction = config.get("validation_fraction", 0.0)

    # Nur splitten, wenn eine valide Fraktion angegeben ist und genügend Daten vorhanden sind.
    if val_fraction > 0 and X_train.shape[0] > 10:
        print(f"Erstelle chronologischen Validierungs-Split. Validation Fraction: {val_fraction}")
        split_index = int((1 - val_fraction) * len(X_train))
        
        X_fit = X_train[:split_index]
        y_fit = y_train[:split_index]
        
        X_val = X_train[split_index:]
        y_val = y_train[split_index:]
        
        return X_fit, y_fit, X_val, y_val
    else:
        # Wenn keine Validierung stattfinden soll, gib die Originaldaten und None zurück.
        print("Kein Validierungs-Split durchgeführt.")
        return X_train, y_train, None, None
    

class DataPipeline2D:
    """
    Eine Klasse zur Kapselung des gesamten Datenvorbereitungsprozesses für 2D-Modelle.
    Sie verwaltet den Zustand (Scaler, Feature-Listen) zwischen Trainings- und
    Test-Pipelines und unterstützt verschiedene Datenlade-Strategien.
    """
    def __init__(self, config: dict):
        """
        Initialisiert die Pipeline mit der Konfiguration.

        Args:
            config (dict): Ein Konfigurationswörterbuch, das alle Einstellungen enthält.
                           Erwartet einen Schlüssel 'loading_strategy' ("split", "separate_csv", "live_mqtt").
        """
        self.config = config
        
        # Zustandsobjekte, die während des Trainings initialisiert werden
        self.scaler_2D = None
        self.y_scaler = None
        self.full_feature_list = None
        self.train_features_dict = None
        
        # Puffer für Live-Daten (z.B. für MQTT)
        # Hält die letzten N Datenpunkte, um Lags und rollierende Features zu berechnen
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
            # Strategie 1: Lade ein Dataset und teile es
            train_df = load_train_data_by_fraction(config=self.config, make_date_as_index=True)
            test_df = load_test_data_by_fraction(config=self.config, make_date_as_index=True)
            return train_df, test_df

        elif strategy == "separate_csv":
            # Strategie 2: Lade aus getrennten CSV-Dateien
            if mode == 'train':
                train_df = pd.read_csv(self.config['train_csv_path'], index_col='datetime', parse_dates=True)
                return train_df, None
            else: # mode == 'test'
                test_df = pd.read_csv(self.config['test_csv_path'], index_col='datetime', parse_dates=True)
                return None, test_df
        
        elif strategy == "live_mqtt":
             # Strategie 3: Trainingsdaten aus CSV, Testdaten kommen live
             if mode == 'train':
                train_df = pd.read_csv(self.config['train_csv_path'], index_col='datetime', parse_dates=True)
                return train_df, None
             else: # mode == 'test'
                # Bei MQTT werden die Daten nicht als Block geladen.
                # Dies wird in der `prepare_live_data_point` Methode behandelt.
                print("Im MQTT-Modus werden Testdaten live verarbeitet, kein initiales Laden.")
                return None, None
        
        else:
            raise ValueError(f"Unbekannte Lade-Strategie: {strategy}")

    def prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Führt die komplette Pipeline für die Trainingsdaten aus.
        1. Lädt Daten.
        2. Führt Feature Engineering durch.
        3. Initialisiert und trainiert die Scaler.
        4. Speichert die Zustandsobjekte (Scaler, Feature-Listen).
        5. Gibt die vorbereiteten 2D-Trainingsdaten zurück.
        """
        print("--- Starte Trainings-Pipeline ---")
        
        # Schritt 1: Daten laden
        train_df, _ = self._load_data(mode='train') # Testdaten werden hier nicht benötigt

        # Schritt 2: Feature Engineering
        print("\nSchritt 2: Feature Engineering...")
        train_df, self.train_features_dict = fe.add_all_features(train_df, self.config)
        self.full_feature_list = self.train_features_dict["all"]
        print(f"Trainingsdaten nach FE (Shape): {train_df.shape}")
        print(f"Verfügbare Features: {self.full_feature_list}")

        # Schritt 3: Daten in 2D-Format bringen und skalieren
        print("\nSchritt 3: Skalierung und 2D-Formatierung...")
        # Die Funktion `prepare_2d_train_data` muss leicht angepasst werden,
        # um die Scaler zurückzugeben, damit wir sie speichern können.
        X_train_2D, y_train_2D, self.scaler_2D, self.y_scaler = prepare_2d_train_data(
            full_feature_train_data=train_df[self.full_feature_list],
            used_lags=self.config["lags"],
            scale_target=self.config.get("scale_target", False),
            scaler_type=self.config.get("scaler_type", "minmax"),
            scale_features=self.config.get("scale_other_features", False)
        )

        # Schritt 4: Multi-Step Target (optional)
        if self.config["horizon"] > 1:
            print("\nSchritt 4: Erstelle Multi-Step Target...")
            y_train_2D = create_multi_step_target(y_train_2D, self.config["horizon"])
            # Schneide X passend zur neuen Länge von y
            X_train_2D = X_train_2D[:len(y_train_2D)]

        print(f"\nTrainings-Pipeline abgeschlossen. Shapes: X_train: {X_train_2D.shape}, y_train: {y_train_2D.shape}")
        return X_train_2D, y_train_2D

    def prepare_testing_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Führt die komplette Pipeline für die (Batch-)Testdaten aus.
        1. Lädt Testdaten.
        2. Führt Feature Engineering durch.
        3. Verwendet die existierenden, trainierten Scaler zur Transformation.
        4. Gibt die vorbereiteten 2D-Testdaten zurück.
        
        HINWEIS: Diese Methode ist für Batch-Verarbeitung gedacht (aus CSV oder Split).
                 Für Live-Daten siehe `prepare_live_data_point`.
        """
        if self.scaler_2D is None or self.full_feature_list is None:
            raise RuntimeError("Die Methode `prepare_training_data` muss zuerst aufgerufen werden.")
        
        print("\n--- Starte Batch-Testing-Pipeline ---")

        # Schritt 1: Daten laden
        _, test_df = self._load_data(mode='test')
        if test_df is None:
            raise ValueError("Keine Testdaten für Batch-Verarbeitung geladen. Falsche Strategie?")

        # Schritt 2: Feature Engineering
        # Wichtig: dieselben Features wie im Training erstellen
        print("\nSchritt 2: Feature Engineering für Testdaten...")
        test_df, _ = fe.add_all_features(test_df, self.config)
        
        # Sicherstellen, dass die Spaltenreihenfolge exakt übereinstimmt
        test_df = test_df[self.full_feature_list]

        # Schritt 3: Daten transformieren (ohne erneutes Fitten!)
        print("\nSchritt 3: Transformierung und 2D-Formatierung...")
        X_test_2D, y_test_2D = prepare_test_data_2D(
            full_feature_test_data=test_df,
            scaler_2D=self.scaler_2D,
            scaler_y=self.y_scaler,
            used_lags=self.config["lags"],
            scale_target=self.config.get("scale_target", False),
            target_column=self.config["base_features"][0]
        )

        # Schritt 4: Multi-Step Target (optional)
        if self.config["horizon"] > 1:
            print("\nSchritt 4: Erstelle Multi-Step Target...")
            y_test_2D = create_multi_step_target(y_test_2D, self.config["horizon"])
            X_test_2D = X_test_2D[:len(y_test_2D)]

        print(f"\nTest-Pipeline abgeschlossen. Shapes: X_test: {X_test_2D.shape}, y_test: {y_test_2D.shape}")
        return X_test_2D, y_test_2D
        
    def prepare_live_data_point(self, mqtt_payload: dict) -> np.ndarray | None:
        """
        Verarbeitet einen einzelnen, live eintreffenden Datenpunkt aus einer MQTT-Payload.
        Extrahiert die benötigten Basis-Features, fügt den Datenpunkt dem internen
        Puffer hinzu und führt die vollständige Vorverarbeitung für die Inferenz durch.

        Args:
            mqtt_payload (dict): Das vollständige Dictionary, das aus dem MQTT JSON-Payload
                                 geparst wurde.

        Returns:
            Ein fertig vorbereiteter 2D-Vektor (np.ndarray) für die Inferenz, oder None,
            wenn der interne Puffer noch nicht genügend Daten hat.
        """
        if self.scaler_2D is None or self.full_feature_list is None:
            raise RuntimeError("Die Methode `prepare_training_data` muss zuerst aufgerufen werden, um die Pipeline zu initialisieren.")

        base_features_needed = self.train_features_dict.get("base", [])
        
        # 'datetime' wird als Index benötigt
        if "datetime" not in mqtt_payload:
            print("Fehler: 'datetime' fehlt in der MQTT-Nachricht. Überspringe Datenpunkt.")
            return None
            
        new_data_point = {"datetime": mqtt_payload["datetime"]}

        for feature in base_features_needed:
            if feature in mqtt_payload:
                new_data_point[feature] = mqtt_payload[feature]
            else:
                # Falls ein benötigtes Feature fehlt, kann der Punkt nicht verarbeitet werden
                print(f"Warnung: Benötigtes Basis-Feature '{feature}' nicht in MQTT-Nachricht gefunden. Überspringe.")
                return None
        
        # Neuen Datenpunkt in einen DataFrame konvertieren und an den Puffer anhängen
        # pd.to_datetime wandelt den String in ein echtes Datumsobjekt um
        new_row = pd.DataFrame([new_data_point])
        new_row['datetime'] = pd.to_datetime(new_row['datetime'])
        new_row = new_row.set_index('datetime')
        
        # Alte Daten im Puffer löschen, die den gleichen Zeitstempel haben könnten
        self._live_data_buffer = self._live_data_buffer[~self._live_data_buffer.index.isin(new_row.index)]
        
        # Neuen Datenpunkt hinzufügen und Puffer sortieren
        self._live_data_buffer = pd.concat([self._live_data_buffer, new_row]).sort_index()

        # Puffer auf eine maximale Länge begrenzen
        max_history_needed = self.config.get('max_fe_window', 50) + self.config['lags']
        if len(self._live_data_buffer) > max_history_needed:
            self._live_data_buffer = self._live_data_buffer.iloc[-max_history_needed:]

        # Prüfen, ob genügend Daten für eine vollständige FE-Berechnung vorhanden sind
        # Diese Logik geht davon aus, dass `add_all_features` mit einem Puffer umgehen kann
        required_len = self.config.get('min_fe_window', 10) # Mindestlänge für rollierende Features
        if len(self._live_data_buffer) < required_len:
            print(f"Puffer füllt sich... Aktuelle Größe: {len(self._live_data_buffer)}/{required_len}")
            return None

        # Feature Engineering auf dem gesamten Puffer durchführen
        featured_buffer, _ = fe.add_all_features(self._live_data_buffer.copy(), self.config)

        # Prüfen, ob nach dem FE genügend Zeilen für die Lag-Features vorhanden sind
        if len(featured_buffer) < self.config['lags'] + 1:
             print("Puffer füllt sich für Lag-Features...")
             return None

        # Den letzten, vollständig berechneten Vektor extrahieren und sicherstellen,
        # dass alle Spalten vorhanden sind.
        try:
            last_vector_full = featured_buffer[self.full_feature_list].iloc[-1:]
        except KeyError as e:
            print(f"Fehler: Eine oder mehrere Spalten aus `full_feature_list` wurden nach dem FE nicht gefunden: {e}")
            return None

        # Transformieren mit dem trainierten Scaler
        X_live_2D = self.scaler_2D.transform(last_vector_full.values)

        return X_live_2D
    
    def _prepare_and_scale_training_data(self, train_df_featured: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Bereitet die Trainingsdaten für das Modell vor, nachdem Feature Engineering stattgefunden hat.
        - Initialisiert und trainiert die Scaler.
        - Speichert die Scaler in der Klasseninstanz.
        - Gibt die skalierten X_train und y_train Arrays zurück.
        """
        scaler_type = self.config.get("scaler_type", "minmax")
        scale_target = self.config.get("scale_target", False)
        scale_features = self.config.get("scale_other_features", False)
        target_column = self.config["base_features"][0] # Annahme: Das erste Basis-Feature ist das Target

        scaler_class = RobustScaler if scaler_type == "robust" else MinMaxScaler

        # 1. Feature-Skalierung (X)
        X_train = train_df_featured.values
        if scale_features:
            print("Passe Feature-Scaler (scaler_2D) auf Trainingsdaten an...")
            self.scaler_2D = scaler_class()
            # WICHTIG: .fit_transform() NUR auf den Trainingsdaten! Kein Data Leakage.
            X_train = self.scaler_2D.fit_transform(X_train)
        
        # 2. Target-Skalierung (y)
        y_train_raw = train_df_featured[target_column].values
        y_train = y_train_raw

        if scale_target:
            print("Passe Target-Scaler (y_scaler) auf Trainingsdaten an...")
            self.y_scaler = scaler_class()
            y_train = self.y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
            
        return X_train, y_train

    def _prepare_and_scale_inference_data(self, df_featured: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        NEUE INTERNE METHODE:
        Bereitet Test- oder Live-Daten für die Inferenz vor.
        - Verwendet die bereits trainierten Scaler.
        - Führt KEIN .fit() oder .fit_transform() aus.
        - Gibt die skalierten X und y Arrays zurück.
        """
        scale_target = self.config.get("scale_target", False)
        scale_features = self.config.get("scale_other_features", False)
        target_column = self.config["base_features"][0]

        # 1. Feature-Transformation (X)
        X_inference = df_featured.values
        if scale_features:
            if not self.scaler_2D:
                raise RuntimeError("Feature-Scaler (scaler_2D) wurde nicht trainiert. Führen Sie zuerst die Trainings-Pipeline aus.")
            print("Transformiere Features mit existierendem scaler_2D...")
            X_inference = self.scaler_2D.transform(X_inference)
            
        # 2. Target-Transformation (y)
        y_inference_raw = df_featured[target_column].values
        y_inference = y_inference_raw
        
        if scale_target:
            if not self.y_scaler:
                raise RuntimeError("Target-Scaler (y_scaler) wurde nicht trainiert. Führen Sie zuerst die Trainings-Pipeline aus.")
            print("Transformiere Target mit existierendem y_scaler...")
            y_inference = self.y_scaler.transform(y_inference_raw.reshape(-1, 1)).flatten()

        return X_inference, y_inference


    def prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        ÜBERARBEITETE Hauptmethode für das Training.
        Nutzt jetzt die neue interne Helper-Methode.
        """
        print("--- Starte Trainings-Pipeline ---")
        
        # Schritt 1: Lade Trainings-CSV
        train_df, _ = self._load_data(mode='train')

        # Schritt 2: Feature Engineering
        print("\nSchritt 2: Feature Engineering...")
        train_df_featured, self.train_features_dict = fe.add_all_features(train_df, self.config)
        self.full_feature_list = self.train_features_dict["all"]
        
        # Schritt 2.1: Entferne Zeilen mit NaNs, die durch Lags/Rolling-Features entstanden sind
        # Dies ist der sauberste Weg, um X und y perfekt auszurichten.
        original_rows = len(train_df_featured)
        train_df_featured.dropna(inplace=True)
        print(f"Entferne Zeilen mit NaNs nach FE. Shape: {original_rows} -> {len(train_df_featured)}")

        # Schritt 3: Skalierung und finale Vorbereitung
        print("\nSchritt 3: Skalierung und finale Formatierung...")
        X_train_2D, y_train_2D = self._prepare_and_scale_training_data(
            train_df_featured=train_df_featured[self.full_feature_list]
        )

        # Schritt 4: Multi-Step Target (optional)
        if self.config.get("horizon", 1) > 1:
            # ... Ihre Logik für create_multi_step_target ...
            pass

        print(f"\nTrainings-Pipeline abgeschlossen. Shapes: X_train: {X_train_2D.shape}, y_train: {y_train_2D.shape}")
        return X_train_2D, y_train_2D

    def prepare_testing_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        ÜBERARBEITETE Hauptmethode für das Testen mit CSV-Dateien.
        """
        if not self.full_feature_list:
            raise RuntimeError("Die Methode `prepare_training_data` muss zuerst aufgerufen werden.")
        
        print("\n--- Starte Batch-Testing-Pipeline ---")

        # Schritt 1: Lade Test-CSV
        _, test_df = self._load_data(mode='test')
        
        # Schritt 2: Feature Engineering
        test_df_featured, _ = fe.add_all_features(test_df, self.config)
        test_df_featured.dropna(inplace=True)

        # Schritt 3: Skalierung mit den trainierten Scalern
        X_test_2D, y_test_2D = self._prepare_and_scale_inference_data(
            df_featured=test_df_featured[self.full_feature_list]
        )

        print(f"\nTest-Pipeline abgeschlossen. Shapes: X_test: {X_test_2D.shape}, y_test: {y_test_2D.shape}")
        return X_test_2D, y_test_2D