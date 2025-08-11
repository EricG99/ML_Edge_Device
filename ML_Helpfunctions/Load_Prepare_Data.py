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
    Gibt den vollständigen Pfad zur Zieldatendatei zurück.
    """
    try:
        data_dir = config["paths"]["input_data"] 
        filename = config["dataset"]
    except KeyError as e:
        raise KeyError(f"Fehlender oder falscher Schlüssel in der Konfiguration: {e}. "
                     f"Stellen Sie sicher, dass config['paths']['Input_Data'] und config['dataset'] existieren.")

    file_path = Path(data_dir) / filename

    if not file_path.exists():
        raise FileNotFoundError(f"Die angegebene Datendatei wurde nicht gefunden unter: {file_path}")
        
    print(f"✔️ Datendatei gefunden: {file_path}")
    return file_path

# ---------------------------------------------------
# LADEN TRAIN/TEST
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


def _load_full_timeseries(config: dict,
                          make_date_as_index: bool = True) -> pd.DataFrame:
    file_path = _get_file_path(config)
    
    # KORREKTUR: Robustes Laden mit dynamischem Separator
    try:
        df = pd.read_csv(file_path, sep=None, engine='python')
        logging.info(f"CSV-Datei '{file_path}' erfolgreich mit autom. Separator-Erkennung geladen.")
    except Exception as e:
        raise IOError(f"Fehler beim Lesen der CSV-Datei '{file_path}': {e}")

    df.columns = df.columns.str.lower()
    
    # Zeitstempel-Logik
    if "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["time"], unit='ms')
    elif "datetime" in df.columns:
         df["datetime"] = pd.to_datetime(df["datetime"])
    else:
        raise ValueError(f"Die CSV-Datei '{file_path}' muss eine 'time' (ms) oder 'datetime'-Spalte enthalten.")
    
    df = df.sort_values("datetime")

    if make_date_as_index:
        df = df.set_index("datetime")
    
    print(f"Geladen: {len(df)} Zeilen aus '{file_path}'")
    return df


# ---------------------------------------------------
# SLIDING WINDOWS
# ---------------------------------------------------
def convert_data_to_sliding_window(data_array: np.ndarray,
                                   lag_horizon: int,
                                   forecast_horizon: int = 1,
                                   shift: int = 1):
    X, y = [], []
    for i in range(0, len(data_array) - lag_horizon - forecast_horizon + 1, shift):
        X.append(data_array[i:i + lag_horizon])
        # y ist immer der Wert der ERSTEN Spalte (Zielvariable)
        y.append(data_array[i + lag_horizon:i + lag_horizon + forecast_horizon, 0])
    return np.array(X), np.array(y)


def create_multi_step_target(y, horizon):
    """Convert 1D y into 2D array with horizon columns"""
    y = np.asarray(y)
    return np.column_stack([y[i:i-horizon or None] for i in range(horizon)])


# =============================================================================
# DATENPIPELINE BASISKLASSE
# =============================================================================

class DataPipelineBase:
    """
    Abstrakte Basisklasse, die die gemeinsame Logik für Datenpipelines kapselt.
    """
    def __init__(self, config: dict):
        self.config = config
        self.scaler = None
        self.y_scaler = None # Wird nur noch für optionale, separate Target-Skalierung benötigt
        self.full_feature_list = None
        self.train_features_dict = None
        self._live_data_buffer = pd.DataFrame()

    def _load_data(self, mode: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
        strategy = self.config.get("loading_strategy", "split")
        print(f"\nLade Daten im Modus '{mode}' mit Strategie '{strategy}'...")
        if strategy == "split":            
            train_df = load_train_data_by_fraction(config=self.config, make_date_as_index=True)
            test_df = load_test_data_by_fraction(config=self.config, make_date_as_index=True)
            return train_df, test_df
        else:
            raise ValueError(f"Unbekannte Lade-Strategie: {strategy}")

    def prepare_training_data(self):
        raise NotImplementedError

# =============================================================================
# DATENPIPELINE 2D (FÜR RANDOM FOREST) - *** KORRIGIERTE VERSION ***
# =============================================================================
class DataPipeline2D(DataPipelineBase):
    """
    Datenvorbereitung für 2D-Modelle wie Random Forest.
    KORRIGIERT: Stellt sicher, dass der Skalierer nur auf X-Features trainiert wird
    und das Modell auf unskalierten y-Werten lernt.
    """
    def prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        print("--- Starte 2D Trainings-Pipeline (RF-Style) ---")
        
        # Schritt 1: Lade Trainingsdaten
        train_df, _ = self._load_data(mode='train')

        # Schritt 2: Feature Engineering
        print("\nSchritt 2: Feature Engineering...")
        train_df_featured, self.train_features_dict = fe.add_all_features(train_df, self.config)
        
        # WICHTIG: NaNs entfernen, die durch Lags/Rolling-Features entstehen
        original_rows = len(train_df_featured)
        train_df_featured.dropna(inplace=True)
        print(f"Entferne Zeilen mit NaNs nach FE. Shape: {original_rows} -> {len(train_df_featured)}")

        # Schritt 3: Daten in X und y aufteilen
        target_column = self.config["base_features"][0]
        
        # Die Feature-Liste (X) enthält NICHT die Zielvariable selbst.
        x_features = [col for col in self.train_features_dict["all"] if col != target_column]
        self.full_feature_list = x_features # Dies wird als Artefakt gespeichert!
        
        X_train_df = train_df_featured[self.full_feature_list]
        y_train_series = train_df_featured[target_column]

        # Schritt 4: Skalierer NUR auf den X-Features anpassen
        print(f"\nSchritt 3: Skaliere {len(self.full_feature_list)} Features (X). Der Zielwert (y) bleibt unskaliert.")
        scaler_class = RobustScaler if self.config.get("scaler_type", "minmax") == "robust" else MinMaxScaler
        self.scaler = scaler_class()
        X_train_scaled = self.scaler.fit_transform(X_train_df)
        
        # Der Zielvektor y bleibt unskaliert. Das Modell lernt die echten Werte.
        y_train = y_train_series.values

        # Optional: Multi-Step-Target erstellen
        if self.config.get("horizon", 1) > 1:
            print(f"\nSchritt 4: Erstelle Multi-Step Target für Horizont={self.config['horizon']}...")
            y_train = create_multi_step_target(y_train, self.config["horizon"])
            # X an die neue Länge von y anpassen
            X_train_scaled = X_train_scaled[:len(y_train)]

        print(f"\nTrainings-Pipeline abgeschlossen. Shapes: X_train: {X_train_scaled.shape}, y_train: {y_train.shape}")
        return X_train_scaled, y_train

# =============================================================================
# DATENPIPELINE 3D (FÜR LSTM) - (Logik war bereits korrekt)
# =============================================================================
class DataPipeline3D(DataPipelineBase):
    """
    Datenvorbereitung für 3D-Modelle wie LSTM.
    Der Skalierer wird hier auf ALLEN Features (inkl. Zielvariable) angepasst.
    """
    def prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        print("--- Starte 3D Trainings-Pipeline (LSTM-Style) ---")
        
        # Schritt 1: Lade Daten
        train_df, _ = self._load_data(mode='train') 

        # Schritt 2: Feature Engineering
        print("\nSchritt 2: Feature Engineering...")
        train_df_featured, self.train_features_dict = fe.add_all_features(train_df, self.config)
        self.full_feature_list = self.train_features_dict["all"]
        print(f"Verfügbare Features: {self.full_feature_list}")

        # WICHTIG: NaN-Zeilen entfernen
        train_df_featured.dropna(inplace=True)
        print(f"Nach FE und dropna verbleiben {len(train_df_featured)} Zeilen für das Training.")
        
        # Schritt 3: Skalierer auf ALLEN Features anpassen (inkl. Zielvariable)
        print("\nSchritt 3: Scaler anpassen (auf allen Features)...")
        scaler_class = RobustScaler if self.config.get("scaler_type", "minmax") == "robust" else MinMaxScaler
        self.scaler = scaler_class()
        # self.full_feature_list enthält die Zielvariable an erster Stelle
        scaled_data = self.scaler.fit_transform(train_df_featured[self.full_feature_list])
        print("Feature-Scaler wurde auf dem gesamten Trainingsdatensatz (inkl. Zielvariable) angepasst.")
        
        # Schritt 4: Skalierte Daten in Sliding Windows umwandeln
        print("\nSchritt 4: Skalierung und 3D-Windowing für Trainingsdaten...")
        X_train_3D, y_train_3D = convert_data_to_sliding_window(
            scaled_data,
            lag_horizon=self.config["lags"],
            forecast_horizon=self.config["horizon"],
            shift=1
        )
        
        # y_train_3D enthält bereits die SKALIERTEN Zielwerte aus Spalte 0
        print(f"\nTrainings-Pipeline abgeschlossen. Shapes: X_train: {X_train_3D.shape}, y_train: {y_train_3D.shape}")
        return X_train_3D, y_train_3D