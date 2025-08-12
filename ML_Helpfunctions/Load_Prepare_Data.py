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


def create_multi_step_target(y, horizon):
    """Convert 1D y into 2D array with horizon columns"""
    y = np.asarray(y)
    return np.column_stack([y[i:i-horizon or None] for i in range(horizon)])


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

    # Diese Methode bleibt, da sie für die Skalierung von X und y zuständig ist.
    def _prepare_and_scale_training_data(self, X_train_df: pd.DataFrame, y_train_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Initialisiert und trainiert die Scaler auf den Trainingsdaten und gibt die skalierten Arrays zurück.
        Nimmt nun X und y als separate DataFrames an.
        """
        scaler_type = self.config.get("scaler_type", "minmax")
        scale_target = self.config.get("scale_target", False) 
        scale_features = self.config.get("scale_other_features", True)

        scaler_class = RobustScaler if scaler_type == "robust" else MinMaxScaler

        # 1. Feature-Skalierung (X)
        X_train_values = X_train_df.values
        if scale_features:
            print("Passe Feature-Scaler (scaler) auf Trainingsdaten an...")
            self.scaler = scaler_class()
            X_train_values = self.scaler.fit_transform(X_train_values)
        
        # 2. Target-Skalierung (y)
        y_train_values = y_train_df.values
        if scale_target:
            print("Passe Target-Scaler (y_scaler) auf Trainingsdaten an...")
            self.y_scaler = scaler_class()
            y_train_values = self.y_scaler.fit_transform(y_train_values)
            
        return X_train_values, y_train_values

    def prepare_training_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Führt die komplette Pipeline für die Trainingsdaten aus.
        *** FINALE KORRIGIERTE VERSION: Richtet X(t) korrekt auf Y(t+1...t+h) aus. ***
        """
        print("--- Starte 2D Trainings-Pipeline (Finale Prognose-Version) ---")
        
        # Schritt 1 & 2: Laden und Feature Engineering
        train_df, _ = self._load_data(mode='train')
        train_df_featured, self.train_features_dict = fe.add_all_features(train_df, self.config)
        self.full_feature_list = self.train_features_dict["all"]
        
        # Schritt 3: Erstelle X und Y explizit für die korrekte Zeit-Ausrichtung
        print("\nSchritt 3: Richte Features (X_t) auf zukünftige Zielvariable (Y_{t+1...t+h}) aus...")
        target_column = self.config["base_features"][0].lower()
        horizon = self.config.get("horizon", 1)

        # X sind die Features zum Zeitpunkt t
        X_df = train_df_featured[self.full_feature_list]
        
        # Y sind die Zielwerte von t+1 bis t+h. Wir erstellen für jeden Schritt eine Spalte.
        y_targets = {}
        for i in range(1, horizon + 1):
            y_targets[f'y_target_t_plus_{i}'] = train_df_featured[target_column].shift(-i)
        
        Y_df = pd.DataFrame(y_targets)
        
        # Kombiniere X und Y und entferne alle Zeilen, wo entweder in X oder in Y ein NaN ist.
        # NaNs in X stammen vom Anfang (durch Lags/Rolling).
        # NaNs in Y stammen vom Ende (durch das Shiften in die Zukunft).
        full_aligned_df = pd.concat([X_df, Y_df], axis=1)
        full_aligned_df.dropna(inplace=True)
        
        X_train_aligned = full_aligned_df[self.full_feature_list]
        Y_train_aligned = full_aligned_df[Y_df.columns]
        print(f"Nach Ausrichtung und NaN-Filterung verbleiben {len(X_train_aligned)} Trainingspunkte.")

        # Schritt 4: Skalierung auf den ausgerichteten und bereinigten Daten
        print("\nSchritt 4: Skalierung und finale Formatierung...")
        X_train_final, y_train_final = self._prepare_and_scale_training_data(
            X_train_df=X_train_aligned,
            y_train_df=Y_train_aligned
        )

        print(f"\nTrainings-Pipeline abgeschlossen. Shapes: X_train: {X_train_final.shape}, y_train: {y_train_final.shape}")
        return X_train_final, y_train_final


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
        *** KORRIGIERTE VERSION: Stellt sicher, dass X die Historie des Targets enthält. ***
        """
        print("--- Starte 3D Trainings-Pipeline (Korrigierte Version) ---")
        
        # Schritt 1 & 2: Daten laden und Feature Engineering
        train_df, _ = self._load_data(mode='train') 
        train_df_featured, self.train_features_dict = fe.add_all_features(train_df, self.config)
        self.full_feature_list = self.train_features_dict["all"]
        
        # KORREKTUR: Stelle sicher, dass die Zielvariable an erster Stelle der Feature-Liste steht.
        # Dies vereinfacht die Fenstererstellung und die spätere inverse Skalierung.
        target_col_lower = self.config["base_features"][0].lower()
        if self.full_feature_list[0] != target_col_lower:
            logging.info(f"Ordne Features neu an, um Target '{target_col_lower}' an Position 0 zu platzieren.")
            self.full_feature_list.remove(target_col_lower)
            self.full_feature_list.insert(0, target_col_lower)
        
        # Ordne den DataFrame entsprechend der neuen Feature-Reihenfolge neu an
        train_df_featured = train_df_featured[self.full_feature_list]
        train_df_featured.dropna(inplace=True)
        print(f"Nach FE und dropna verbleiben {len(train_df_featured)} Zeilen für das Training.")
        
        # Schritt 3: Scaler anpassen
        # KORREKTUR: Wir verwenden jetzt EINEN Scaler für alle Features, da das Modell alle als Input erhält.
        # Ein separater y_scaler wird aber trotzdem für die leichtere inverse Transformation trainiert.
        print("\nSchritt 3: Scaler anpassen...")
        scaler_class = RobustScaler if self.config.get("scaler_type", "minmax") == "robust" else MinMaxScaler
        
        # Haupt-Scaler wird auf allen Daten (inkl. Target) trainiert
        self.scaler = scaler_class()
        self.scaler.fit(train_df_featured)
        print("✅ Haupt-Scaler (scaler) wurde auf ALLEN Spalten angepasst.")

        # Separater y_scaler wird NUR auf der Target-Spalte trainiert (für einfache Inferenz)
        self.y_scaler = scaler_class()
        self.y_scaler.fit(train_df_featured[[target_col_lower]])
        print("✅ Target-Scaler (y_scaler) wurde NUR auf der Target-Spalte angepasst.")

        # Schritt 4: Daten skalieren und 3D-Fenster erstellen
        print("\nSchritt 4: Gesamte Daten skalieren und 3D-Fenster erstellen...")
        
        # Skaliere den gesamten DataFrame mit dem Haupt-Scaler
        all_data_scaled = self.scaler.transform(train_df_featured)
        
        lags = self.config.get("lags", 1)
        horizon = self.config.get("horizon", 1)
        
        X_train_list, y_train_list = [], []
        
        # In DataPipeline3D.prepare_training_data (logische Darstellung)
        for i in range(len(all_data_scaled) - lags - horizon + 1):
            # Input-Fenster X: Nimm 'lags' Zeitschritte von Index i bis i+lags-1
            # Das repräsentiert die komplette Historie bis zum Zeitpunkt t = i+lags-1
            X_window = all_data_scaled[i : i + lags]
            X_train_list.append(X_window)

            # Output-Fenster y: Nimm 'horizon' Zeitschritte AB Index i+lags
            # Das repräsentiert die Zukunft von Zeitpunkt t+1 bis t+horizon
            y_window = all_data_scaled[i + lags : i + lags + horizon, 0] # Index 0 ist die Target-Spalte
            y_train_list.append(y_window)
        
        X_train_3D = np.array(X_train_list)
        # y_train ist eine Liste von 1D-Arrays der Länge 'horizon', wir formen es zu (samples, horizon)
        y_train_3D = np.array(y_train_list)

        print(f"\nTrainings-Pipeline abgeschlossen. Shapes: X_train: {X_train_3D.shape}, y_train: {y_train_3D.shape}")
        return X_train_3D, y_train_3D

    def prepare_testing_data(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Führt die komplette Pipeline für die (Batch-)Testdaten aus.
        *** KORRIGIERTE VERSION ZUR VERMEIDUNG VON DOPPEL-SKALIERUNG ***
        """
        if not self.scaler or not self.y_scaler or not self.full_feature_list:
            raise RuntimeError("Die Methode `prepare_training_data` muss zuerst aufgerufen werden, um Scaler zu initialisieren.")
        
        print("\n--- Starte 3D Batch-Testing-Pipeline (Korrigierte Version) ---")

        _, test_df = self._load_data(mode='test')
        if test_df is None or test_df.empty:
            logging.warning("Keine Testdaten zum Verarbeiten vorhanden.")
            return np.array([]), np.array([])
        
        # Feature Engineering & NaN-Entfernung
        test_df_featured, _ = fe.add_all_features(test_df, self.config)
        test_df_featured.dropna(inplace=True)
        
        # KORREKTUR: Bereite X und y getrennt und konsistent zum Training vor
        
        # 1. Definiere Feature- und Target-Spalten
        target_col = self.config["base_features"][0].lower()
        feature_cols = [col for col in self.full_feature_list if col != target_col]
        
        # 2. Skaliere Features und Target getrennt mit den trainierten Scalern
        test_features_scaled = self.scaler.transform(test_df_featured[feature_cols])
        test_target_scaled = self.y_scaler.transform(test_df_featured[[target_col]])

        # 3. Kombiniere sie wieder für die einheitliche Windowing-Funktion
        # (Target muss für die Funktion an Index 0 stehen)
        combined_scaled_data = np.hstack([test_target_scaled, test_features_scaled])

        # 4. Erstelle 3D-Fenster. `y_test_3D` wird aus der bereits korrekt skalierten
        #    Target-Spalte des `combined_scaled_data`-Arrays extrahiert.
        X_test_3D, y_test_3D = convert_data_to_sliding_window(
            combined_scaled_data,
            lag_horizon=self.config["lags"],
            forecast_horizon=self.config["horizon"],
            shift=1
        )
        
        print(f"\nTest-Pipeline abgeschlossen. Shapes: X_test: {X_test_3D.shape}, y_test: {y_test_3D.shape}")
        return X_test_3D, y_test_3D