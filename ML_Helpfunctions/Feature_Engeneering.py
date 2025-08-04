import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def add_time_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    selected_features = config.get("time_features", [])
    feature_dict = {"time": []}

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be a DatetimeIndex.")

    if "second" in selected_features:
        df["second"] = df.index.second
        feature_dict["time"].append("second")
    if "minute" in selected_features:
        df["minute"] = df.index.minute
        feature_dict["time"].append("minute")
    if "hour" in selected_features:
        df["hour"] = df.index.hour
        feature_dict["time"].append("hour")
    if "day_of_month" in selected_features:
        df["day_of_month"] = df.index.day
        feature_dict["time"].append("day_of_month")
    if "day_of_week" in selected_features:
        df["day_of_week"] = df.index.dayofweek
        feature_dict["time"].append("day_of_week")
    if "is_weekend" in selected_features:
        df["is_weekend"] = df.index.dayofweek.isin([5, 6]).astype(int)
        feature_dict["time"].append("is_weekend")
    if "month" in selected_features:
        df["month"] = df.index.month
        feature_dict["time"].append("month")

    # Zyklische Transformationen
    if "minute_sin" in selected_features:
        df["minute_sin"] = np.sin(2 * np.pi * df.index.minute / 60)
        feature_dict["time"].append("minute_sin")
    if "minute_cos" in selected_features:
        df["minute_cos"] = np.cos(2 * np.pi * df.index.minute / 60)
        feature_dict["time"].append("minute_cos")
    if "hour_sin" in selected_features:
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        feature_dict["time"].append("hour_sin")
    if "hour_cos" in selected_features:
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        feature_dict["time"].append("hour_cos")
    if "month_sin" in selected_features:        
        df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
        feature_dict["time"].append("month_sin")
    if "month_cos" in selected_features:
        df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
        feature_dict["time"].append("month_cos")
    if "dayofweek_sin" in selected_features:
        df["dayofweek_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        feature_dict["time"].append("dayofweek_sin")
    if "dayofweek_cos" in selected_features:
        df["dayofweek_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        feature_dict["time"].append("dayofweek_cos")

    return df, feature_dict


def add_lag_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    base_features = config["base_features"]
    max_lag = config["lags"]
    feature_dict = {"lagged": []}

    for feature in base_features:
        for lag in range(1, max_lag + 1):
            lagged_name = f'{feature}_lag_{lag}'
            df[lagged_name] = df[feature].shift(lag)
            feature_dict["lagged"].append(lagged_name)

    return df, feature_dict


def add_rolling_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    base_features = config["base_features"]
    window_size = config.get("rolling_window_size", 1)
    include_roll_mean = config.get("include_roll_mean", False)
    include_roll_std = config.get("include_roll_std", False)

    feature_dict = {"rolling": []}

    for feature in base_features:
        if include_roll_mean:
            mean_name = f'{feature}_roll_mean_{window_size}'
            df[mean_name] = df[feature].rolling(window=window_size).mean() # OHNE .shift(1)
            feature_dict["rolling"].append(mean_name)

        if include_roll_std:
            if window_size >= 2:
                std_name = f'{feature}_roll_std_{window_size}'
                df[std_name] = df[feature].rolling(window=window_size).std() # OHNE .shift(1)
                feature_dict["rolling"].append(std_name)
            else:
                print(f"[WARNUNG] 'rolling_window_size' ist {window_size}. "
                      f"Rollierende Standardabweichung wird übersprungen, da Fenstergröße >= 2 sein muss.")

    return df, feature_dict


def add_all_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    Führt alle Schritte des Feature Engineerings aus.
    MODIFIZIERTE DEBUG-VERSION.
    """
    feature_dict = {
        "base": config["base_features"].copy(),
        "lagged": [],
        "rolling": [],
        "time": [],
    }

    # Schritt 1: Zeit-Features (falls konfiguriert)
    df, time_dict = add_time_features(df, config)
    feature_dict["time"] = time_dict["time"]

    # Schritt 2: Lag-Features
    df, lag_dict = add_lag_features(df, config)
    feature_dict["lagged"] = lag_dict["lagged"]

    # Schritt 3: Rolling-Features
    df, roll_dict = add_rolling_features(df, config)
    feature_dict["rolling"] = roll_dict["rolling"]

    # --- WICHTIGER DEBUG-SCHRITT ---
    # Wir kommentieren den dropna()-Befehl aus, um zu sehen, ob die Daten
    # mit NaN-Werten an die nächste Funktion weitergegeben werden.
    # df = df.dropna()
    # --- ENDE DEBUG-SCHRITT ---


    # Alle Features zusammenführen
    all_features = (
        feature_dict["base"]
        + feature_dict["lagged"]
        + feature_dict["rolling"]
        + feature_dict["time"]
    )
    feature_dict["all"] = all_features

    # Debug-Ausgabe, um den Zustand des DataFrames zu prüfen
    print(f"[DEBUG in add_all_features] DataFrame hat {len(df)} Zeilen VOR dem Verlassen der Funktion.")
    print(f"[DEBUG in add_all_features] Anzahl der NaN-Werte pro Spalte:\n{df.isnull().sum()}")

    return df, feature_dict


def create_feature_list_from_dict(feature_dict: dict) -> list:
    return feature_dict["all"]
