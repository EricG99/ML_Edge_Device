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
    """
    Fügt rollierende Mittelwerte und Standardabweichungen für die Basis-Features hinzu.
    """
    rolling_features_dict = {"rolling_mean": [], "rolling_std": []}
    window_size = config.get('rolling_window_size', 2)

    if config.get("add_rolling_features", False):
        # --- KORREKTUR: Immer mit der Kleinbuchstaben-Version des Features arbeiten ---
        for feature in config.get("base_features", []):
            feature_lower = feature.lower() # Sicherstellen, dass der Feature-Name klein ist

            # Rolling Mean
            mean_name = f"{feature_lower}_roll_mean_{window_size}"
            df[mean_name] = df[feature_lower].rolling(window=window_size).mean()
            rolling_features_dict["rolling_mean"].append(mean_name)

            # Rolling Std
            std_name = f"{feature_lower}_roll_std_{window_size}"
            df[std_name] = df[feature_lower].rolling(window=window_size).std()
            rolling_features_dict["rolling_std"].append(std_name)

    return df, rolling_features_dict


# ----------------------------------
# Feature Engineering: Lag Features
# ----------------------------------
def add_lag_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    Fügt Lag-Features (zeitlich verschobene Werte) für die Basis-Features hinzu.
    """
    lag_features_dict = {"lags": []}
    num_lags = config.get("num_lags", 1)

    if config.get("add_lag_features", False):
        # --- KORREKTUR: Immer mit der Kleinbuchstaben-Version des Features arbeiten ---
        for feature in config.get("base_features", []):
            feature_lower = feature.lower() # Sicherstellen, dass der Feature-Name klein ist
            
            for lag in range(1, num_lags + 1):
                lagged_name = f"{feature_lower}_lag_{lag}"
                df[lagged_name] = df[feature_lower].shift(lag)
                lag_features_dict["lags"].append(lagged_name)
    
    return df, lag_features_dict


# ----------------------------------
# Hauptfunktion: Alle Features hinzufügen
# ----------------------------------
def add_all_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """
    Führt alle konfigurierten Feature-Engineering-Schritte aus.
    """
    # WICHTIG: Stellen Sie sicher, dass die Spalten des Eingabe-DataFrames klein geschrieben sind,
    # da alle folgenden Funktionen dies erwarten.
    df.columns = df.columns.str.lower()
    
    # Basis-Features (sind die Originalspalten, die wir modifizieren)
    # Wir stellen sicher, dass die Liste der base_features selbst klein geschrieben ist.
    base_features_lower = [f.lower() for f in config.get("base_features", [])]

    # Führe die einzelnen Schritte aus
    df, lag_dict = add_lag_features(df, config)
    df, rolling_dict = add_rolling_features(df, config)

    # Sammle alle erstellten Features
    all_features = (
        base_features_lower +
        lag_dict.get("lags", []) +
        rolling_dict.get("rolling_mean", []) +
        rolling_dict.get("rolling_std", [])
    )

    # Erstelle das finale Dictionary
    features_summary = {
        "base": base_features_lower,
        "lags": lag_dict.get("lags", []),
        "rolling": rolling_dict,
        "all": sorted(list(set(all_features))) # Eindeutige und sortierte Liste aller Features
    }

    return df, features_summary


def create_feature_list_from_dict(feature_dict: dict) -> list:
    return feature_dict["all"]
