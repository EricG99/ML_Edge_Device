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

def add_rolling_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """ Fügt rollierende Mittelwerte und Standardabweichungen hinzu. """
    print("\n[DEBUG] --- Betrete Funktion: add_rolling_features ---")
    rolling_features_dict = {"rolling_mean": [], "rolling_std": []}
    
    # Prüfen, ob der Schritt ausgeführt werden soll
    should_run = config.get("add_rolling_features", True)
    # print(f"[DEBUG] 'add_rolling_features' in config gefunden? -> {should_run}")

    if should_run:
        window_size = config.get('rolling_window_size', 2)
        base_features_to_process = config.get("base_features", [])
        # print(f"[DEBUG] Fenstergröße für Rolling-Features: {window_size}")
        # print(f"[DEBUG] Basis-Features, die verarbeitet werden sollen: {base_features_to_process}")
        
        # # Verfügbare Spalten im DataFrame vor der Bearbeitung
        # print(f"[DEBUG] Verfügbare Spalten im DF: {df.columns.to_list()}")

        for feature in base_features_to_process:
            feature_lower = feature.lower()
            # print(f"[DEBUG] Verarbeite Rolling-Feature für: '{feature_lower}'")

            # Sicherheitscheck: Existiert die Spalte überhaupt im DataFrame?
            if feature_lower not in df.columns:
                # print(f"!!!!!!!!!! [DEBUG] FEHLER: Die Spalte '{feature_lower}' wurde im DataFrame nicht gefunden! Überspringe. !!!!!!!!!!")
                continue

            try:
                # Rolling Mean
                mean_name = f"{feature_lower}_roll_mean_{window_size}"
                df[mean_name] = df[feature_lower].rolling(window=window_size).mean()
                rolling_features_dict["rolling_mean"].append(mean_name)
                # print(f"    -> ✅ '{mean_name}' erstellt.")

                # Rolling Std
                std_name = f"{feature_lower}_roll_std_{window_size}"
                df[std_name] = df[feature_lower].rolling(window=window_size).std()
                rolling_features_dict["rolling_std"].append(std_name)
                # print(f"    -> ✅ '{std_name}' erstellt.")
            except Exception as e:
                print(f"!!!!!!!!!! [DEBUG] FEHLER bei der Erstellung von Rolling-Features für '{feature_lower}': {e} !!!!!!!!!!")

    # print("[DEBUG] --- Verlasse Funktion: add_rolling_features ---")
    return df, rolling_features_dict


def add_lag_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """ Fügt Lag-Features (zeitlich verschobene Werte) hinzu. """
    lag_features_dict = {"lags": []}
    should_run = config.get("add_lag_features", True)
    if should_run:
        num_lags = config.get("lags", 1)
        base_features_to_process = config.get("base_features", [])
        for feature in base_features_to_process:
            feature_lower = feature.lower()
            if feature_lower in df.columns:
                for lag in range(1, num_lags + 1):
                    lagged_name = f"{feature_lower}_lag_{lag}"
                    df[lagged_name] = df[feature_lower].shift(lag)
                    lag_features_dict["lags"].append(lagged_name)
    return df, lag_features_dict


def add_all_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """ Führt alle konfigurierten Feature-Engineering-Schritte aus. """
    df.columns = df.columns.str.lower()
    base_features_lower = [f.lower() for f in config.get("base_features", [])]

    df, time_dict = add_time_features(df, config)
    df, lag_dict = add_lag_features(df, config)
    df, rolling_dict = add_rolling_features(df, config)

    all_features = (
        base_features_lower +
        time_dict.get("time", []) + # Hinzugefügt
        lag_dict.get("lags", []) +
        rolling_dict.get("rolling_mean", []) +
        rolling_dict.get("rolling_std", [])
    )
    
    features_summary = {
        "base": base_features_lower,
        "time": time_dict.get("time", []), # Hinzugefügt
        "lags": lag_dict.get("lags", []),
        "rolling": rolling_dict,
        "all": sorted(list(set(all_features)))
    }
    
    return df, features_summary

def create_feature_list_from_dict(feature_dict: dict) -> list:
    return feature_dict["all"]