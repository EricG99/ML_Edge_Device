
# config/config_ml_xgboost.py

# --- Empfohlene Default-Konfig für tabulares XGBoost-Training mit Horizon H ---
# Die Keys orientieren sich an euren bestehenden Pipelines/FE-Flags.
xgb_default = {
    "model_name": "xgboost_csv_split",
    "dataset": "mqtt_data_filtered.csv",
    "loading_strategy": "split",      # "split" | "live_mqtt"
    "train_fraction": 0.8,

    # Zeitreihenparameter
    "lags": 4,
    "horizon": 4,
    "rolling_window_size": 2,

    # Feature Engineering (bitte mit eurer Feature_Engeneering.add_all_features abgleichen)
    "base_features": ['group4-2_s6_massflowrate'],  # Zielspalte zuerst!
    "add_lag_features": True,
    "add_rolling_features": True,
    "scale_other_features": True,   # X-Scaler
    "scale_target": True,          # y wird bei XGB nicht skaliert

    # Inferenz
    "inference_interval_sec": 1.0,

    # Retraining (Best-Practice für XGBoost)
    "xgb_additional_estimators": 200,
    "xgb_early_stopping_rounds": 20,  # 0 = deaktiviert
    "xgb_retrain_hist_rows": 5000,    # kleine Historie für Stabilität

    # XGBoost-Parameter (für XGBRegressor)
    "xgb_params": {
        "n_estimators": 400,
        "max_depth": 8,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "reg:squarederror",
        "tree_method": "hist"
    }
}

# "Produktions"-Preset als Alias, damit euer Loader 'xgboost' direkt findet.
xgboost = { **xgb_default }
