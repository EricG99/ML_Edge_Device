# config/config_ml_ridge_lasso.py
# -----------------------------------------------------------------------------
# Ridge & Lasso Konfigurationen im Stil der bestehenden Configs
# Erwartete Profil-Variablen:
#   - ridge_edge, ridge_server, ridge
#   - lasso_edge, lasso_server, lasso
# -----------------------------------------------------------------------------

_COMMON = {
    "model_filename": "model.joblib",
    "dataset": "mqtt_data_filtered.csv",
    "loading_strategy": "split",
    "train_fraction": 0.8,

    # Fallbacks – werden von eurer Experiment-CLI überschrieben
    "lags": 4,
    "horizon": 4,

    # Target zuerst!
    "base_features": ["Group4-2_S6_VolumetricFlowRate", "Group4-2_S6_MassFlowRate"],
    "time_features": [],
    "target_feature": "Group4-2_S6_VolumetricFlowRate",

    "scale_other_features": True,
    "scale_target": False,
    "scaler_type": "robust",

    "inference_interval_sec": 1.0,

    "edge_device": False,
    "enable_edge": False,
}

# -----------------------
# RIDGE
# -----------------------
ridge_edge = {
    **_COMMON,
    "model_name": "ridge_edge",
    "algorithm": "ridge",
    "edge_device": True,
    "enable_edge": True,

    # Lineare Regularisierung – robust & schnell
    "alpha": 1.0,
    "fit_intercept": True,
    "tol": 1e-4,
    "max_iter": 10000,
    "random_state": 42,

    "model_params": {
        "alpha": 1.0,
        "fit_intercept": True,
        "tol": 1e-4,
        "max_iter": 10000,
        "random_state": 42,
    },

    "include_roll_mean": True,
    "include_roll_std": False,
}

ridge_server = {
    **_COMMON,
    "model_name": "ridge_server",
    "algorithm": "ridge",
    "edge_device": False,
    "enable_edge": False,

    # Etwas strengere Toleranz, mehr Iterationen
    "alpha": 0.5,
    "fit_intercept": True,
    "tol": 5e-5,
    "max_iter": 20000,
    "random_state": 42,

    "model_params": {
        "alpha": 0.5,
        "fit_intercept": True,
        "tol": 5e-5,
        "max_iter": 20000,
        "random_state": 42,
    },

    "include_roll_mean": True,
    "include_roll_std": True,
}

ridge = ridge_edge  # Alias

# -----------------------
# LASSO
# -----------------------
lasso_edge = {
    **_COMMON,
    "model_name": "lasso_edge",
    "algorithm": "lasso",
    "edge_device": True,
    "enable_edge": True,

    # Sparsity-freundlich, konservatives Alpha für Edge
    "alpha": 0.001,
    "fit_intercept": True,
    "tol": 1e-4,
    "max_iter": 15000,
    "random_state": 42,

    "model_params": {
        "alpha": 0.001,
        "fit_intercept": True,
        "tol": 1e-4,
        "max_iter": 15000,
        "random_state": 42,
    },

    "include_roll_mean": True,
    "include_roll_std": False,
}

lasso_server = {
    **_COMMON,
    "model_name": "lasso_server",
    "algorithm": "lasso",
    "edge_device": False,
    "enable_edge": False,

    # Feineres Alpha, mehr Iterationen für Konvergenz
    "alpha": 0.0005,
    "fit_intercept": True,
    "tol": 5e-5,
    "max_iter": 30000,
    "random_state": 42,

    "model_params": {
        "alpha": 0.0005,
        "fit_intercept": True,
        "tol": 5e-5,
        "max_iter": 30000,
        "random_state": 42,
    },

    "include_roll_mean": True,
    "include_roll_std": True,
}

lasso = lasso_edge  # Alias
