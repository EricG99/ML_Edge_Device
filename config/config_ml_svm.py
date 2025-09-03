# config/config_ml_svm.py
# -----------------------------------------------------------------------------
# SVM (LinearSVR / SVR) Konfigurationen im Stil der bestehenden Configs
# Erwartete Profil-Variablen:
#   - svm_edge, svm_server, svm
# -----------------------------------------------------------------------------

_COMMON = {
    "model_filename": "model.joblib",
    "dataset": "mqtt_data_filtered.csv",
    "loading_strategy": "split",
    "train_fraction": 0.8,

    "lags": 4,
    "horizon": 4,

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
# EDGE: LinearSVR – sehr schnell & speicherschonend
# -----------------------
svm_edge = {
    **_COMMON,
    "model_name": "svm_edge",
    "algorithm": "svm",
    "edge_device": True,
    "enable_edge": True,

    "svm_kernel": "linear",   # -> LinearSVR
    "C": 1.0,
    "epsilon": 0.1,
    "tol": 1e-3,
    "max_iter": 10000,
    "random_state": 42,

    "model_params": {
        "svm_kernel": "linear",
        "C": 1.0,
        "epsilon": 0.1,
        "tol": 1e-3,
        "max_iter": 10000,
        "random_state": 42,
    },

    "include_roll_mean": True,
    "include_roll_std": False,
}

# -----------------------
# SERVER: RBF-SVR – nichtlinear, aber schwerer
# -----------------------
svm_server = {
    **_COMMON,
    "model_name": "svm_server",
    "algorithm": "svm",
    "edge_device": False,
    "enable_edge": False,

    "svm_kernel": "rbf",      # -> SVR(kernel='rbf')
    "C": 2.0,
    "epsilon": 0.1,
    "gamma": "scale",
    "tol": 1e-3,
    "max_iter": 25000,

    "model_params": {
        "svm_kernel": "rbf",
        "C": 2.0,
        "epsilon": 0.1,
        "gamma": "scale",
        "tol": 1e-3,
        "max_iter": 25000,
    },

    "include_roll_mean": True,
    "include_roll_std": True,
}

svm = svm_edge  # Alias
