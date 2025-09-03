# ML_Algorithms/SVM/svm_utils.py
import time
import numpy as np
from typing import Tuple
from sklearn.svm import SVR, LinearSVR
from sklearn.multioutput import MultiOutputRegressor

def build_model(config: dict, horizon: int):
    """
    Creates an SVM regressor for time series.
    Defaults to a fast, edge-friendly LinearSVR for linear patterns.
    If config["svm_kernel"] != "linear", uses SVR(kernel=...).
    """
    kernel = str(config.get("svm_kernel", "linear")).lower()
    C = float(config.get("C", 1.0))
    epsilon = float(config.get("epsilon", 0.1))
    tol = float(config.get("tol", 1e-3))
    max_iter = int(config.get("max_iter", 10000))
    random_state = config.get("random_state", None)

    if kernel == "linear":
        base = LinearSVR(C=C, epsilon=epsilon, tol=tol, max_iter=max_iter, random_state=random_state, dual=True)
    else:
        # RBF/poly/sigmoid – heavier but sometimes necessary for non-linearities
        gamma = config.get("gamma", "scale")
        degree = int(config.get("degree", 3))
        base = SVR(kernel=kernel, C=C, epsilon=epsilon, tol=tol, gamma=gamma, degree=degree, max_iter=max_iter)

    if horizon > 1:
        return MultiOutputRegressor(base)
    return base

def train_model(config: dict, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[object, float]:
    """Fits (multi-output) SVM model and returns (model, train_time_s)."""
    H = int(config.get("horizon", 1))
    model = build_model(config, H)
    t0 = time.perf_counter()
    model.fit(X_train, y_train if H > 1 else np.ravel(y_train))
    dt = time.perf_counter() - t0
    return model, dt
