# ML_Algorithms/RidgeLasso/ridge_lasso_utils.py
import time
import numpy as np
from typing import Tuple
from sklearn.linear_model import Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor

def build_model(config: dict, horizon: int):
    """
    Creates a Ridge or Lasso regressor based on config["algorithm"] or config["model_name"].
    Uses MultiOutputRegressor when horizon > 1.
    """
    algo = str(config.get("algorithm") or config.get("model_name") or "ridge").lower()
    alpha = float(config.get("alpha", 1.0 if "ridge" in algo else 0.001))
    max_iter = int(config.get("max_iter", 10000))
    tol = float(config.get("tol", 1e-4))
    fit_intercept = bool(config.get("fit_intercept", True))
    random_state = config.get("random_state", None)

    if "lasso" in algo:
        base = Lasso(alpha=alpha, max_iter=max_iter, tol=tol, fit_intercept=fit_intercept, random_state=random_state)
    else:
        base = Ridge(alpha=alpha, tol=tol, fit_intercept=fit_intercept, random_state=random_state)

    if horizon > 1:
        return MultiOutputRegressor(base)
    return base

def train_model(config: dict, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[object, float]:
    """Fits the (multi-output) Ridge/Lasso model and returns (model, train_time_s)."""
    H = int(config.get("horizon", 1))
    model = build_model(config, H)
    t0 = time.perf_counter()
    model.fit(X_train, y_train if H > 1 else np.ravel(y_train))
    dt = time.perf_counter() - t0
    return model, dt
