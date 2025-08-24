# ML_Helpfunctions/Light_XGBOOST_Utils.py
"""
Utilities für LightGBM ("Light_XGBoost") in der vereinheitlichten Pipeline.
- Training (inkl. Multi-Output via MultiOutputRegressor)
- Speichern von Modellen
- Ergebnispersistierung analog XGBOOST_Utils
"""
from __future__ import annotations
import os, sys, time, joblib, numpy as np, traceback

# Projekt-Root auf sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ML_Helpfunctions import pipeline_utils as PipelineUtils  # type: ignore

def train_light_xgboost_model(config: dict, X_train: np.ndarray, y_train: np.ndarray):
    """Training eines LightGBM-Regressors. Multi-Output via MultiOutputRegressor."""
    from lightgbm import LGBMRegressor
    from sklearn.multioutput import MultiOutputRegressor

    y_arr = np.asarray(y_train)
    if y_arr.ndim == 1:
        y_arr = y_arr.reshape(-1, 1)

    params = dict(config.get("lgbm_params") or {})
    base = LGBMRegressor(**params)

    if y_arr.shape[1] > 1:
        model = MultiOutputRegressor(base)
    else:
        H = int(config.get("horizon", 1))
        model = base if H == 1 else MultiOutputRegressor(base)

    t0 = time.perf_counter()
    model.fit(X_train, y_arr)
    t1 = time.perf_counter()
    return model, (t1 - t0)


def save_sklearn_model(model, config: dict, paths: dict) -> str | None:
    import os, joblib
    model_dir = paths.get("Models") or os.path.join(paths.get("output","."), "Models")
    os.makedirs(model_dir, exist_ok=True)
    model_filename = config.get("model_filename") or "model.joblib"  # <- wichtig
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(model, model_path, compress=3)
    print(f"📤 LightGBM-Modell gespeichert unter: {model_path}")
    return model_path



# ML_Helpfunctions/Light_XGBOOST_Utils.py

def save_sklearn_model(model, config: dict, paths: dict) -> str | None:
    import joblib, os, traceback
    try:
        model_dir = paths.get("Models") or os.path.join(paths.get("output", "."), "Models")
        os.makedirs(model_dir, exist_ok=True)
        # <-- WICHTIG: fixen Namen verwenden
        fname = config.get("model_filename", "model.joblib")
        model_path = os.path.join(model_dir, fname)
        joblib.dump(model, model_path, compress=3)
        print(f"[LGBM] Modell gespeichert: {model_path}")
        return model_path
    except Exception as e:
        print(f"[LGBM] Fehler beim Speichern: {e}\n{traceback.format_exc()}")
        return None
