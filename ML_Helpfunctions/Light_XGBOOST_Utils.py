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

from ML_Helpfunctions import Pipeline_Utils as PipelineUtils  # type: ignore

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
    """Speichert das (sklearn-kompatible) Modell als .joblib."""
    try:
        model_dir = paths.get("Models") or os.path.join(paths.get("output", "."), "Models")
        os.makedirs(model_dir, exist_ok=True)
        model_name = (config.get("model_name") or "light_xgboost_model").replace(" ", "_")
        dataset_name = (config.get("dataset") or "data").replace(".csv", "").replace(" ", "_")
        model_filename = f"{model_name}_{dataset_name}_{config.get('run_id','')}_{config.get('time_stamp','')}.joblib"
        model_path = os.path.join(model_dir, model_filename)
        joblib.dump(model, model_path, compress=3)
        print(f"[LGBM] Modell gespeichert: {model_path}")
        return model_path
    except Exception as e:
        print(f"[LGBM] Fehler beim Speichern: {e}\n{traceback.format_exc()}")
        return None


def save_results_light_xgboost(config, model, scaler, pred_orig, true_orig, dates, metrics, paths, power_time):
    """Persistierung analog XGB-Utils._save_common_results + Modellpfad."""
    common = PipelineUtils._save_common_results(
        config=config,
        pred_orig=pred_orig,
        true_orig=true_orig,
        dates=dates,
        metrics=metrics,
        paths=paths,
        power_time=power_time,
        scaler=scaler
    )
    model_path = save_sklearn_model(model, config, paths)
    edge_artifacts_path = PipelineUtils.save_edge_artifacts(config, paths)
    out = dict(common)
    if model_path:
        out["model_path"] = model_path
    if edge_artifacts_path:
        out["edge_artifacts"] = edge_artifacts_path
    return out
