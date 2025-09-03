# run_optimization.py (v6 – robust + scaler-fix + korrekte trainerpfade)
# -----------------------------------------------------------------------------
# Änderungen ggü. v5:
# - TRAINER_MAP: korrigierte Modulpfade für XGBoost/Light_XGBoost (lowercase).
# - safe_scaler: benutze den vom Trainer gelieferten y_scaler für inverse_transform.
# - Val-Daten: wenn die Pipeline (X_train,y_train),(X_val,y_val) liefert, nutze direkt diese.
# -----------------------------------------------------------------------------

import os
import sys
import argparse
import importlib
import logging
import gc
import json
import random
from pathlib import Path
from datetime import datetime

import optuna
import numpy as np
import pandas as pd

# ----------------------------- Reproduzierbarkeit -----------------------------
random.seed(42)
np.random.seed(42)
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import tensorflow as tf  # noqa
    try:
        for d in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(d, True)
    except Exception:
        pass
except Exception:
    tf = None  # noqa


# ---------------------- 1) Dynamische Pfad-Ermittlung ------------------------
def find_project_root() -> Path:
    current_path = Path(__file__).resolve().parent
    sentinels = {"config", "ML_Algorithms", "Input"}
    for parent in [current_path, *current_path.parents]:
        try:
            if sentinels.issubset({p.name for p in parent.iterdir() if p.is_dir()}):
                print(f"✅ Projekt-Root gefunden: {parent}")
                return parent
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "Projekt-Root konnte nicht gefunden werden. Bitte das Skript im Projekt oder darunter platzieren."
    )

try:
    PROJECT_ROOT = find_project_root()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
except FileNotFoundError as e:
    print(f"Fehler: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("optimization_script")

# ---------------- 2) Projektimporte & Basis-Helfer ---------------------------
try:
    from config.config_general import CONFIG_PATH
    from ML_Helpfunctions import pipeline_utils as PU
except ImportError as e:
    logger.error(f"Kritischer Fehler beim Importieren von Projekt-Modulen: {e}")
    sys.exit(1)


def create_trial_config(algorithm: str, lags: int, horizon: int, optuna_params: dict) -> dict:
    algo = algorithm.lower()
    cfg = {
        "algorithm": algo,
        "model_name": f"{algo}_hpo",
        "dataset": "mqtt_data_filtered.csv",
        "lags": int(lags),
        "horizon": int(horizon),
        "rolling_window_size": max(1, int(lags // 2)),
        "base_features": ["Group4-2_S6_VolumetricFlowRate", "Group4-2_S6_MassFlowRate"],
        "time_features": [],
        "target_feature": "Group4-2_S6_VolumetricFlowRate",
        "validation_fraction": 0.2,
        "scale_target": True,
        "scale_other_features": True,
        "scaler_type": "minmax",
        "use_early_stopping": True,
        "early_stopping_patience": 10,
        "use_reduce_lr_on_plateau": True,
        "loss": "mse",
        "metrics": ["mae"],
        "no_save": True,
        "headless": True,
        "paths": CONFIG_PATH["paths"],
    }

    if algo in ("random_forest", "xgboost", "light_xgboost"):
        cfg["scale_other_features"] = False
        cfg["scale_target"] = False

    # HPO-Parameter-Payload passend für eure Trainer
    if algo == "xgboost":
        cfg["xgb_params"] = dict(optuna_params)
    elif algo == "light_xgboost":
        cfg["lgbm_params"] = dict(optuna_params)
    else:
        cfg.setdefault("model_params", {}).update(optuna_params)

    return cfg


# --------------- 3) Trainer-Mapping & Suchräume (nach Level) ----------------
TRAINER_MAP = {
    "lstm": ("ML_Algorithms.LSTM.lstm_train", "LSTMTrainer"),
    "cnn1d": ("ML_Algorithms.CNN1D.cnn1d_train", "CNN1DTrainer"),
    "random_forest": ("ML_Algorithms.Random_Forest.rf_train", "RandomForestTrainer"),
    # KORRIGIERT: lowercase-Dateinamen
    "xgboost": ("ML_Algorithms.XGBOOST.xgboost_train", "XGBoostTrainer"),
    "light_xgboost": ("ML_Algorithms.Light_XGBOOST.light_xgboost_train", "LightXGBoostTrainer"),
    "ridge": ("ML_Algorithms.RIDGE.ridge_lasso_train", "RidgeLassoTrainer"),
    "lasso": ("ML_Algorithms.RIDGE.ridge_lasso_train", "RidgeLassoTrainer"),
    "svm": ("ML_Algorithms.SVM.svm_train", "SVMTrainer"),
}

def get_trainer_class(algo: str) -> type:
    algo_key = "ridge" if algo == "lasso" else algo
    if algo_key not in TRAINER_MAP:
        raise ValueError(f"Kein Trainer für Algorithmus '{algo}' im TRAINER_MAP gefunden.")
    mod_path, cls_name = TRAINER_MAP[algo_key]
    try:
        module = importlib.import_module(mod_path)
        return getattr(module, cls_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Konnte Trainer '{cls_name}' aus '{mod_path}' nicht importieren: {e}")

def suggest_params_level(trial: optuna.Trial, algo: str, level: str) -> dict:
    """
    Schlägt Hyperparameter vor, mit klarer Trennung zwischen simple, medium und high.
    - simple: Schnelle, einfache Modelle mit geringer Komplexität.
    - medium: Feste, mittlere Komplexität zur soliden Evaluierung.
    - high: Hohe Komplexität und weite Suchräume für maximale Performance.
    """
    P = {}
    a, L = algo.lower(), level.lower()

    # ---------- LSTM ----------
    if a == "lstm":
        if L == "simple":
            P["num_layers"] = 1
            P["initial_units"] = trial.suggest_int("initial_units", 16, 56, step=4)
            P["dropout"] = trial.suggest_float("dropout", 0.0, 0.2)
            P["learning_rate"] = trial.suggest_float("learning_rate", 8e-4, 5e-3, log=True)
            P["batch_size"] = trial.suggest_categorical("batch_size", [64, 128])
            P["epochs"] = trial.suggest_int("epochs", 10, 25)
        elif L == "medium":
            P["num_layers"] = 2  # Feste mittlere Komplexität
            P["initial_units"] = trial.suggest_int("initial_units", 64, 128, step=16)
            P["dropout"] = trial.suggest_float("dropout", 0.15, 0.35)
            P["learning_rate"] = trial.suggest_float("learning_rate", 3e-4, 2e-3, log=True)
            P["batch_size"] = trial.suggest_categorical("batch_size", [32, 64])
            P["epochs"] = trial.suggest_int("epochs", 25, 50)
            P["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "nadam"])
        elif L == "high":
            P["num_layers"] = trial.suggest_int("num_layers", 3, 5)  # Hohe Komplexität
            P["initial_units"] = trial.suggest_int("initial_units_1", 128, 256, step=32)
            P["units_2"] = trial.suggest_int("units_2", 64, 128, step=32)
            P["units_3"] = trial.suggest_int("units_3", 32, 64, step=16)
            P["dropout"] = trial.suggest_float("dropout", 0.25, 0.5)
            P["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
            P["batch_size"] = trial.suggest_categorical("batch_size", [32, 64])
            P["epochs"] = trial.suggest_int("epochs", 50, 100)
            P["optimizer"] = trial.suggest_categorical("optimizer", ["adam", "nadam"])
            P["clipnorm"] = trial.suggest_float("clipnorm", 0.5, 2.0)

    # ---------- 1D-CNN ----------
    elif a == "cnn1d":
        if L == "simple":
            P["conv_blocks"] = 1
            P["filters"] = trial.suggest_int("filters", 8, 32, step=8)
            P["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5])
            P["dropout"] = trial.suggest_float("dropout", 0.0, 0.15)
            P["learning_rate"] = trial.suggest_float("learning_rate", 8e-4, 4e-3, log=True)
            P["epochs"] = trial.suggest_int("epochs", 10, 25)
        elif L == "medium":
            P["conv_blocks"] = 2
            P["filters"] = trial.suggest_int("filters", 32, 96, step=16)
            P["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7])
            P["dropout"] = trial.suggest_float("dropout", 0.15, 0.35)
            P["learning_rate"] = trial.suggest_float("learning_rate", 3e-4, 1.5e-3, log=True)
            P["epochs"] = trial.suggest_int("epochs", 25, 50)
            P["pooling_type"] = trial.suggest_categorical("pooling_type", ["max", "average"])
        elif L == "high":
            P["conv_blocks"] = trial.suggest_int("conv_blocks", 3, 5)
            P["filters"] = trial.suggest_int("filters", 64, 256, step=32)
            P["kernel_size"] = trial.suggest_categorical("kernel_size", [3, 5, 7, 9])
            P["dropout"] = trial.suggest_float("dropout", 0.3, 0.5)
            P["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 9e-4, log=True)
            P["epochs"] = trial.suggest_int("epochs", 50, 100)
            P["pooling_type"] = trial.suggest_categorical("pooling_type", ["max", "average"])

    # ---------- Random Forest ----------
    elif a == "random_forest":
        if L == "simple":
            P["n_estimators"] = trial.suggest_int("n_estimators", 40, 100, step=10)
            P["max_depth"] = trial.suggest_int("max_depth", 5, 10)
            P["min_samples_split"] = trial.suggest_int("min_samples_split", 10, 20)
            P["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 5, 10)
        elif L == "medium":
            P["n_estimators"] = trial.suggest_int("n_estimators", 120, 300, step=20)
            P["max_depth"] = trial.suggest_int("max_depth", 12, 24)
            P["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 8)
            P["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 4)
            P["max_features"] = trial.suggest_float("max_features", 0.5, 0.9)
        elif L == "high":
            P["n_estimators"] = trial.suggest_int("n_estimators", 350, 800, step=50)
            P["max_depth"] = trial.suggest_int("max_depth", 25, 60)
            P["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 5)
            P["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 3)
            P["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.8])
            P["criterion"] = trial.suggest_categorical("criterion", ["squared_error", "absolute_error"])
            
    # ---------- XGBoost ----------
    elif a == "xgboost":
        if L == "simple":
            P["n_estimators"] = trial.suggest_int("n_estimators", 100, 400, step=50)
            P["max_depth"] = trial.suggest_int("max_depth", 3, 5)
            P["learning_rate"] = trial.suggest_float("learning_rate", 0.03, 0.15, log=True)
        elif L == "medium":
            P["n_estimators"] = trial.suggest_int("n_estimators", 400, 800, step=50)
            P["max_depth"] = trial.suggest_int("max_depth", 5, 8)
            P["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.08, log=True)
            P["subsample"] = trial.suggest_float("subsample", 0.7, 1.0)
            P["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.7, 1.0)
            P["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
            P["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)
        elif L == "high":
            P["n_estimators"] = trial.suggest_int("n_estimators", 800, 1500, step=100)
            P["max_depth"] = trial.suggest_int("max_depth", 7, 12)
            P["learning_rate"] = trial.suggest_float("learning_rate", 0.005, 0.05, log=True)
            P["subsample"] = trial.suggest_float("subsample", 0.6, 0.9)
            P["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 0.9)
            P["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 5.0)
            P["reg_lambda"] = trial.suggest_float("reg_lambda", 0.5, 10.0)
            P["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 10)
            P["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)

    # ---------- LightGBM (Projekt: Light_XGBOOST) ----------
    elif a == "light_xgboost":
        if L == "simple":
            P["n_estimators"] = trial.suggest_int("n_estimators", 100, 400, step=50)
            P["max_depth"] = trial.suggest_int("max_depth", -1, 8)
            P["learning_rate"] = trial.suggest_float("learning_rate", 0.03, 0.15, log=True)
        elif L == "medium":
            P["n_estimators"] = trial.suggest_int("n_estimators", 400, 800, step=50)
            P["max_depth"] = trial.suggest_int("max_depth", -1, 12)
            P["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.08, log=True)
            P["subsample"] = trial.suggest_float("subsample", 0.7, 1.0)
            P["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.7, 1.0)
            P["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True)
            P["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True)
        elif L == "high":
            P["n_estimators"] = trial.suggest_int("n_estimators", 800, 1500, step=100)
            P["max_depth"] = trial.suggest_int("max_depth", -1, 16)
            P["learning_rate"] = trial.suggest_float("learning_rate", 0.005, 0.05, log=True)
            P["subsample"] = trial.suggest_float("subsample", 0.6, 0.9)
            P["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 0.9)
            P["reg_alpha"] = trial.suggest_float("reg_alpha", 0.0, 5.0)
            P["reg_lambda"] = trial.suggest_float("reg_lambda", 0.5, 10.0)
            P["min_child_samples"] = trial.suggest_int("min_child_samples", 5, 50)

    # ---------- Ridge / Lasso ----------
    elif a in ("ridge", "lasso"):
        if L == "simple":
            P["alpha"] = trial.suggest_float("alpha", 1.0, 50.0, log=True)
        elif L == "medium":
            P["alpha"] = trial.suggest_float("alpha", 0.1, 10.0, log=True)
        elif L == "high":
            P["alpha"] = trial.suggest_float("alpha", 0.001, 1.0, log=True)

    # ---------- SVM ----------
    elif a == "svm":
        if L == "simple":
            P["svm_kernel"] = "linear"
            P["C"] = trial.suggest_float("C", 0.01, 1.0, log=True)
            P["epsilon"] = trial.suggest_float("epsilon", 0.1, 0.5)
        elif L == "medium":
            P["svm_kernel"] = "rbf"
            P["C"] = trial.suggest_float("C", 1.0, 100.0, log=True)
            P["epsilon"] = trial.suggest_float("epsilon", 0.01, 0.2)
            P["gamma"] = trial.suggest_float("gamma", 1e-3, 1e-1, log=True)
        elif L == "high":
            P["svm_kernel"] = trial.suggest_categorical("svm_kernel", ["rbf", "poly"])
            P["C"] = trial.suggest_float("C", 100.0, 1000.0, log=True)
            P["epsilon"] = trial.suggest_float("epsilon", 0.001, 0.1, log=True)
            if P["svm_kernel"] == "rbf":
                P["gamma"] = trial.suggest_float("gamma", 1e-4, 0.5, log=True)
            if P["svm_kernel"] == "poly":
                P["degree"] = trial.suggest_int("degree", 2, 4)

    if not P:
        raise ValueError(f"Keine Parameter-Definition für '{a}' im Level '{L}' vorhanden.")
    return P

# ------------------------ 4) Optuna-Objective -------------------------------
def _ensure_2d(a):
    a = np.asarray(a)
    return a.reshape(-1, 1) if a.ndim == 1 else a

class Objective:
    def __init__(self, algorithm: str, level: str, lags: int, horizon: int, metric: str):
        self.algorithm = algorithm
        self.level = level
        self.lags = lags
        self.horizon = horizon
        self.metric = metric
        self.trainer_class = get_trainer_class(self.algorithm)

    def __call__(self, trial: optuna.Trial) -> float:
        random.seed(42)
        np.random.seed(42)

        try:
            params = suggest_params_level(trial, self.algorithm, self.level)
            cfg = create_trial_config(self.algorithm, self.lags, self.horizon, params)

            # Trainieren (liefert y_scaler zurück)
            trainer = self.trainer_class(config=cfg, folder_flag=self.algorithm.upper())
            model, _, y_scaler, _ = trainer.run(save_artifacts=False)

            # Validierungsdaten robust holen
            X_val, y_val = None, None
            try:
                pipeline = trainer._setup_pipeline()  # falls keine öffentliche API existiert
                ret = pipeline.prepare_training_data()
                # Bevorzugt: Pipeline liefert explizit Train/Val
                if isinstance(ret, tuple):
                    if len(ret) >= 2 and isinstance(ret[0], tuple) and isinstance(ret[1], tuple):
                        (_, _), (X_val, y_val) = ret[0], ret[1]
                    elif len(ret) >= 2 and not isinstance(ret[0], tuple):
                        X_all, y_all = ret[0], ret[1]
                        _, _, X_val, y_val = PU.create_timeseries_validation_split(X_all, y_all, cfg)
                if X_val is None or y_val is None:
                    raise ValueError("Validierungssplit konnte nicht ermittelt werden.")
            except Exception:
                raise

            # Vorhersage (generischer predict)
            try:
                y_pred_scaled = model.predict(X_val, verbose=0)
            except TypeError:
                y_pred_scaled = model.predict(X_val)

            # Rückskalierung mit dem vom Trainer gelieferten y_scaler
            if y_scaler is not None:
                y_pred = y_scaler.inverse_transform(_ensure_2d(y_pred_scaled)).ravel()
                y_true = y_scaler.inverse_transform(_ensure_2d(y_val)).ravel()
            else:
                y_pred = np.asarray(y_pred_scaled).ravel()
                y_true = np.asarray(y_val).ravel()

            # Metrik
            metrics = PU.evaluate_all_metrics(y_true=y_true, y_pred=y_pred)
            mval = metrics.get(self.metric)
            if mval is None:
                # Fallbacks
                if self.metric.startswith("mae"):
                    mval = float(np.nanmean(np.abs(y_true - y_pred)))
                elif self.metric == "rmse":
                    mval = float(np.sqrt(np.nanmean((y_true - y_pred) ** 2)))
                elif self.metric == "mse":
                    mval = float(np.nanmean((y_true - y_pred) ** 2))
                else:
                    mval = float(np.nanmean(np.abs(y_true - y_pred)))
            objective_value = float(np.asarray(mval).mean())
            return objective_value

        except Exception as e:
            logger.warning(f"Trial {trial.number} fehlgeschlagen: {e}", exc_idx=False)
            try:
                trial.set_user_attr("error", str(e))
            except Exception:
                pass
            return float("inf")
        finally:
            gc.collect()
            if "lstm" in self.algorithm or "cnn1d" in self.algorithm:
                try:
                    import tensorflow as tf
                    tf.keras.backend.clear_session()
                except Exception:
                    pass


# ---------------------------- 5) Hauptlogik ---------------------------------
def main():
    parser = argparse.ArgumentParser(description="Skript zur Hyperparameter-Optimierung.")
    parser.add_argument("-t", "--trials", type=int, required=True, help="Anzahl der Optuna-Trials pro Kombination.")
    parser.add_argument(
        "-l", "--levels", nargs='+', required=True, choices=['simple', 'medium', 'high'],
        help="Liste der Komplexitätslevel (z. B. simple medium)."
    )
    parser.add_argument(
        "-a", "--algorithms", nargs='+', default=list(TRAINER_MAP.keys()),
        choices=list(TRAINER_MAP.keys()), help="Zu optimierende Algorithmen (default: alle)."
    )
    parser.add_argument("--lags", type=int, default=4, help="Anzahl der Lags.")
    parser.add_argument("--horizon", type=int, default=4, help="Prognosehorizont.")
    args = parser.parse_args()

    out_root = PROJECT_ROOT / "Output" / "Optimization_Results"
    out_root.mkdir(parents=True, exist_ok=True)

    all_results_df, best_hyperparams = [], {}

    for algo in args.algorithms:
        for level in args.levels:
            study_name = f"opt_{algo}_{level}_{datetime.now().strftime('%Y%m%d')}"
            logger.info(f"=== STARTE OPTIMIERUNG für '{algo.upper()}', Level: '{level.upper()}' ===")

            # File-Logging je Studie
            log_path = out_root / f"{study_name}.log"
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(fh)

            # SQLite-Storage (resume-fähig)
            storage = f"sqlite:///{(out_root / f'{study_name}.db').as_posix()}"

            study = optuna.create_study(
                direction="minimize",
                study_name=study_name,
                storage=storage,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=42),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            )

            objective = Objective(
                algorithm=algo, level=level, lags=args.lags, horizon=args.horizon, metric="mae"
            )

            try:
                study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
            except KeyboardInterrupt:
                logger.warning("Optimierung durch Benutzer abgebrochen.")

            try:
                best_trial = study.best_trial
                logger.info(f"Bester Trial für '{algo}/{level}': Wert={best_trial.value}, Params={best_trial.params}")
                best_hyperparams[f"{algo}_{level}"] = {"value": best_trial.value, "params": best_trial.params}
            except ValueError:
                logger.error(f"Keine erfolgreichen Trials für '{algo}/{level}' abgeschlossen.")

            df_trial = study.trials_dataframe()
            df_trial['algorithm'] = algo
            df_trial['level'] = level
            all_results_df.append(df_trial)

            logger.removeHandler(fh)
            fh.close()

    if all_results_df:
        final_df = pd.concat(all_results_df, ignore_index=True)
        final_csv_path = out_root / f"HPO_Trials_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        final_df.to_csv(final_csv_path, index=False)
        logger.info(f"✅ Detaillierte Trial-Ergebnisse gespeichert in: {final_csv_path}")

    if best_hyperparams:
        params_path = out_root / "best_hyperparameters.json"
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(best_hyperparams, f, indent=4, ensure_ascii=False)
        logger.info(f"✅ Beste Hyperparameter gespeichert in: {params_path}")


if __name__ == "__main__":
    main()
