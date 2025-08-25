
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment-Pipeline (Multi-Config, inline)
------------------------------------------
* Läuft für N Horizons pro Modell je drei Komplexitätsstufen (simple, medium, high).
* Konfigurationen stehen **in diesem Skript** (keine externen Config-Module nötig).
* Ablauf ansonsten angelehnt an die bestehende experiment_pipeline:
  - programmatisches Training
  - anschließende Inferenz via pipeline_web_app.py (Subprozess) mit --load_id
  - kompakte Metrik-Zusammenfassung (Durchschnittswerte)

Unterstützte Modelle:
  - lstm
  - cnn1d
  - random_forest
  - xgboost

Beispiel:
  python experiment_pipeline_multiconfig.py \
    --algorithms lstm,cnn1d,random_forest,xgboost \
    --horizons 1,4,8 \
    --lags 4 \
    --inference-steps 60 \
    --loading-strategy split

Hinweis:
  * Die Komplexität der "high"-Varianten wurde bewusst moderat gehalten, damit
    Training auf einem Edge-Device weiterhin realistisch bleibt.
"""
import argparse
import json
import os
import sys
import time
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --- Projektpfad robust ergänzen ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # erwartet Skript im Projekt-/Subordner
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Import: allgemeine Pfade / MQTT / Artifacts ---
try:
    from config.config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG  # type: ignore
except ModuleNotFoundError:
    # Fallback, falls Skript direkt im Root liegt und 'config' nicht als Paket installiert ist
    from config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG  # type: ignore

try:
    from ML_Helpfunctions import pipeline_utils as PU  # type: ignore
except ModuleNotFoundError:
    try:
        from ML_Helpfunktions import pipeline_utils as PU  # type: ignore
    except ModuleNotFoundError:
        import importlib
        PU = importlib.import_module('pipeline_utils')  # letzte Chance (Root)

# --- Trainer-Zuordnung je Algorithmus ---
TRAINER_MAP = {
    "lstm": ("ML_Algorithms.LSTM.lstm_train", "LSTMTrainer", "LSTM"),
    "cnn1d": ("ML_Algorithms.CNN1D.cnn1d_train", "CNN1DTrainer", "CNN1D"),
    "random_forest": ("ML_Algorithms.Random_Forest.rf_train", "RandomForestTrainer", "Random_Forest"),
    "xgboost": ("ML_Algorithms.XGBOOST.xgboost", "XGBoostTrainer", "XGBOOST"),
    # Light XGBoost -> nutzt den XGBoost-Trainer (leichtere Hyperparameter via Config), eigener Ordner-Flag
    "light_xgboost": ("ML_Algorithms.ight_XGBOOST.light_xgboost_train", "LightXGBoostTrainer", "LIGHT_XGBOOST"),
}

MODEL_FILENAME_DEFAULTS = {
    "lstm": "model.keras",
    "cnn1d": "model.keras",
    "random_forest": "model.joblib",
    "xgboost": "model.json",
    "light_xgboost": "model.json", 
}

COMPLEXITY_PRESETS = {
  'lstm': {
    'simple': {'num_layers':1,'initial_units':32,'dropout':0.1,'batch_size':64,'epochs':20,'learning_rate':0.003,'optimizer':'adam','loss':'mse','clipnorm':1.0,'model_params':{'num_layers':1,'initial_units':32,'dropout':0.1,'batch_size':64,'epochs':20,'learning_rate':0.003,'optimizer':'adam','loss':'mse','clipnorm':1.0}},
    'medium': {'num_layers':2,'initial_units':64,'dropout':0.2,'batch_size':64,'epochs':40,'learning_rate':0.002,'optimizer':'adam','loss':'mse','clipnorm':1.5,'model_params':{'num_layers':2,'initial_units':64,'dropout':0.2,'batch_size':64,'epochs':40,'learning_rate':0.002,'optimizer':'adam','loss':'mse','clipnorm':1.5}},
    'high':   {'num_layers':3,'initial_units':96,'dropout':0.25,'batch_size':32,'epochs':60,'learning_rate':0.002,'optimizer':'nadam','loss':'mse','clipnorm':2.0,'model_params':{'num_layers':3,'initial_units':96,'dropout':0.25,'batch_size':32,'epochs':60,'learning_rate':0.002,'optimizer':'nadam','loss':'mse','clipnorm':2.0}},
  },
  'cnn1d': {
    'simple': {'cnn_blocks':1,'cnn_base_filters':32,'cnn_kernel_size':3,'cnn_dropout':0.05,'cnn_activation':'relu','batch_size':64,'epochs':20,'optimizer':'adam','learning_rate':0.003,'clipnorm':1.0,'loss':'huber','model_params':{'cnn_blocks':1,'cnn_base_filters':32,'cnn_kernel_size':3,'cnn_dropout':0.05,'cnn_activation':'relu','batch_size':64,'epochs':20,'optimizer':'adam','learning_rate':0.003,'clipnorm':1.0,'loss':'huber'}},
    'medium': {'cnn_blocks':2,'cnn_base_filters':64,'cnn_kernel_size':5,'cnn_dropout':0.15,'cnn_activation':'relu','batch_size':64,'epochs':40,'optimizer':'adam','learning_rate':0.002,'clipnorm':1.5,'loss':'huber','model_params':{'cnn_blocks':2,'cnn_base_filters':64,'cnn_kernel_size':5,'cnn_dropout':0.15,'cnn_activation':'relu','batch_size':64,'epochs':40,'optimizer':'adam','learning_rate':0.002,'clipnorm':1.5,'loss':'huber'}},
    'high':   {'cnn_blocks':3,'cnn_base_filters':96,'cnn_kernel_size':7,'cnn_dropout':0.2,'cnn_activation':'relu','batch_size':32,'epochs':60,'optimizer':'adam','learning_rate':0.0015,'clipnorm':2.0,'loss':'huber','model_params':{'cnn_blocks':3,'cnn_base_filters':96,'cnn_kernel_size':7,'cnn_dropout':0.2,'cnn_activation':'relu','batch_size':32,'epochs':60,'optimizer':'adam','learning_rate':0.0015,'clipnorm':2.0,'loss':'huber'}},
  },
  'random_forest': {
    'simple': {'n_estimators':120,'max_depth':6,'min_samples_split':4,'min_samples_leaf':4,'max_features':0.8,'bootstrap':True,'n_jobs':1,'random_state':42,'model_params':{'n_estimators':120,'max_depth':6,'min_samples_split':4,'min_samples_leaf':4,'max_features':0.8,'bootstrap':True,'n_jobs':1,'random_state':42}},
    'medium': {'n_estimators':280,'max_depth':10,'min_samples_split':4,'min_samples_leaf':3,'max_features':0.6,'bootstrap':True,'n_jobs':-1,'random_state':42,'model_params':{'n_estimators':280,'max_depth':10,'min_samples_split':4,'min_samples_leaf':3,'max_features':0.6,'bootstrap':True,'n_jobs':-1,'random_state':42}},
    'high':   {'n_estimators':400,'max_depth':12,'min_samples_split':4,'min_samples_leaf':2,'max_features':0.5,'bootstrap':True,'n_jobs':-1,'random_state':42,'model_params':{'n_estimators':400,'max_depth':12,'min_samples_split':4,'min_samples_leaf':2,'max_features':0.5,'bootstrap':True,'n_jobs':-1,'random_state':42}},
  },
  'xgboost': {
    'simple': {'n_estimators':200,'max_depth':3,'learning_rate':0.02,'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':1,'gamma':0.0,'reg_lambda':1.0,'reg_alpha':0.0,'tree_method':'hist','n_jobs':-1,'random_state':42,'objective':'reg:squarederror','xgb_params':{'n_estimators':200,'max_depth':3,'learning_rate':0.02,'subsample':0.8,'colsample_bytree':0.8,'min_child_weight':1,'gamma':0.0,'reg_lambda':1.0,'reg_alpha':0.0,'tree_method':'hist','n_jobs':-1,'random_state':42,'objective':'reg:squarederror'}},
    'medium': {'n_estimators':400,'max_depth':5,'learning_rate':0.015,'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':5,'gamma':1.5,'reg_lambda':1.0,'reg_alpha':0.0,'tree_method':'hist','n_jobs':-1,'random_state':42,'objective':'reg:squarederror','xgb_params':{'n_estimators':400,'max_depth':5,'learning_rate':0.015,'subsample':0.8,'colsample_bytree':0.7,'min_child_weight':5,'gamma':1.5,'reg_lambda':1.0,'reg_alpha':0.0,'tree_method':'hist','n_jobs':-1,'random_state':42,'objective':'reg:squarederror'}},
    'high':   {'n_estimators':600,'max_depth':6,'learning_rate':0.013,'subsample':0.75,'colsample_bytree':0.65,'min_child_weight':8,'gamma':2.0,'reg_lambda':1.5,'reg_alpha':1e-5,'tree_method':'hist','n_jobs':-1,'random_state':42,'objective':'reg:squarederror','xgb_params':{'n_estimators':600,'max_depth':6,'learning_rate':0.013,'subsample':0.75,'colsample_bytree':0.65,'min_child_weight':8,'gamma':2.0,'reg_lambda':1.5,'reg_alpha':1e-5,'tree_method':'hist','n_jobs':-1,'random_state':42,'objective':'reg:squarederror'}},
  },
  'light_xgboost': {
    'simple': {'n_estimators':100,'max_depth':3,'num_leaves':16,'learning_rate':0.03,'bagging_fraction':0.90,'bagging_freq':1,'feature_fraction':0.90,'min_child_samples':20,'min_split_gain':0.0,'reg_lambda':0.5,'reg_alpha':0.0,'max_bin':64,'n_jobs':1,'random_state':42,'objective':'regression','lgbm_params':{'n_estimators':100,'max_depth':3,'num_leaves':16,'learning_rate':0.03,'bagging_fraction':0.90,'bagging_freq':1,'feature_fraction':0.90,'min_child_samples':20,'min_split_gain':0.0,'reg_lambda':0.5,'reg_alpha':0.0,'max_bin':64,'n_jobs':1,'random_state':42,'objective':'regression'}},
    'medium': {'n_estimators':200,'max_depth':4,'num_leaves':32,'learning_rate':0.02,'bagging_fraction':0.85,'bagging_freq':1,'feature_fraction':0.80,'min_child_samples':25,'min_split_gain':0.5,'reg_lambda':0.8,'reg_alpha':0.0,'max_bin':96,'n_jobs':1,'random_state':42,'objective':'regression','lgbm_params':{'n_estimators':200,'max_depth':4,'num_leaves':32,'learning_rate':0.02,'bagging_fraction':0.85,'bagging_freq':1,'feature_fraction':0.80,'min_child_samples':25,'min_split_gain':0.5,'reg_lambda':0.8,'reg_alpha':0.0,'max_bin':96,'n_jobs':1,'random_state':42,'objective':'regression'}},
    'high':   {'n_estimators':300,'max_depth':5,'num_leaves':64,'learning_rate':0.015,'bagging_fraction':0.80,'bagging_freq':1,'feature_fraction':0.75,'min_child_samples':30,'min_split_gain':1.0,'reg_lambda':1.0,'reg_alpha':0.0,'max_bin':128,'n_jobs':1,'random_state':42,'objective':'regression','lgbm_params':{'n_estimators':300,'max_depth':5,'num_leaves':64,'learning_rate':0.015,'bagging_fraction':0.80,'bagging_freq':1,'feature_fraction':0.75,'min_child_samples':30,'min_split_gain':1.0,'reg_lambda':1.0,'reg_alpha':0.0,'max_bin':128,'n_jobs':1,'random_state':42,'objective':'regression'}},
  },
}

# --- Basis-Defaults, die für alle Modelle gelten ---
BASE_COMMON = {
    "dataset": "mqtt_data_filtered.csv",
    "loading_strategy": "split",
    "train_fraction": 0.8,
    "base_features": ["Group4-2_S6_VolumetricFlowRate", "Group4-2_S6_MassFlowRate"],
    "time_features": [],
    "target_feature": "Group4-2_S6_VolumetricFlowRate",
    "scale_other_features": True,
    "scale_target": True,
    "scaler_type": "robust",
    "inference_interval_sec": 1.0,
    "edge_device": True,    # Fokus: Edge-trainierbar
    "enable_edge": True,
    "validation_fraction": 0.2,
    "early_stopping_patience": 10,
}


INFERENCE_CONFIG_BY_ALGO = {
    "lstm": "lstm_edge",
    "cnn1d": "cnn1d_edge",
    "random_forest": "random_forest_edge",
    "xgboost": "xgboost_edge",
    "light_xgboost": "light_xgboost_edge",  # <- hinzufügen
}

def _try_import(module: str, attr: str):
    import importlib
    m = importlib.import_module(module)
    return getattr(m, attr)

def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out

def algorithm_to_folder(flag: str) -> str:
    f = flag.lower()
    if "lstm" in f: return "LSTM"
    if "cnn" in f: return "CNN1D"
    if "light_xgboost" in f or "light_xgb" in f: return "Light_XGBOOST"  # <- neu
    if "xgb" in f or "xgboost" in f: return "XGBOOST"
    if "rf" in f or "random_forest" in f: return "Random_Forest"
    return f.upper()

def build_training_config(algorithm: str, level: str, lags: int, horizon: int) -> dict:
    """Erzeugt eine zusammengeführte Trainings-Config aus Basisteilen + Komplexitätsprofilen."""
    algo = algorithm.lower()
    if algo not in COMPLEXITY_PRESETS:
        raise ValueError(f"Unsupported algorithm '{algorithm}'")
    preset = COMPLEXITY_PRESETS[algo][level]
    model_filename = MODEL_FILENAME_DEFAULTS[algo]

    cfg = _deep_merge(BASE_COMMON, {
        "lags": int(lags),
        "horizon": int(horizon),
        "model_name": f"{algo}_{level}",
        "model_filename": model_filename,
        "inference_mode": "load_artifacts_path",
    })
    cfg = _deep_merge(cfg, preset)
    cfg = _deep_merge(cfg, {"paths": CONFIG_PATH["paths"]})
    cfg = _deep_merge(cfg, CONFIG_LOAD_ARTIFACTS)
    cfg = _deep_merge(cfg, MQTT_CONFIG)
    return cfg

# ---------------------------
# Training (programmatisch)
# ---------------------------
def run_training(algorithm: str, config: dict, folder_flag: str) -> Tuple[str, Path]:
    module, clsname, default_flag = TRAINER_MAP[algorithm]
    Trainer = _try_import(module, clsname)
    use_flag = folder_flag or default_flag
    t0 = time.perf_counter()
    trainer = Trainer(config=config, folder_flag=use_flag)
    trainer.run(save_artifacts=True)
    dur_s = time.perf_counter() - t0

    # Trainingszeit in training_config.json ergänzen (falls vorhanden)
    try:
        training_cfg_json = Path(config["paths"]["Models"]) / "training_config.json"
        if training_cfg_json.exists():
            with open(training_cfg_json, "r", encoding="utf-8") as f:
                js = json.load(f)
            js["training_time_s"] = round(float(dur_s), 3)
            with open(training_cfg_json, "w", encoding="utf-8") as f:
                json.dump(js, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ Could not write training_time_s: {e}")

    run_id = str(config.get("run_id"))
    models_dir = Path(config["paths"]["Models"])  # durch setup_experiment gesetzt
    return run_id, models_dir

# ---------------------------
# Inferenz
# ---------------------------
def _import_pipeline_web_app():
    import importlib.util as _ilu
    candidates = [
        ('pipeline_web_app', PROJECT_ROOT / 'pipeline_web_app.py'),
        ('ML_Algorithms.pipeline_web_app', PROJECT_ROOT / 'ML_Algorithms' / 'pipeline_web_app.py'),
    ]
    ml_path = PROJECT_ROOT / 'ML_Algorithms'
    if ml_path.exists() and str(ml_path) not in sys.path:
        sys.path.append(str(ml_path))
    for modname, fpath in candidates:
        try:
            return __import__(modname, fromlist=['*'])
        except ModuleNotFoundError:
            if fpath.exists():
                spec = _ilu.spec_from_file_location(modname, fpath)
                if spec and spec.loader:
                    m = _ilu.module_from_spec(spec)
                    spec.loader.exec_module(m)
                    sys.modules[modname] = m
                    return m
    raise ModuleNotFoundError("pipeline_web_app not found in expected locations.")

def run_inference_via_subprocess(
    algorithm: str,
    run_id: str,
    model_filename: str,
    inference_steps: int,
    loading_strategy: str = "live_mqtt",
    interval_sec: float = 1.0,
    config_name: Optional[str] = None,
) -> int:
    """Ruft pipeline_web_app.py im reinen Inferenzmodus auf. Gibt den Exitcode zurück."""
    py = sys.executable
    # Resolve pipeline_web_app path robust
    app_path_candidates: List[Path] = []
    try:
        from types import ModuleType
        _pwa = _import_pipeline_web_app()
        if isinstance(_pwa, ModuleType):
            app_path_candidates.append(Path(getattr(_pwa, '__file__', '')))
    except Exception:
        pass
    app_path_candidates += [
        PROJECT_ROOT / 'pipeline_web_app.py',
        PROJECT_ROOT / 'ML_Algorithms' / 'pipeline_web_app.py',
    ]
    app: Optional[Path] = None
    for cand in app_path_candidates:
        if cand and isinstance(cand, Path) and cand.exists():
            app = cand
            break
    if app is None:
        print("❌ Could not locate pipeline_web_app.py in expected locations:")
        for cand in app_path_candidates:
            print(" -", cand)
        return 2

    cmd = [
        str(py), str(app),
        "--algorithm", algorithm,
        "--load_id", run_id,
        "--model_filename", model_filename,
        "--no-web",
        "--inference-steps", str(inference_steps),
        "--set", f"loading_strategy={loading_strategy}",
        "--set", f"inference_interval_sec={interval_sec}",
    ]
    if config_name:
        cmd += ["--config-name", config_name]

    print("[SPAWN] ", " ".join(map(str, cmd)))
    print(f"[INFO] Using pipeline_web_app at: {app}")
    import subprocess
    proc = subprocess.run(cmd)
    return proc.returncode

# ---------------------------
# Auswertung
# ---------------------------
def _discover_predictions_file_from_json(run_id: str, error_metrics_dir: Path) -> Optional[Path]:
    """Falls ErrorMetrics_all_runs.csv vorhanden ist, extrahiere predictions_file_path aus der Run-JSON."""
    agg_csv = error_metrics_dir / "ErrorMetrics_all_runs.csv"
    if not agg_csv.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(agg_csv)
        row = df[df["run_id"] == run_id].tail(1)
        if row.empty:
            return None
        json_path = Path(row.iloc[0]["json_path"]).resolve()
        if not json_path.exists():
            return None
        with open(json_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        p = js.get("extra_info", {}).get("predictions_file_path")
        return Path(p) if p else None
    except Exception:
        return None

def _fallback_find_step_csv(run_id: str, prediction_data_dir: Path) -> Optional[Path]:
    pattern = f"StepPredictions_{run_id}_*.csv"
    hits = list(prediction_data_dir.glob(pattern))
    return hits[-1] if hits else None

def summarize_step_csv(step_csv: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Gibt Ø inference_time_ms, Ø total_time_ms, Ø cpu_percent, Ø ram_percent zurück."""
    try:
        import pandas as pd
        df = pd.read_csv(step_csv)
        inf_ms = float(df["inference_time_s"].mean() * 1000.0) if "inference_time_s" in df else None
        tot_ms = float(df["total_time_s"].mean() * 1000.0) if "total_time_s" in df else None
        cpu = float(df["cpu_percent"].mean()) if "cpu_percent" in df else None
        ram = float(df["ram_percent"].mean()) if "ram_percent" in df else None
        return inf_ms, tot_ms, cpu, ram
    except Exception:
        return None, None, None, None

@dataclass
class InferenceResult:
    algorithm: str
    level: str
    lags: int
    horizon: int
    model_variant: str
    avg_inference_time_ms: Optional[float]
    avg_total_time_ms: Optional[float]
    avg_cpu_percent: Optional[float]
    avg_ram_percent: Optional[float]
    model_size_mb: Optional[float]
    run_id: str

def read_model_size_mb(training_config_json: Path) -> Optional[float]:
    if not training_config_json.exists():
        return None
    try:
        with open(training_config_json, "r", encoding="utf-8") as f:
            js = json.load(f)
        v = js.get("model_size_MB")
        return float(v) if v is not None else None
    except Exception:
        return None

def append_summary_row(summary_csv: Path, row: InferenceResult) -> None:
    header = [
        "algorithm", "level", "lags", "horizon", "model_variant",
        "avg_inference_time_ms", "avg_total_time_ms", "avg_cpu_percent", "avg_ram_percent",
        "model_size_mb", "run_id",
    ]
    exists = summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            row.algorithm, row.level, row.lags, row.horizon, row.model_variant,
            row.avg_inference_time_ms, row.avg_total_time_ms, row.avg_cpu_percent, row.avg_ram_percent,
            row.model_size_mb, row.run_id
        ])

# ---------------------------
# Main
# ---------------------------
def _range_from_str(spec: str) -> List[int]:
    spec = spec.strip()
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        else:
            start, stop, step = parts
        return list(range(start, stop + (1 if step > 0 else -1), step))
    return [int(x) for x in spec.split(",") if x]

def main():
    ap = argparse.ArgumentParser(description="Multi-Config Experiment Pipeline (inline configs)")
    ap.add_argument("--algorithms", type=str, default="lstm,cnn1d,random_forest,xgboost",
                    help="Kommagetrennte Liste: lstm,cnn1d,random_forest,xgboost")
    ap.add_argument("--horizons", type=str, default="1,4,8",
                    help="Kommagetrennt oder Range 'start:stop:step'")
    ap.add_argument("--lags", type=int, default=4, help="Anzahl Lags (gemeinsam für alle Läufe)")
    ap.add_argument("--inference-steps", type=int, default=60, help="Schritte pro Inferenzlauf")
    ap.add_argument("--loading-strategy", type=str, default="split", help="split | separate_csv | live_mqtt")
    ap.add_argument("--interval-sec", type=float, default=1.0, help="Inferenz-Intervall in Sekunden")
    ap.add_argument("--summary", type=str, default="Experiment_Summary_MultiConfig.csv", help="Ziel-CSV")
    args = ap.parse_args()

    algorithms = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]
    horizons = _range_from_str(args.horizons)
    lags = int(args.lags)
    summary_csv = Path(CONFIG_PATH["paths"]["output"]) / args.summary

    print("=== EXPERIMENT START ===")
    print("Algos     :", algorithms)
    print("Horizons  :", horizons)
    print("Lags      :", lags)
    print("Summary   :", summary_csv)

    for algo in algorithms:
        if algo not in TRAINER_MAP:
            print(f"⚠️  Überspringe unbekanntes Modell: {algo}")
            continue

        for H in horizons:
            for level in ("simple", "medium", "high"):
                print(f"\n--- Train {algo} | level={level} | lags={lags} | horizon={H} ---")

                # Config bauen und Experiment-Ordner aufsetzen
                cfg = build_training_config(algo, level, lags, H)
                folder_flag = algorithm_to_folder(algo)   # -> "LSTM", "CNN1D", "Random_Forest", "XGBOOST"
                cfg, _ = PU.setup_experiment(cfg, folder_flag, run_type="train")

                # Train
                run_id, models_dir = run_training(algo, cfg, folder_flag=folder_flag)
                print(f"[OK] Training done. run_id={run_id} @ {models_dir}")

                # Inferenz
                model_filename = cfg.get("model_filename", MODEL_FILENAME_DEFAULTS.get(algo, "model.bin"))
                rc = run_inference_via_subprocess(
                    algorithm=algo,
                    run_id=run_id,
                    model_filename=model_filename,
                    inference_steps=args.inference_steps,
                    loading_strategy=args.loading_strategy,
                    interval_sec=args.interval_sec,
                    config_name=INFERENCE_CONFIG_BY_ALGO.get(algo, f"{algo}_edge"),
                )
                if rc != 0:
                    print(f"❌ Inference subprocess exited with {rc} — Metriken evtl. unvollständig.")

                # Metrik-Zusammenfassung (StepPredictions)
                error_metrics_dir = Path(CONFIG_PATH["paths"]["output"]) / "Error_Metrics"
                prediction_data_dir = Path(CONFIG_PATH["paths"]["output"]) / "Prediction_Data"
                step_csv = _discover_predictions_file_from_json(run_id, error_metrics_dir)
                if step_csv is None:
                    step_csv = _fallback_find_step_csv(run_id, prediction_data_dir)
                inf_ms, tot_ms, cpu, ram = summarize_step_csv(step_csv) if step_csv else (None, None, None, None)

                # Modellgröße aus training_config.json lesen (falls vorhanden)
                model_size_mb = read_model_size_mb(models_dir / "training_config.json")

                result = InferenceResult(
                    algorithm=algo, level=level, lags=lags, horizon=H,
                    model_variant=model_filename,
                    avg_inference_time_ms=inf_ms, avg_total_time_ms=tot_ms,
                    avg_cpu_percent=cpu, avg_ram_percent=ram,
                    model_size_mb=model_size_mb,
                    run_id=run_id,
                )
                append_summary_row(summary_csv, result)
                print(f"[OK] Summary updated -> {summary_csv}")

    print("\n=== EXPERIMENT DONE ===")

if __name__ == "__main__":
    main()
