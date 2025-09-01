
#!/usr/bin/env python3
from __future__ import annotations
"""
Experiment Pipeline
-------------------

Zweck
  * Führt automatisiert Trainings- und Inferenzläufe über Modelle, Komplexitätsstufen (simple/medium/high)A,
    einen Horizon-Grid und (falls zutreffend) quantisierte Modellvarianten aus.
  * Nutzt fest integrierte Hyperparameter-Sets je nach Komplexitätsstufe.
  * Startet das initiale Training programmatisch, sammelt die erzeugte run_id
    und ruft anschließend die bestehende Web/Headless-Pipeline (`pipeline_web_app.py`) für die
    Inferenz mit `--load_id` auf.
  * Aggregiert Metriken (Ø Inferenzzeit, Ø Total Time, Ø CPU %, Ø RAM %) je Kombination und
    speichert eine kompakte Übersichtstabelle.
  * Löscht nach jeder Inferenz die verwendeten Modellbinaries, um Speicherplatz zu sparen.

Wichtige Hinweise
  * Lags sind fest auf 20 gesetzt, rolling_window_size auf 10.
  * Für `cnn1d` und `lstm` werden automatisch Inferenzen für `no-quant`, `quant-16` und `quant-8` durchgeführt.
    Für die anderen Modelle nur `no-quant`.
  * Inferenzmodus ist standardmäßig `split`.

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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# bevorzugt: aus dem Paket importieren
from ML_Algorithms.pipeline_web_app import (
    algorithm_to_folder,
    normalize_quant_label,
    list_model_variants,
    summarize_step_csv,
    discover_predictions_file_from_json,
    fallback_find_step_csv,
    get_summary_output_path,
    run_inference_via_subprocess,
)



# ---- Projektpfad sicherstellen ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---- Imports aus bestehendem Projekt ----
try:
    from config.config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG  # type: ignore
except ModuleNotFoundError:
    from config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG  # type: ignore

try:
    from ML_Helpfunctions import pipeline_utils as PU  # type: ignore
except ModuleNotFoundError:
    import importlib
    PU = importlib.import_module('pipeline_utils')

# ---- Trainer-Klassen je Algorithmus ----
TRAINER_MAP = {
    "lstm": ("ML_Algorithms.LSTM.lstm_train", "LSTMTrainer", "LSTM"),
    "cnn1d": ("ML_Algorithms.CNN1D.cnn1d_train", "CNN1DTrainer", "CNN1D"),
    "random_forest": ("ML_Algorithms.Random_Forest.rf_train", "RandomForestTrainer", "Random_Forest"),
    "xgboost": ("ML_Algorithms.XGBOOST.xgboost_train", "XGBoostTrainer", "XGBOOST"),
    "light_xgboost": ("ML_Algorithms.Light_XGBOOST.light_xgboost_train", "LightXGBoostTrainer", "Light_XGBOOST"),
}

# --- NEUE HYPERPARAMETER (aus CSV integriert am 28.08.2025) ---
COMPLEXITY_PRESETS = {
  'lstm': {
    'simple': {'dropout': 0.3502, 'batch_size': 64, 'epochs': 100, 'learning_rate': 0.0001587, 'optimizer': 'nadam', 'loss': 'huber', 'clipnorm': 0.7164, 'model_params': {'dropout': 0.3502, 'batch_size': 64, 'epochs': 100, 'learning_rate': 0.0001587, 'optimizer': 'nadam', 'loss': 'huber', 'clipnorm': 0.7164, 'num_layers':1, 'initial_units':32}},
    'medium': {'dropout': 0.4091, 'batch_size': 32, 'epochs': 40, 'learning_rate': 0.000803, 'optimizer': 'adam', 'loss': 'huber', 'clipnorm': 2.5225, 'model_params': {'dropout': 0.4091, 'batch_size': 32, 'epochs': 40, 'learning_rate': 0.000803, 'optimizer': 'adam', 'loss': 'huber', 'clipnorm': 2.5225, 'num_layers':2, 'initial_units':64}},
    'high':   {'dropout': 0.3120, 'batch_size': 32, 'epochs': 40, 'learning_rate': 0.003931, 'optimizer': 'rmsprop', 'loss': 'mse', 'clipnorm': 4.9347, 'model_params': {'dropout': 0.3120, 'batch_size': 32, 'epochs': 40, 'learning_rate': 0.003931, 'optimizer': 'rmsprop', 'loss': 'mse', 'clipnorm': 4.9347, 'num_layers':3, 'initial_units':96}},
  },
  'cnn1d': {
    'simple': {'batch_size': 128, 'epochs': 40, 'learning_rate': 0.001296, 'optimizer': 'rmsprop', 'loss': 'mse', 'clipnorm': 1.0063, 'cnn_dropout': 0.2048, 'cnn_activation': 'gelu', 'model_params': {'batch_size': 128, 'epochs': 40, 'learning_rate': 0.001296, 'optimizer': 'rmsprop', 'loss': 'mse', 'clipnorm': 1.0063, 'cnn_dropout': 0.2048, 'cnn_activation': 'gelu', 'cnn_blocks':1, 'cnn_base_filters':32, 'cnn_kernel_size':3}},
    'medium': {'batch_size': 128, 'epochs': 90, 'learning_rate': 0.000210, 'optimizer': 'rmsprop', 'loss': 'mse', 'clipnorm': 1.7381, 'cnn_dropout': 0.1939, 'cnn_activation': 'tanh', 'model_params': {'batch_size': 128, 'epochs': 90, 'learning_rate': 0.000210, 'optimizer': 'rmsprop', 'loss': 'mse', 'clipnorm': 1.7381, 'cnn_dropout': 0.1939, 'cnn_activation': 'tanh', 'cnn_blocks':2, 'cnn_base_filters':64, 'cnn_kernel_size':5}},
    'high':   {'batch_size': 32, 'epochs': 90, 'learning_rate': 0.000346, 'optimizer': 'nadam', 'loss': 'huber', 'clipnorm': 2.6112, 'cnn_dropout': 0.00473, 'cnn_activation': 'relu', 'model_params': {'batch_size': 32, 'epochs': 90, 'learning_rate': 0.000346, 'optimizer': 'nadam', 'loss': 'huber', 'clipnorm': 2.6112, 'cnn_dropout': 0.00473, 'cnn_activation': 'relu', 'cnn_blocks':3, 'cnn_base_filters':96, 'cnn_kernel_size':7}},
  },
  'random_forest': {
    'simple': {'min_samples_split': 16, 'min_samples_leaf': 1, 'max_features': 0.6052, 'bootstrap': False, 'n_jobs':-1, 'random_state':42, 'model_params': {'min_samples_split': 16, 'min_samples_leaf': 1, 'max_features': 0.6052, 'bootstrap': False, 'n_estimators': 120, 'max_depth': 6}},
    'medium': {'min_samples_split': 2, 'min_samples_leaf': 6, 'max_features': 0.5581, 'bootstrap': False, 'n_jobs':-1, 'random_state':42, 'model_params': {'min_samples_split': 2, 'min_samples_leaf': 6, 'max_features': 0.5581, 'bootstrap': False, 'n_estimators': 280, 'max_depth': 10}},
    'high':   {'min_samples_split': 12, 'min_samples_leaf': 7, 'max_features': 0.8435, 'bootstrap': False, 'n_jobs':-1, 'random_state':42, 'model_params': {'min_samples_split': 12, 'min_samples_leaf': 7, 'max_features': 0.8435, 'bootstrap': False, 'n_estimators': 400, 'max_depth': 12}},
  },
  'xgboost': {
    'simple': {'learning_rate': 0.005, 'subsample': 0.8129, 'colsample_bytree': 0.4073, 'min_child_weight': 15, 'gamma': 4.3521, 'reg_lambda': 0.0156, 'reg_alpha': 0.0014, 'tree_method':'hist', 'n_jobs':-1, 'random_state':42, 'objective':'reg:squarederror', 'xgb_params': {'learning_rate': 0.005, 'subsample': 0.8129, 'colsample_bytree': 0.4073, 'min_child_weight': 15, 'gamma': 4.3521, 'reg_lambda': 0.0156, 'reg_alpha': 0.0014, 'n_estimators': 200, 'max_depth': 3}},
    'medium': {'learning_rate': 0.0145, 'subsample': 0.8846, 'colsample_bytree': 0.9263, 'min_child_weight': 4, 'gamma': 2.4491, 'reg_lambda': 0.0549, 'reg_alpha': 1.798e-6, 'tree_method':'hist', 'n_jobs':-1, 'random_state':42, 'objective':'reg:squarederror', 'xgb_params': {'learning_rate': 0.0145, 'subsample': 0.8846, 'colsample_bytree': 0.9263, 'min_child_weight': 4, 'gamma': 2.4491, 'reg_lambda': 0.0549, 'reg_alpha': 1.798e-6, 'n_estimators': 400, 'max_depth': 5}},
    'high':   {'learning_rate': 0.0332, 'subsample': 0.9226, 'colsample_bytree': 0.4054, 'min_child_weight': 6, 'gamma': 4.1146, 'reg_lambda': 0.4810, 'reg_alpha': 0.0076, 'tree_method':'hist', 'n_jobs':-1, 'random_state':42, 'objective':'reg:squarederror', 'xgb_params': {'learning_rate': 0.0332, 'subsample': 0.9226, 'colsample_bytree': 0.4054, 'min_child_weight': 6, 'gamma': 4.1146, 'reg_lambda': 0.4810, 'reg_alpha': 0.0076, 'n_estimators': 600, 'max_depth': 6}},
  },
  'light_xgboost': {
    'simple': {'learning_rate': 0.0094, 'bagging_fraction': 0.9910, 'feature_fraction': 0.6422, 'min_child_samples': 2, 'reg_lambda': 0.0110, 'reg_alpha': 3.081e-6, 'max_bin': 256, 'n_jobs':-1, 'random_state':42, 'objective':'regression', 'lgbm_params': {'learning_rate': 0.0094, 'bagging_fraction': 0.9910, 'feature_fraction': 0.6422, 'min_child_samples': 2, 'reg_lambda': 0.0110, 'reg_alpha': 3.081e-6, 'max_bin': 256, 'n_estimators': 100, 'num_leaves': 198}},
    'medium': {'learning_rate': 0.0060, 'bagging_fraction': 0.8470, 'feature_fraction': 0.4016, 'min_child_samples': 13, 'reg_lambda': 0.0039, 'reg_alpha': 1.996e-5, 'max_bin': 224, 'n_jobs':-1, 'random_state':42, 'objective':'regression', 'lgbm_params': {'learning_rate': 0.0060, 'bagging_fraction': 0.8470, 'feature_fraction': 0.4016, 'min_child_samples': 13, 'reg_lambda': 0.0039, 'reg_alpha': 1.996e-5, 'max_bin': 224, 'n_estimators': 200, 'num_leaves': 108}},
    'high':   {'learning_rate': 0.0233, 'bagging_fraction': 0.7452, 'feature_fraction': 0.8878, 'min_child_samples': 11, 'reg_lambda': 0.0028, 'reg_alpha': 0.0026, 'max_bin': 160, 'n_jobs':-1, 'random_state':42, 'objective':'regression', 'lgbm_params': {'learning_rate': 0.0233, 'bagging_fraction': 0.7452, 'feature_fraction': 0.8878, 'min_child_samples': 11, 'reg_lambda': 0.0028, 'reg_alpha': 0.0026, 'max_bin': 160, 'n_estimators': 300, 'num_leaves': 126}},
  },
}

# --- Basis-Defaults, die für alle Modelle gelten ---
BASE_COMMON = {
    "dataset": "mqtt_data_filtered.csv",
    "train_fraction": 0.85,
    "base_features": ["Group4-2_S6_VolumetricFlowRate", "Group4-2_S6_MassFlowRate"],
    "time_features": [],
    "target_feature": "Group4-2_S6_VolumetricFlowRate",
    "scale_other_features": True,
    "scale_target": True,
    "scaler_type": "robust",
    "edge_device": True,
    "enable_edge": True,
    "validation_fraction": 0.2,
    "early_stopping_patience": 10,
    "rolling_window_size": 10, # Fest auf 10 gesetzt
    "lags": 20, # Fest auf 20 gesetzt
}

# Modell-Dateien, die als "groß" gelten und nach der Inferenz entfernt werden dürfen
MODEL_BLOBS_WHITELIST = {
    "keras": ["model.keras"],
    "tflite": ["model_quant_float16.tflite", "model_quant_int8.tflite", "model_quant_int8_full.tflite"],
    "sklearn": ["model.joblib"],
    "xgb": ["model.json"],
}
SUMMARY_CSV_NAME = "Experiment_Summary_Server_multiconfig_run_merged.csv"
SCALER_FILE_NAMES = ["scaler.joblib", "y_scaler.joblib"]

# ---------------------------
# Hilfs-Datenstrukturen
# ---------------------------
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
    quant_mode: Optional[str]
    run_id: str


# ---------------------------
# Utils
# ---------------------------
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

def _range_from_str(spec: str) -> List[int]:
    spec = spec.strip()
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        start, stop, step = (parts[0], parts[1], 1) if len(parts) == 2 else parts
        return list(range(start, stop + (1 if step > 0 else -1), step))
    return [int(x) for x in spec.split(",") if x]

def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (ValueError, TypeError):
        return None

def algorithm_to_folder(name_or_flag: str) -> str:
    n = (name_or_flag or "").lower()
    if "light_xgboost" in n: return "Light_XGBOOST"
    if "lstm" in n: return "LSTM"
    if "cnn" in n: return "CNN1D"
    if "xgb" in n: return "XGBOOST"
    if "random_forest" in n: return "Random_Forest"
    return name_or_flag.upper() or "MODEL"


# ---------------------------
# Konfigurationsaufbau
# ---------------------------
def build_training_config(algorithm: str, level: str, horizon: int, folder_flag: str, quant_modes: list) -> dict:
    """Mergt allgemeine Pfade/Flags, setzt Horizon, Komplexitäts-Level etc."""
    algo = algorithm.lower()
    if algo not in COMPLEXITY_PRESETS or level not in COMPLEXITY_PRESETS[algo]:
        raise ValueError(f"Keine Presets für Algorithmus '{algo}' mit Level '{level}' gefunden.")

    preset_cfg = COMPLEXITY_PRESETS[algo][level]
    merged = _deep_merge(BASE_COMMON, preset_cfg)

    # --- ANPASSUNG FÜR BAUM-MODELLE ---
    # Baum-basierte Modelle (RF, XGB) benötigen i.d.R. weder Ziel- noch Feature-Skalierung.
    if algo in ["random_forest", "xgboost", "light_xgboost"]:
        merged["scale_target"] = False
        merged["scale_other_features"] = False # <-- DIESE ZEILE HINZUFÜGEN

    runtime_cfg = {
        "paths": CONFIG_PATH["paths"],
        "inference_mode": "load_artifacts_path",
        "horizon": int(horizon),
        "quant_modes": quant_modes,
        # FIX: model_name-Schlüssel hinzugefügt, um den KeyError zu beheben
        "model_name": f"{algo}_{level}"
    }
    merged = _deep_merge(merged, runtime_cfg)
    merged = _deep_merge(merged, CONFIG_LOAD_ARTIFACTS)
    merged = _deep_merge(merged, MQTT_CONFIG)

    merged, _ = PU.setup_experiment(merged, folder_flag, run_type="train")
    return merged


# ---------------------------
# Training (programmatisch)
# ---------------------------
def run_training(algorithm: str, config: dict, folder_flag: str) -> Tuple[str, Path]:
    """Startet das programmatische Training und gibt (run_id, models_dir) zurück."""
    module, clsname, default_flag = TRAINER_MAP[algorithm]
    Trainer = _try_import(module, clsname)
    use_flag = folder_flag or default_flag
    t0 = time.perf_counter()
    trainer = Trainer(config=config, folder_flag=use_flag)
    trainer.run(save_artifacts=True)
    dur_s = time.perf_counter() - t0

    try:
        training_cfg_json = Path(config["paths"]["Models"]) / "training_config.json"
        if training_cfg_json.exists():
            with open(training_cfg_json, "r", encoding="utf-8") as f:
                js = json.load(f)
            js["training_time_s"] = round(float(dur_s), 3)
            with open(training_cfg_json, "w", encoding="utf-8") as f:
                json.dump(js, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"?? Konnte training_time_s nicht schreiben: {e}")

    run_id = str(config.get("run_id"))
    models_dir = Path(config["paths"]["Models"])
    return run_id, models_dir


# ---------------------------
# Inferenz
# ---------------------------
def _import_pipeline_web_app_path() -> Path:
    """Findet den Pfad zu pipeline_web_app.py robust."""
    import importlib.util as _ilu
    candidates = [PROJECT_ROOT / 'pipeline_web_app.py', PROJECT_ROOT / 'ML_Algorithms' / 'pipeline_web_app.py']
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"pipeline_web_app.py nicht in erwarteten Pfaden gefunden: {candidates}")

def run_inference_via_subprocess(
    algorithm: str, run_id: str, model_filename: str, inference_steps: int,
    loading_strategy: str, interval_sec: float
) -> int:
    """Ruft pipeline_web_app.py für eine reine Inferenz auf."""
    try:
        app_path = _import_pipeline_web_app_path()
    except FileNotFoundError as e:
        print(f"? {e}")
        return 2

    cmd = [
        sys.executable, str(app_path),
        "--algorithm", algorithm,
        "--load_id", run_id,
        "--model_filename", model_filename,
        "--no-web",
        "--inference-steps", str(inference_steps),
        "--set", f"loading_strategy={loading_strategy}",
        "--set", f"inference_interval_sec={interval_sec}",
    ]
    print("[SPAWN] ", " ".join(cmd))
    import subprocess
    proc = subprocess.run(cmd)
    return proc.returncode


# ---------------------------
# Auswertung & Speicherung
# ---------------------------
def list_model_variants(models_dir: Path) -> List[str]:
    """Findet vorhandene Modellvarianten in einem Models-Ordner."""
    found = set()
    for category in MODEL_BLOBS_WHITELIST.values():
        for filename in category:
            if (models_dir / filename).exists():
                found.add(filename)
    return sorted(list(found))

def _discover_predictions_file_from_json(run_id: str, error_metrics_dir: Path) -> Optional[Path]:
    """Liest ErrorMetrics_all_runs.csv und extrahiert den Pfad zur Step-Prediction-Datei."""
    agg_csv = error_metrics_dir / "ErrorMetrics_all_runs.csv"
    if not agg_csv.exists(): return None
    try:
        import pandas as pd
        df = pd.read_csv(agg_csv)
        row = df[df["run_id"] == run_id].tail(1)
        if row.empty: return None
        json_path = Path(row.iloc[0]["json_path"])
        if not json_path.exists(): return None
        with open(json_path, "r", encoding="utf-8") as f:
            js = json.load(f)
        p = js.get("extra_info", {}).get("predictions_file_path")
        return Path(p) if p else None
    except Exception:
        return None

def _fallback_find_step_csv(run_id: str, prediction_data_dir: Path) -> Optional[Path]:
    hits = sorted(list(prediction_data_dir.glob(f"StepPredictions_{run_id}_*.csv")))
    return hits[-1] if hits else None

def summarize_step_csv(step_csv: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Gibt Ø inference_time_ms, Ø total_time_ms, Ø cpu_percent, Ø ram_percent zurück."""
    try:
        import pandas as pd
        df = pd.read_csv(step_csv)
        inf_ms = _safe_float(df["inference_time_s"].mean() * 1000.0) if "inference_time_s" in df else None
        tot_ms = _safe_float(df["total_time_s"].mean() * 1000.0) if "total_time_s" in df else None
        cpu = _safe_float(df["cpu_percent"].mean()) if "cpu_percent" in df else None
        ram = _safe_float(df["ram_percent"].mean()) if "ram_percent" in df else None
        return inf_ms, tot_ms, cpu, ram
    except Exception:
        return None, None, None, None

def read_model_size_mb(training_config_json: Path, model_variant_file: str) -> Optional[float]:
    """
    Liest die spezifische Modellgröße (MB) aus training_config.json.
    Bevorzugt 'model_sizes_mb[<Dateiname>]', fällt zurück auf 'model_size_MB'.
    """
    if not training_config_json.exists():
        return None
    try:
        import os, json
        with open(training_config_json, "r", encoding="utf-8") as f:
            js = json.load(f)

        sizes_dict = js.get("model_sizes_mb") or js.get("model_sizes_MB") or {}
        if isinstance(sizes_dict, dict):
            key = os.path.basename(model_variant_file)
            if key in sizes_dict:
                return _safe_float(sizes_dict[key])
        
        return _safe_float(js.get("model_size_MB"))
    except Exception:
        return None


def _map_variant_to_quant_mode(model_file: str) -> str:
    if "int8_full" in model_file: return "quant-8"
    if "int8" in model_file: return "quant-8"
    if "float16" in model_file: return "quant-16"
    return "no-quant"

def append_summary_row(summary_csv: Path, row: InferenceResult) -> None:
    header = [
        "algorithm", "level", "lags", "horizon", "model_variant", "quant_mode",
        "avg_inference_time_ms", "avg_total_time_ms", "avg_cpu_percent", "avg_ram_percent",
        "model_size_mb", "run_id",
    ]
    exists = summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not exists: writer.writerow(header)
        writer.writerow([
            row.algorithm, row.level, row.lags, row.horizon, row.model_variant, row.quant_mode,
            row.avg_inference_time_ms, row.avg_total_time_ms, row.avg_cpu_percent, row.avg_ram_percent,
            row.model_size_mb, row.run_id,
        ])


# ---------------------------
# Aufräumen
# ---------------------------
def cleanup_model_binaries_and_scalers(models_dir: Path, scalers_dir: Path) -> None:
    """Löscht große Modellbinaries und Scaler-Dateien."""
    to_delete = []
    for category in MODEL_BLOBS_WHITELIST.values():
        for filename in category:
            fp = models_dir / filename
            if fp.exists(): to_delete.append(fp)

    for fname in SCALER_FILE_NAMES:
        fp = scalers_dir / fname
        if fp.exists(): to_delete.append(fp)

    for fp in sorted(to_delete):
        try:
            fp.unlink()
            print(f"?? Gelöscht: {fp}")
        except Exception as e:
            print(f"?? Fehler beim Löschen von {fp}: {e}")


# ---------------------------
# Orchestrierung
# ---------------------------
def _file_candidates_for_mode(algo: str, mode: str) -> list[str]:
    mode = (mode or "no-quant").lower()
    if mode == "quant-16": return ["model_quant_float16.tflite"]
    if mode == "quant-8": return ["model_quant_int8_full.tflite", "model_quant_int8.tflite"]
    if algo in ("lstm", "cnn1d"): return ["model.keras"]
    if algo == "random_forest": return ["model.joblib"]
    if algo in ("xgboost", "light_xgboost"): return ["model.json", "model.joblib"]
    return []

def _run_all_inferences_and_summarize(
    algo: str, cfg: dict, run_id: str, models_dir: Path, inference_steps: int,
    loading_strategy: str, interval_sec: float, summary_csv: Path, delete_models: bool
) -> None:
    """Führt für jeden gewünschten/verfügbaren Quantisierungsmodus eine Inferenz aus und schreibt die Summary-Zeilen."""
    variants_present = set(list_model_variants(models_dir))
    if not variants_present:
        print("?? Kein Modell gefunden – Inferenz für diesen Lauf übersprungen.")
        return

    modes_to_run = cfg.get("quant_modes", ["no-quant"])
    queue: list[tuple[str, str]] = []
    for qmode in modes_to_run:
        candidates = _file_candidates_for_mode(algo, qmode)
        chosen = next((f for f in candidates if f in variants_present), None)
        if chosen and (qmode, chosen) not in queue:
            queue.append((qmode, chosen))
        else:
            print(f"?? Überspringe Modus '{qmode}': keine passende Datei gefunden (gesucht: {candidates})")
    
    if not queue:
        best_available = next((f for f in ["model.keras", "model.joblib", "model.json"] if f in variants_present), None)
        if best_available:
            queue.append(("no-quant", best_available))

    for qmode, model_file in queue:
        print(f"\n? Starte Inferenz für Modus '{qmode}' mit Datei '{model_file}'.")
        rc = run_inference_via_subprocess(
            algorithm=algo, run_id=run_id, model_filename=model_file,
            inference_steps=inference_steps, loading_strategy=loading_strategy, interval_sec=interval_sec
        )
        if rc != 0:
            print(f"?? Inferenz-Subprozess mit Fehlercode {rc} für {model_file} fehlgeschlagen.")
            res = InferenceResult(
                algorithm=algo, level=cfg.get("level_used", "unknown"),
                lags=int(cfg.get("lags", 0)), horizon=int(cfg.get("horizon", 0)),
                model_variant=f"FAILED: {model_file}", quant_mode=qmode,
                avg_inference_time_ms=None, avg_total_time_ms=None,
                avg_cpu_percent=None, avg_ram_percent=None,
                model_size_mb=None, run_id=run_id,
            )
            append_summary_row(summary_csv, res)
            continue

        err_dir = Path(cfg["paths"]["Error_Metrics"])
        pred_dir = Path(cfg["paths"]["Prediction_Data"])
        step_csv = _discover_predictions_file_from_json(run_id, err_dir) or _fallback_find_step_csv(run_id, pred_dir)

        avg_inf_ms = avg_total_ms = avg_cpu = avg_ram = None
        if step_csv and step_csv.exists():
            avg_inf_ms, avg_total_ms, avg_cpu, avg_ram = summarize_step_csv(step_csv)
        else:
            print("⚠️ StepPredictions CSV nicht gefunden – Zusammenfassung unvollständig.")

        training_cfg_json = models_dir / "training_config.json"
        model_size_mb = read_model_size_mb(training_cfg_json, model_file)

        res = InferenceResult(
            algorithm=algo, level=cfg.get("level_used", "unknown"),
            lags=int(cfg.get("lags", 0)), horizon=int(cfg.get("horizon", 0)),
            model_variant=model_file, quant_mode=_map_variant_to_quant_mode(model_file),
            avg_inference_time_ms=avg_inf_ms, avg_total_time_ms=avg_total_ms,
            avg_cpu_percent=avg_cpu, avg_ram_percent=avg_ram,
            model_size_mb=model_size_mb, run_id=run_id,
        )
        append_summary_row(summary_csv, res)
        print(f"?? Zusammenfassung für {res.quant_mode} ({model_file}) in CSV geschrieben.")

    if delete_models:
        cleanup_model_binaries_and_scalers(models_dir, Path(cfg["paths"]["Scalers"]))


def run_experiments(
    algorithms: List[str], horizon_values: List[int], inference_steps: int,
    loading_strategy: str, interval_sec: float, delete_models: bool, limit_runs: Optional[int]
) -> None:
    out_dir = Path(CONFIG_PATH["paths"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / SUMMARY_CSV_NAME

    runs_done = 0
    levels = ["simple", "medium", "high"]

    for algorithm in algorithms:
        algo = algorithm.lower()
        if algo not in TRAINER_MAP:
            print(f"?? Unbekannter Algorithmus '{algorithm}' wird übersprungen.")
            continue

        quant_modes_for_algo = ["no-quant", "quant-16", "quant-8"] if algo in ["cnn1d", "lstm"] else ["no-quant"]
        print(f"\n--- Algorithmus: {algo} (Quantisierungsmodi: {quant_modes_for_algo}) ---")

        for horizon in horizon_values:
            for level in levels:
                if limit_runs is not None and runs_done >= limit_runs:
                    print("Maximalzahl an Läufen erreicht – Experiment wird beendet.")
                    return

                folder_flag = algorithm_to_folder(algo)
                cfg = build_training_config(algo, level, horizon, folder_flag, quant_modes_for_algo)
                cfg["level_used"] = level

                print(f"\n=== LAUF {runs_done + 1} | Train: {algo} | Level: {level} | Horizon: {horizon} ===")
                run_id, models_dir = run_training(algo, cfg, folder_flag)
                print(f"? Training abgeschlossen. run_id={run_id}")

                _run_all_inferences_and_summarize(
                    algo, cfg, run_id, models_dir,
                    inference_steps, loading_strategy, interval_sec,
                    summary_csv, delete_models
                )
                runs_done += 1


def main():
    p = argparse.ArgumentParser(description="Experiment-Pipeline über Modelle, Komplexitätsstufen und Horizons")
    p.add_argument("--algorithms", default="cnn1d,lstm,random_forest,xgboost,light_xgboost", help="Kommagetrennte Liste der Algorithmen")
    p.add_argument("--horizon", default="1:16:2", help="Range 'start:stop:step' oder kommagetrennt für den Horizont")
    p.add_argument("--inference-steps", type=int, default=60, help="Anzahl Inferenzschritte")
    p.add_argument("--loading-strategy", default="split", choices=["split", "live_mqtt"], help="Datenquelle für Inferenz")
    p.add_argument("--interval-sec", type=float, default=0.0, help="Ziel-Inferenzintervall in Sekunden")
    p.add_argument("--keep-models", action="store_true", help="Modellbinaries nach Inferenz NICHT löschen")
    p.add_argument("--limit-runs", type=int, help="Max. Anzahl an Experimenten (Kombinationen aus Algo/Level/Horizon)")
    args = p.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    horizon_vals = _range_from_str(args.horizon)

    run_experiments(
        algorithms=algorithms,
        horizon_values=horizon_vals,
        inference_steps=args.inference_steps,
        loading_strategy=args.loading_strategy,
        interval_sec=args.interval_sec,
        delete_models=(not args.keep_models),
        limit_runs=args.limit_runs,
    )

if __name__ == "__main__":
    main()