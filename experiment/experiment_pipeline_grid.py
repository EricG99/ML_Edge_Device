#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment Grid Pipeline for RevPi
----------------------------------

Purpose:
  * Systematically runs training and inference experiments across a grid of
    pre-defined model complexities (C1-C6) and feature complexities (F1-F3).
  * Extends the base `experiment_pipeline.py` to automate comprehensive
    benchmarking on edge devices like the Revolution Pi.
  * Captures an expanded set of metrics including detailed latency breakdowns
    (preprocessing, postprocessing), model load time, parameter count, and
    placeholders for energy consumption (Watt, Joule).
  * Features robust error handling to prevent the entire pipeline from crashing
    if a single experimental run fails (e.g., due to memory overflow).
  * Includes an optional flag (`--quantize`) to enable post-training
    quantization for TFLite-compatible models.
  * Persists all results in a single, comprehensive CSV file for later analysis.

Methodology:
  1. Defines complexity levels for models (e.g., layers, units, estimators)
     and features (e.g., number of base features, lags, rolling windows).
  2. Iterates through every combination of algorithm, profile, model complexity,
     feature complexity, and prediction horizon.
  3. For each combination:
     a. Builds a temporary training configuration by merging base, feature, and
        model complexity parameters.
     b. Executes a programmatic training run using the corresponding Trainer class.
     c. Runs inference for all available model variants (e.g., .keras, .tflite)
        via a subprocess call to `pipeline_web_app.py`.
     d. Parses the detailed `StepPredictions_*.csv` output to calculate
        average and percentile metrics.
     e. Reads metadata like parameter count from the saved `training_config.json`.
     f. Appends a summary row with all metrics to `Experiment_Grid_Summary.csv`.
     g. Cleans up large model binaries to conserve disk space.
  4. Each run is wrapped in a try-except block to ensure pipeline continuity.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import subprocess
import pandas as pd

# ---- Project Path Setup ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---- Core Application Imports ----
# These are assumed to exist based on the provided file context.
try:
    from config.config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG
    from ML_Helpfunctions import Pipeline_Utils as PU
    # Dynamically import the config loader from pipeline_web_app.py
    from ML_Algorithms.pipeline_web_app import load_config_dynamically
except (ImportError, ModuleNotFoundError) as e:
    print(f"FATAL: Could not import necessary project modules: {e}", file=sys.stderr)
    print("Please ensure this script is run from a location where 'config' and 'ML_Helpfunctions' are accessible.", file=sys.stderr)
    sys.exit(1)


# ---- Trainer Class Mapping ----
TRAINER_MAP = {
    "lstm": ("ML_Algorithms.LSTM.LSTM_train", "LSTMTrainer", "LSTM"),
    "cnn1d": ("ML_Algorithms.CNN1D.cnn1d_train", "CNN1DTrainer", "CNN1D"),
    "random_forest": ("ML_Algorithms.Random_Forest.rf_train", "RandomForestTrainer", "Random_Forest"),
    "xgboost": ("ML_Algorithms.XGBOOST.XGBOOST_train", "XGBoostTrainer", "XGBOOST"),
    "light_xgboost": ("ML_Algorithms.XGBOOST.XGBOOST_train", "XGBoostTrainer", "LIGHT_XGBOOST"),
}

# ---- Model & Profile Configuration Mapping ----
DEFAULT_PROFILE_VARS: Dict[str, Dict[str, str]] = {
    "lstm": {"server": "param_lstm_server", "edge": "param_lstm_edge"},
    "cnn1d": {"server": "param_cnn1d_server", "edge": "param_cnn1d_edge"},
    "random_forest": {"server": "random_forest_server", "edge": "random_forest_edge"},
    "xgboost": {"server": "xgboost_server", "edge": "xgboost_edge"},
    "light_xgboost": {"server": "light_xgboost_server", "edge": "light_xgboost_edge"},
}

MODEL_BLOBS_WHITELIST = {
    "keras": ["model.keras"],
    "tflite": ["model_quant_float16.tflite", "model_quant_int8.tflite", "model_quant_int8_full.tflite"],
    "sklearn": ["model.joblib"],
    "xgb": ["model.json"],
}

# ---- Experiment Grid Definitions ----
SUMMARY_CSV_NAME = "Experiment_Grid_Summary.csv"

MODEL_COMPLEXITY_LEVELS = {
    "lstm": {
        "C1": {"lags": 4, "model_params": {"lstm_units": [16], "dropout_rate": 0.0}, "epochs": 10},
        "C2": {"lags": 8, "model_params": {"lstm_units": [32], "dropout_rate": 0.1}, "epochs": 20},
        "C3": {"lags": 12, "model_params": {"lstm_units": [64, 64], "dropout_rate": 0.2}, "epochs": 30},
        "C4": {"lags": 16, "model_params": {"lstm_units": [128, 128], "dropout_rate": 0.2}, "epochs": 50},
        "C5": {"lags": 20, "model_params": {"lstm_units": [192, 192, 192], "dropout_rate": 0.3}, "epochs": 70},
        "C6": {"lags": 24, "model_params": {"lstm_units": [256, 256, 256], "dropout_rate": 0.3}, "epochs": 90},
    },
    "cnn1d": {
        "C1": {"lags": 4, "model_params": {"filters": [16], "kernel_size": 3}},
        "C2": {"lags": 8, "model_params": {"filters": [32, 32], "kernel_size": 3}},
        "C3": {"lags": 12, "model_params": {"filters": [64, 64, 64], "kernel_size": 5}},
        "C4": {"lags": 16, "model_params": {"filters": [128, 128, 128, 128], "kernel_size": 5}},
        "C5": {"lags": 20, "model_params": {"filters": [192, 192, 192, 192, 192], "kernel_size": 7}},
        "C6": {"lags": 24, "model_params": {"filters": [256, 256, 256, 256, 256, 256], "kernel_size": 7}},
    },
    "random_forest": {
        "C1": {"model_params": {"n_estimators": 50, "max_depth": 6}},
        "C2": {"model_params": {"n_estimators": 100, "max_depth": 10}},
        "C3": {"model_params": {"n_estimators": 200, "max_depth": 14}},
        "C4": {"model_params": {"n_estimators": 400, "max_depth": 18}},
        "C5": {"model_params": {"n_estimators": 800, "max_depth": None}},
        "C6": {"model_params": {"n_estimators": 1200, "max_depth": None}},
    },
    "xgboost": { # Also used for light_xgboost
        "C1": {"model_params": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10}},
        "C2": {"model_params": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.07}},
        "C3": {"model_params": {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05}},
        "C4": {"model_params": {"n_estimators": 800, "max_depth": 10, "learning_rate": 0.04}},
        "C5": {"model_params": {"n_estimators": 1200, "max_depth": 12, "learning_rate": 0.03}},
        "C6": {"model_params": {"n_estimators": 1600, "max_depth": 14, "learning_rate": 0.025}},
    }
}

FEATURE_COMPLEXITY_LEVELS = {
    "F1": {
        "lags": 4,
        "base_features": ["value"],
        "rolling_windows": []
    },
    "F2": {
        "lags": 12,
        "base_features": ["value", "feature1"], # Assuming feature1 exists
        "rolling_windows": [{"window_size": 3, "statistic": "mean"}, {"window_size": 5, "statistic": "mean"}]
    },
    "F3": {
        "lags": 20,
        "base_features": ["value", "feature1", "feature2"], # Assuming feature1, feature2 exist
        "rolling_windows": [
            {"window_size": 3, "statistic": "mean"}, {"window_size": 5, "statistic": "mean"}, {"window_size": 7, "statistic": "mean"},
            {"window_size": 3, "statistic": "std"}, {"window_size": 5, "statistic": "std"}, {"window_size": 7, "statistic": "std"}
        ]
    }
}

# ---- Data Structures ----
@dataclass
class InferenceResult:
    # Experiment identifiers
    algorithm: str
    profile: str
    model_complexity: str
    feature_complexity: str
    horizon: int
    model_variant: str
    run_id: str
    # Performance Metrics
    avg_inference_time_ms: Optional[float]
    p95_inference_time_ms: Optional[float]
    avg_total_time_ms: Optional[float]
    avg_preprocess_time_ms: Optional[float]
    avg_postprocess_time_ms: Optional[float]
    model_load_time_s: Optional[float]
    # Resource Metrics
    avg_cpu_percent: Optional[float]
    avg_ram_percent: Optional[float]
    model_size_mb: Optional[float]
    param_count: Optional[int]
    # Energy Metrics (Placeholders)
    avg_power_w: Optional[float] = None
    total_energy_j: Optional[float] = None
    # Accuracy Metrics (Example, add more as needed)
    mae: Optional[float] = None
    rmse: Optional[float] = None


# ---- Utility Functions ----
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
        return list(range(start, stop + 1, step))
    return [int(x) for x in spec.split(",") if x]

def _safe_float(v) -> Optional[float]:
    try: return float(v)
    except (ValueError, TypeError): return None

def _safe_int(v) -> Optional[int]:
    try: return int(v)
    except (ValueError, TypeError): return None

# ---- Configuration & Orchestration ----
def load_profile_config(algorithm: str, profile: str) -> Tuple[dict, str]:
    """Loads a profile config, with fallbacks for base and light_xgboost."""
    algo = algorithm.lower()
    varname = DEFAULT_PROFILE_VARS.get(algo, {}).get(profile)
    cfg, used = None, None
    if varname:
        try:
            cfg = load_config_dynamically(algo, varname)
            used = varname
        except SystemExit: pass

    if cfg is None:
        try:
            cfg = load_config_dynamically(algo, algo)
            used = algo
        except SystemExit: pass

    if cfg is None and algo == "light_xgboost":
        base_var = DEFAULT_PROFILE_VARS.get("xgboost", {}).get(profile, "xgboost")
        cfg = load_config_dynamically("xgboost", base_var)
        used = f"{base_var} (via light_xgboost)"

    if cfg is None:
        raise ValueError(f"Could not load any configuration for algorithm '{algorithm}' and profile '{profile}'.")

    return cfg, (used or algo)


def build_training_config(
    base_cfg: dict, profile: str, horizon: int, folder_flag: str,
    feature_params: dict, model_params: dict, quantize: bool
) -> dict:
    """Merges all config layers for a specific experiment run."""
    # Layer 1: Base paths and flags
    merged = {"paths": CONFIG_PATH["paths"], "inference_mode": "load_artifacts_path"}
    # Layer 2: Base config for the algorithm/profile
    merged = _deep_merge(merged, base_cfg)
    # Layer 3: Feature complexity parameters (lags, features, etc.)
    merged = _deep_merge(merged, feature_params)
    # Layer 4: Model complexity parameters (layers, units, etc.)
    merged = _deep_merge(merged, model_params)
    # Layer 5: Specific run parameters (horizon, quantization)
    merged = _deep_merge(merged, {
        "horizon": int(horizon),
        "edge_device": (profile == "edge"),
        "enable_edge": (profile == "edge"),
        "enable_quantization": bool(quantize),
    })
    # Layer 6: General runtime values
    merged = _deep_merge(merged, CONFIG_LOAD_ARTIFACTS)
    merged = _deep_merge(merged, MQTT_CONFIG)

    # Final step: create versioned experiment folders
    final_config, _ = PU.setup_experiment(merged, folder_flag, run_type="train")
    return final_config


def run_training(algorithm: str, config: dict, folder_flag: str) -> Tuple[str, Path, float]:
    """Starts programmatic training and returns run_id, models_dir, and training time."""
    module, clsname, default_flag = TRAINER_MAP[algorithm]
    Trainer = _try_import(module, clsname)
    
    t0 = time.perf_counter()
    trainer = Trainer(config=config, folder_flag=folder_flag)
    trainer.run(save_artifacts=True)
    training_time_s = time.perf_counter() - t0

    run_id = str(config.get("run_id"))
    models_dir = Path(config["paths"]["Models"])
    return run_id, models_dir, training_time_s


def run_inference_via_subprocess(
    algorithm: str, run_id: str, model_filename: str, inference_steps: int,
    loading_strategy: str, interval_sec: float, config_name: str | None
) -> int:
    """Calls pipeline_web_app.py for a pure inference run."""
    py = sys.executable
    app_path = PROJECT_ROOT / 'pipeline_web_app.py'
    if not app_path.exists():
        app_path = PROJECT_ROOT / 'ML_Algorithms' / 'pipeline_web_app.py'
    if not app_path.exists():
        raise FileNotFoundError(f"Could not locate pipeline_web_app.py in standard locations.")

    cmd = [
        str(py), str(app_path),
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

    print(f"[SPAWN] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"--- SUBPROCESS FAILED (code {proc.returncode}) ---")
        print("STDOUT:", proc.stdout)
        print("STDERR:", proc.stderr)
        print("------------------------------------------")
    return proc.returncode


def list_model_variants(models_dir: Path) -> List[str]:
    """Finds all existing model binaries in a directory."""
    candidates = []
    all_blobs = (
        MODEL_BLOBS_WHITELIST["keras"] + MODEL_BLOBS_WHITELIST["tflite"] +
        MODEL_BLOBS_WHITELIST["sklearn"] + MODEL_BLOBS_WHITELIST["xgb"]
    )
    for p in all_blobs:
        if (models_dir / p).exists():
            candidates.append(p)
    return candidates or [x.name for x in models_dir.glob("*") if x.is_file()]


# ---- Results & Summary ----
def summarize_inference_run(run_id: str, config: dict) -> Dict[str, Optional[float]]:
    """
    Parses StepPredictions CSV and ErrorMetrics JSON to get all metrics.
    This is now the primary source for performance metrics.
    """
    results: Dict[str, Optional[float]] = {}
    pred_dir = Path(config["paths"]["Prediction_Data"])
    err_dir = Path(config["paths"]["Error_Metrics"])

    # Find the StepPredictions CSV file
    step_csv_files = list(pred_dir.glob(f"StepPredictions_{run_id}_*.csv"))
    if not step_csv_files:
        print(f"⚠️  No StepPredictions CSV found for run_id {run_id}")
        return results
    step_csv = sorted(step_csv_files)[-1] # Use the latest one

    try:
        df = pd.read_csv(step_csv).dropna(subset=['inference_time_s'])
        if not df.empty:
            results['avg_inference_time_ms'] = _safe_float(df["inference_time_s"].mean() * 1000.0)
            results['p95_inference_time_ms'] = _safe_float(df["inference_time_s"].quantile(0.95) * 1000.0)
            results['avg_total_time_ms'] = _safe_float(df.get("total_time_s", pd.Series(dtype=float)).mean() * 1000.0)
            results['avg_cpu_percent'] = _safe_float(df.get("cpu_percent", pd.Series(dtype=float)).mean())
            results['avg_ram_percent'] = _safe_float(df.get("ram_percent", pd.Series(dtype=float)).mean())
            # Add new time breakdowns if they exist in the CSV
            # NOTE: This assumes base_inference.py is modified to log these columns.
            if "preprocess_time_s" in df.columns:
                results['avg_preprocess_time_ms'] = _safe_float(df["preprocess_time_s"].mean() * 1000.0)
            if "postprocess_time_s" in df.columns:
                results['avg_postprocess_time_ms'] = _safe_float(df["postprocess_time_s"].mean() * 1000.0)
            if "power_w" in df.columns:
                 results['avg_power_w'] = _safe_float(df["power_w"].mean())
                 results['total_energy_j'] = _safe_float((df["power_w"] * df["total_time_s"]).sum())

    except Exception as e:
        print(f"⚠️ Error parsing step predictions CSV {step_csv}: {e}")

    # Find the ErrorMetrics JSON file to get accuracy metrics
    err_json_files = list(err_dir.glob(f"*_{run_id}.json"))
    if err_json_files:
        try:
            with open(err_json_files[-1], "r") as f:
                metrics_data = json.load(f)
            # Example: Extract MAE and RMSE for the whole horizon
            results['mae'] = _safe_float(metrics_data.get("metrics", {}).get("mae_mean"))
            results['rmse'] = _safe_float(metrics_data.get("metrics", {}).get("rmse_mean"))
        except Exception as e:
            print(f"⚠️ Error parsing error metrics JSON: {e}")

    return results


def read_training_metadata(models_dir: Path) -> Dict[str, Optional[float]]:
    """Reads model size and parameter count from the saved training_config.json."""
    meta = {'model_size_mb': None, 'param_count': None}
    config_path = models_dir / "training_config.json"
    if not config_path.exists():
        return meta
    try:
        with open(config_path, "r") as f:
            data = json.load(f)
        meta['model_size_mb'] = _safe_float(data.get("model_size_MB"))
        meta['param_count'] = _safe_int(data.get("param_count"))
    except Exception as e:
        print(f"⚠️ Could not read training metadata from {config_path}: {e}")
    return meta


def append_summary_row(summary_csv: Path, row: InferenceResult) -> None:
    """Appends a result row to the main summary CSV."""
    header = [f.name for f in fields(InferenceResult)]
    exists = summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(header)
        writer.writerow([getattr(row, h) for h in header])


def cleanup_model_binaries(models_dir: Path) -> None:
    """Deletes large model binaries to save space, keeping metadata."""
    for blob_type, filenames in MODEL_BLOBS_WHITELIST.items():
        for filename in filenames:
            fp = models_dir / filename
            if fp.exists():
                try:
                    fp.unlink()
                    print(f"🧹 Deleted model binary: {fp}")
                except Exception as e:
                    print(f"⚠️ Could not delete {fp}: {e}")

# ---- Main Experiment Runner ----
def run_experiments(
    algorithms: List[str], profiles: List[str],
    model_complexities: List[str], feature_complexities: List[str],
    horizon_values: List[int], quantize: bool,
    inference_steps: int, loading_strategy: str, interval_sec: float,
    delete_models: bool, limit_runs: Optional[int]
) -> None:

    out_dir = Path(CONFIG_PATH["paths"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / SUMMARY_CSV_NAME
    print(f"📈 Saving summary to: {summary_csv}")

    runs_done = 0
    total_combinations = len(algorithms) * len(profiles) * len(model_complexities) * len(feature_complexities) * len(horizon_values)
    print(f"🔬 Starting experiment grid with {total_combinations} total combinations.")


    for algorithm in algorithms:
        algo_key = "xgboost" if algorithm == "light_xgboost" else algorithm
        model_levels = MODEL_COMPLEXITY_LEVELS.get(algo_key, {})

        for profile in profiles:
            for model_level in model_complexities:
                if model_level not in model_levels: continue
                model_params = model_levels[model_level]

                for feature_level in feature_complexities:
                    feature_params = FEATURE_COMPLEXITY_LEVELS.get(feature_level)
                    if not feature_params: continue

                    for horizon in horizon_values:
                        if limit_runs is not None and runs_done >= limit_runs:
                            print(f"🏁 Reached run limit of {limit_runs}. Stopping.")
                            return

                        run_description = (
                            f"{algorithm} | {profile} | {model_level} | "
                            f"{feature_level} | H={horizon}"
                        )
                        print(f"\n[{runs_done + 1}/{total_combinations}] 🚀 RUNNING: {run_description}")

                        # --- ROBUSTNESS: Wrap each run in a try-except block ---
                        try:
                            # 1. Build Config
                            base_cfg, used_var = load_profile_config(algorithm, profile)
                            folder_flag = TRAINER_MAP[algorithm][2]
                            config = build_training_config(
                                base_cfg, profile, horizon, folder_flag,
                                feature_params, model_params, quantize
                            )

                            # 2. Training
                            print(f"  - Training with config: {used_var}...")
                            run_id, models_dir, train_time = run_training(algorithm, config, folder_flag)
                            print(f"  ✅ Training complete. run_id={run_id}, time={train_time:.2f}s")
                            
                            # 3. Inference for each model variant
                            variants = list_model_variants(models_dir)
                            if not variants:
                                print("  ⚠️ No model variants found, skipping inference.")
                                continue

                            config_name = DEFAULT_PROFILE_VARS.get(algorithm, {}).get(profile, algorithm)
                            for variant in variants:
                                print(f"  - Inferencing with variant: {variant}...")
                                rc = run_inference_via_subprocess(
                                    algorithm, run_id, variant, inference_steps,
                                    loading_strategy, interval_sec, config_name
                                )
                                if rc != 0:
                                    print(f"  ⚠️ Inference subprocess failed for {variant} (exit code {rc}).")
                                    # Still try to summarize what we have
                                
                                # 4. Summarize & Save Results
                                perf_metrics = summarize_inference_run(run_id, config)
                                meta_metrics = read_training_metadata(models_dir)

                                res = InferenceResult(
                                    algorithm=algorithm, profile=profile,
                                    model_complexity=model_level, feature_complexity=feature_level,
                                    horizon=horizon, model_variant=variant, run_id=run_id,
                                    avg_inference_time_ms=perf_metrics.get('avg_inference_time_ms'),
                                    p95_inference_time_ms=perf_metrics.get('p95_inference_time_ms'),
                                    avg_total_time_ms=perf_metrics.get('avg_total_time_ms'),
                                    avg_preprocess_time_ms=perf_metrics.get('avg_preprocess_time_ms'),
                                    avg_postprocess_time_ms=perf_metrics.get('avg_postprocess_time_ms'),
                                    model_load_time_s=None, # Placeholder, needs to be measured in subprocess
                                    avg_cpu_percent=perf_metrics.get('avg_cpu_percent'),
                                    avg_ram_percent=perf_metrics.get('avg_ram_percent'),
                                    model_size_mb=meta_metrics.get('model_size_mb'),
                                    param_count=meta_metrics.get('param_count'),
                                    mae=perf_metrics.get('mae'),
                                    rmse=perf_metrics.get('rmse')
                                )
                                append_summary_row(summary_csv, res)
                                print(f"  ✅ Summary row appended for {variant}.")

                            # 5. Cleanup
                            if delete_models:
                                cleanup_model_binaries(models_dir)

                        except Exception as e:
                            print(f"❌❌❌ CRITICAL ERROR during run: {run_description} ❌❌❌")
                            print(f"      Error Type: {type(e).__name__}")
                            print(f"      Error Details: {e}")
                            print("      Skipping to the next combination...")
                            # Optionally log this to a separate error file
                            with open(out_dir / "error_log.txt", "a") as f:
                                f.write(f"[{time.ctime()}] FAILED: {run_description}\n")
                                f.write(f"  ERROR: {e}\n\n")

                        runs_done += 1



def main():
    p = argparse.ArgumentParser(description="Experiment Grid Pipeline for Edge AI Benchmarking")

    # --- GEÄNDERT ---
    # Statt einer Liste von Algorithmen wird jetzt ein einzelner, erforderlicher Algorithmus übergeben.
    p.add_argument(
        "--algorithm",
        required=True,
        choices=list(TRAINER_MAP.keys()),
        help="The specific algorithm to run the experiment for (e.g., 'lstm', 'cnn1d')."
    )
    # --- ENDE ÄNDERUNG ---

    p.add_argument("--profiles", default="edge,server", help="Comma-separated list of profiles: server,edge")
    p.add_argument("--model-complexity", default="C1,C2,C3,C4,C5,C6", help="Comma-separated list of model complexity levels (e.g., C1,C2,C3)")
    p.add_argument("--feature-complexity", default="F1,F2,F3", help="Comma-separated list of feature complexity levels (e.g., F1,F2,F3)")
    p.add_argument("--horizon", default="1:19:2", help="Range 'start:stop:step' or comma-separated list for prediction horizon.")
    p.add_argument("--quantize", action="store_true", help="Enable post-training quantization (default: disabled).")
    p.add_argument("--inference-steps", type=int, default=100, help="Number of inference steps to run for metrics gathering.")
    p.add_argument("--loading-strategy", default="split", choices=["live_mqtt", "split"], help="Data source for inference.")
    p.add_argument("--interval-sec", type=float, default=0.0, help="Target inference interval. Set to 0 for max speed.")
    p.add_argument("--keep-models", action="store_true", help="Do NOT delete model binaries after inference.")
    p.add_argument("--limit-runs", type=int, help="Stop after this many successful experiment combinations (for debugging).")
    args = p.parse_args()

    # --- GEÄNDERT ---
    # Die Liste der Algorithmen enthält jetzt nur noch das eine übergebene Element.
    algorithms = [args.algorithm]
    # --- ENDE ÄNDERUNG ---

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    model_levels = [mc.strip().upper() for mc in args.model_complexity.split(",") if mc.strip()]
    feature_levels = [fc.strip().upper() for fc in args.feature_complexity.split(",") if fc.strip()]
    horizon_vals = _range_from_str(args.horizon)

    run_experiments(
        algorithms=algorithms,
        profiles=profiles,
        model_complexities=model_levels,
        feature_complexities=feature_levels,
        horizon_values=horizon_vals,
        quantize=args.quantize,
        inference_steps=args.inference_steps,
        loading_strategy=args.loading_strategy,
        interval_sec=args.interval_sec,
        delete_models=(not args.keep_models),
        limit_runs=args.limit_runs,
    )

if __name__ == "__main__":
    main()