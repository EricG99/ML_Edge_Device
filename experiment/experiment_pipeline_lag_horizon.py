#!/usr/bin/env python3
from __future__ import annotations
"""
Experiment-Pipeline (Lag/Horizon-Grid, Level=fixed "medium")
------------------------------------------------------------

Was macht dieses Skript?
  * Führt automatisiert Trainings- und Inferenzläufe über eine Liste von Lags **und** eine Liste von Horizons aus.
  * Die Modell-Komplexität ist für alle Modelle fest auf **"medium"** gesetzt (basierend auf den Presets aus multiconfig).
  * Für LSTM/CNN1D werden – falls vorhanden – mehrere Modellvarianten (no-quant/quant-16/quant-8) inferiert, bei den
    klassischen ML-Modellen nur no-quant.
  * Aggregiert Kennzahlen pro Kombination (Ø Inferenzzeit, Ø Total Time, Ø CPU %, Ø RAM %, Modellgröße) in einer CSV.
  * Optional: entfernt nach der Inferenz große Modell-/Scaler-Dateien (Platz sparen).

Baseline: Dieses Skript ist an `experiment_pipeline_multiconfig.py` angelehnt und fokussiert sich ausschließlich
auf die Grid-Suche über (lags × horizon) bei fixem Level=medium.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --- Projektpfad sicherstellen ---
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

# === Trainer-Klassen je Algorithmus ===
TRAINER_MAP = {
    "lstm": ("ML_Algorithms.LSTM.lstm_train", "LSTMTrainer", "LSTM"),
    "cnn1d": ("ML_Algorithms.CNN1D.cnn1d_train", "CNN1DTrainer", "CNN1D"),
    "random_forest": ("ML_Algorithms.Random_Forest.rf_train", "RandomForestTrainer", "Random_Forest"),
    "xgboost": ("ML_Algorithms.XGBOOST.xgboost_train", "XGBoostTrainer", "XGBOOST"),
    "light_xgboost": ("ML_Algorithms.Light_XGBOOST.light_xgboost_train", "LightXGBoostTrainer", "Light_XGBOOST"),
    "ridge": ("ML_Algorithms.RIDGE.ridge_lasso_train", "RidgeLassoTrainer", "RIDGE_LASSO"),
    "lasso": ("ML_Algorithms.RIDGE.ridge_lasso_train", "RidgeLassoTrainer", "RIDGE_LASSO"),
    "svm": ("ML_Algorithms.SVM.svm_train", "SVMTrainer", "SVM"),
}

# === Presets (werden aus multiconfig übernommen) ===
# Wir nutzen die dortigen Presets via build_training_config und überschreiben nur lags/horizon.
from experiment_pipeline_multiconfig import (
    build_training_config as _build_cfg_base,
)

# === Summary & Dateinamen ===
SUMMARY_CSV_NAME = "Experiment_Summary_LagHorizonGrid_revpi.csv"
SCALER_FILE_NAMES = ["scaler.joblib", "y_scaler.joblib"]
MODEL_BLOBS_WHITELIST = {
    "keras": ["model.keras"],
    "tflite": ["model_quant_float16.tflite", "model_quant_int8.tflite", "model_quant_int8_full.tflite"],
    "sklearn": ["model.joblib"],
    "xgb": ["model.json"],
}

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
# Utility-Funktionen
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
    """"start:stop:step" oder kommagetrennt (z. B. "5,10,20")."""
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
    if "ridge" in n or "lasso" in n: return "RIDGE_LASSO"
    if "svm" in n: return "SVM"
    return name_or_flag.upper() or "MODEL"


def list_model_variants(models_dir: Path) -> List[str]:
    found = set()
    for category in MODEL_BLOBS_WHITELIST.values():
        for filename in category:
            if (models_dir / filename).exists():
                found.add(filename)
    return sorted(list(found))


def _discover_predictions_file_from_json(run_id: str, error_metrics_dir: Path) -> Optional[Path]:
    agg_csv = error_metrics_dir / "ErrorMetrics_all_runs.csv"
    if not agg_csv.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(agg_csv)
        row = df[df["run_id"] == run_id].tail(1)
        if row.empty:
            return None
        json_path = Path(row.iloc[0]["json_path"])
        if not json_path.exists():
            return None
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
    if not training_config_json.exists():
        return None
    try:
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
        if not exists:
            writer.writerow(header)
        writer.writerow([
            row.algorithm, row.level, row.lags, row.horizon, row.model_variant, row.quant_mode,
            row.avg_inference_time_ms, row.avg_total_time_ms, row.avg_cpu_percent, row.avg_ram_percent,
            row.model_size_mb, row.run_id,
        ])


def cleanup_model_binaries_and_scalers(models_dir: Path, scalers_dir: Path) -> None:
    to_delete = []
    for category in MODEL_BLOBS_WHITELIST.values():
        for filename in category:
            fp = models_dir / filename
            if fp.exists():
                to_delete.append(fp)
    for fname in SCALER_FILE_NAMES:
        fp = scalers_dir / fname
        if fp.exists():
            to_delete.append(fp)
    for fp in sorted(to_delete):
        try:
            fp.unlink()
            print(f"🗑️  Gelöscht: {fp}")
        except Exception as e:
            print(f"⚠️  Fehler beim Löschen von {fp}: {e}")


# ---------------------------
# Training & Inferenz
# ---------------------------
def build_training_config_medium_with_lags(
    algorithm: str, horizon: int, lags: int, *, folder_flag: str, quant_modes: List[str]
) -> dict:
    """Baut die Konfiguration via multiconfig (Level=fixed "medium") und überschreibt lags/horizon/quant_modes."""
    cfg = _build_cfg_base(algorithm, level="medium", horizon=int(horizon), folder_flag=folder_flag, quant_modes=quant_modes)
    # Sicherstellen, dass lags gesetzt/überschrieben wird
    cfg["lags"] = int(lags)
    # (Optional) weitere Overrides hier möglich
    return cfg


def run_training(algorithm: str, config: dict, folder_flag: str) -> Tuple[str, Path]:
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
        print(f"(ignoriere) Konnte training_time_s nicht schreiben: {e}")

    run_id = str(config.get("run_id"))
    models_dir = Path(config["paths"]["Models"])
    return run_id, models_dir


def _file_candidates_for_mode(algo: str, mode: str) -> List[str]:
    mode = (mode or "no-quant").lower()
    if mode == "quant-16":
        return ["model_quant_float16.tflite"]
    if mode == "quant-8":
        return ["model_quant_int8_full.tflite", "model_quant_int8.tflite"]
    if algo in ("lstm", "cnn1d"):
        return ["model.keras"]
    if algo == "random_forest":
        return ["model.joblib"]
    if algo in ("xgboost", "light_xgboost"):
        return ["model.json", "model.joblib"]
    if algo == "random_forest": 
        return ["model.joblib"]
    if algo in ("random_forest", "ridge", "lasso", "svm"):
        return ["model.joblib"]

    return []


# --- Inferenz über Subprozess (an pipeline_web_app.py) ---
def run_inference_via_subprocess(
    algorithm: str, run_id: str, model_filename: str, inference_steps: int,
    loading_strategy: str, interval_sec: float
) -> int:
    """Startet pipeline_web_app.py im Headless-Modus für die Inferenz."""
    # robust Pfadfindung
    candidates = [
        PROJECT_ROOT / 'pipeline_web_app.py',
        PROJECT_ROOT / 'ML_Algorithms' / 'pipeline_web_app.py',
    ]
    script_path = None
    for c in candidates:
        if c.exists():
            script_path = c
            break
    if script_path is None:
        print("⚠️  pipeline_web_app.py nicht gefunden.")
        return 2

    cmd = [
        sys.executable, str(script_path),
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


def run_single_inference_and_append_summary(
    algo: str, cfg: dict, run_id: str, models_dir: Path,
    inference_steps: int, loading_strategy: str, interval_sec: float, summary_csv: Path,
    delete_models: bool
) -> None:
    variants_present = set(list_model_variants(models_dir))
    if not variants_present:
        print("⚠️  Kein Modell gefunden – Inferenz übersprungen.")
        return

    modes_to_run = cfg.get("quant_modes", ["no-quant"])
    # Für LSTM/CNN1D alle vorhandenen Modi, sonst nur no-quant
    queue: List[Tuple[str, str]] = []
    for qmode in modes_to_run:
        candidates = _file_candidates_for_mode(algo, qmode)
        chosen = next((f for f in candidates if f in variants_present), None)
        if chosen and (qmode, chosen) not in queue:
            queue.append((qmode, chosen))
        else:
            print(f"(skip) Modus '{qmode}': keine passende Datei gefunden (gesucht: {candidates})")

    if not queue:
        best_available = next((f for f in ["model.keras", "model.joblib", "model.json"] if f in variants_present), None)
        if best_available:
            queue.append(("no-quant", best_available))

    for qmode, model_file in queue:
        print(f"\n▶️  Inferenz: Modus '{qmode}' mit Datei '{model_file}'")
        rc = run_inference_via_subprocess(
            algorithm=algo, run_id=run_id, model_filename=model_file,
            inference_steps=inference_steps, loading_strategy=loading_strategy, interval_sec=interval_sec
        )
        if rc != 0:
            print(f"⚠️  Inferenz-Subprozess Fehlercode {rc} für {model_file}")
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
            print("⚠️  StepPredictions CSV nicht gefunden – Kennzahlen unvollständig.")

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
        print(f"✅ Summary-Zeile für {res.quant_mode} ({model_file}) geschrieben.")

    if delete_models:
        cleanup_model_binaries_and_scalers(models_dir, Path(cfg["paths"]["Scalers"]))


# ---------------------------
# Orchestrierung
# ---------------------------

def run_experiments(
    algorithms: List[str], lag_values: List[int], horizon_values: List[int],
    inference_steps: int, loading_strategy: str, interval_sec: float,
    delete_models: bool, limit_runs: Optional[int]
) -> None:
    out_dir = Path(CONFIG_PATH["paths"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / SUMMARY_CSV_NAME

    runs_done = 0
    FIXED_LEVEL = "medium"

    for algorithm in algorithms:
        algo = algorithm.lower()
        if algo not in TRAINER_MAP:
            print(f"⚠️  Unbekannter Algorithmus '{algorithm}' – überspringe.")
            continue

        quant_modes_for_algo = ["no-quant", "quant-16", "quant-8"] if algo in ["cnn1d", "lstm"] else ["no-quant"]
        print(f"\n=== Algorithmus: {algo} (Quantisierungsmodi: {quant_modes_for_algo}) ===")

        for lags in lag_values:
            for horizon in horizon_values:
                if limit_runs is not None and runs_done >= limit_runs:
                    print("⏹️  Maximalzahl an Läufen erreicht – stop.")
                    return

                folder_flag = algorithm_to_folder(algo)
                cfg = build_training_config_medium_with_lags(
                    algorithm=algo, horizon=horizon, lags=lags,
                    folder_flag=folder_flag, quant_modes=quant_modes_for_algo,
                )
                cfg["level_used"] = FIXED_LEVEL

                print(f"\n📦 LAUF {runs_done + 1} | Algo: {algo} | Level: {FIXED_LEVEL} | Lags: {lags} | Horizon: {horizon}")
                run_id, models_dir = run_training(algo, cfg, folder_flag)
                print(f"🏁 Training ok. run_id={run_id}")

                run_single_inference_and_append_summary(
                    algo, cfg, run_id, models_dir,
                    inference_steps, loading_strategy, interval_sec,
                    summary_csv, delete_models
                )
                runs_done += 1


# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Experiment-Pipeline (Lag/Horizon-Grid, Level=medium)")
    p.add_argument("--algorithms", default="cnn1d,lstm,random_forest,xgboost,light_xgboost,ridge,lasso,svm",
               help="Kommagetrennte Liste der Algorithmen")
    p.add_argument("--lags", default="5,10,20",
                   help="Range 'start:stop:step' oder kommagetrennt für Lags (z. B. '4:40:4' oder '5,10,20')")
    p.add_argument("--horizons", default="1:16:2",
                   help="Range 'start:stop:step' oder kommagetrennt für den Horizont")
    p.add_argument("--inference-steps", type=int, default=60, help="Anzahl Inferenzschritte pro Lauf")
    p.add_argument("--loading-strategy", default="split", choices=["split", "live_mqtt"],
                   help="Datenquelle für Inferenz")
    p.add_argument("--interval-sec", type=float, default=0.0, help="Ziel-Inferenzintervall in Sekunden")
    p.add_argument("--keep-models", action="store_true", help="Modellbinaries nach Inferenz NICHT löschen")
    p.add_argument("--limit-runs", type=int, help="Max. Anzahl an Experimenten (Kombinationen)")

    args = p.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    lag_vals = _range_from_str(args.lags)
    horizon_vals = _range_from_str(args.horizons)

    run_experiments(
        algorithms=algorithms,
        lag_values=lag_vals,
        horizon_values=horizon_vals,
        inference_steps=args.inference_steps,
        loading_strategy=args.loading_strategy,
        interval_sec=args.interval_sec,
        delete_models=(not args.keep_models),
        limit_runs=args.limit_runs,
    )


if __name__ == "__main__":
    main()
