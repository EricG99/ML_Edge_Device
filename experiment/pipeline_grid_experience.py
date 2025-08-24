#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment Pipeline - Grid Experience (v2.1)
--------------------------------------------
• Grid über (Algorithmen × Feature-Profile × Modell-Komplexität × Horizont)
• Train → Inferenz (headless über pipeline_web_app.py) → Auswertung
• Robust gegen fehlende Felder (z. B. kein paths['run_dir']) und flexible Dateinamensschemata
• Erkennt StepPredictions zuerst über ErrorMetrics_all_runs.csv (extra_info.predictions_file_path)
  und fällt erst dann auf Dateimuster in Prediction_Data zurück

Aufruf (Beispiel):
  python pipeline_grid_experience.py \
    --algorithms "lstm" \
    --features "F1,F3" \
    --complexity "C1,C6" \
    --horizon "1,19" \
    --inference-steps 50 \
    --loading-strategy split \
    --keep-artifacts \
    --limit-runs 4 \
    --no-quantization
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

# ---- Projektpfad sicherstellen ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---- Basis-Imports aus dem Projekt (robust) ----
try:
    from config.config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG
except ModuleNotFoundError:
    from config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG  # type: ignore

try:
    from ML_Helpfunctions import pipeline_utils as PU  # type: ignore
except ModuleNotFoundError:
    import importlib
    PU = importlib.import_module('pipeline_utils')  # letzter Fallback

# pipeline_web_app dynamisch importieren (damit --config-name funktioniert)
import importlib.util as _ilu

def _import_pipeline_web_app():
    candidates = [
        ('pipeline_web_app', PROJECT_ROOT / 'pipeline_web_app.py'),
        ('ML_Algorithms.pipeline_web_app', PROJECT_ROOT / 'ML_Algorithms' / 'pipeline_web_app.py'),
    ]
    # Sicherstellen, dass ML_Algorithms importierbar ist
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

_pwa = _import_pipeline_web_app()
load_config_dynamically = getattr(_pwa, 'load_config_dynamically')

# ---- Trainer-Klassen je Algorithmus ----
TRAINER_MAP = {
    "lstm": ("ML_Algorithms.LSTM.LSTM_train", "LSTMTrainer", "LSTM"),
    "cnn1d": ("ML_Algorithms.CNN1D.cnn1d_train", "CNN1DTrainer", "CNN1D"),
    "random_forest": ("ML_Algorithms.Random_Forest.rf_train", "RandomForestTrainer", "Random_Forest"),
    "xgboost": ("ML_Algorithms.XGBOOST.XGBOOST_train", "XGBoostTrainer", "XGBOOST"),
}

# Welche Dateien dürfen optional gelöscht werden (wenn --keep-artifacts nicht gesetzt ist)
MODEL_BLOBS_WHITELIST = {
    "keras": ["model.keras"],
    "tflite": ["*.tflite"],
    "sklearn": ["model.joblib"],
    "xgb": ["model.json"],
}
SUMMARY_CSV_NAME = "Grid_Experiment_Summary.csv"

# ======================================================================================
# Definition der Grid-Stufen
# ======================================================================================
FEATURE_COMPLEXITY_LEVELS: Dict[str, Dict] = {
    # Bx: Anzahl Basis-Features, Lx: lags, Rx: Rolling-Features
    "F1": {"lags": 4,  "base_features_n": 1, "rolling_features": "none"},
    "F2": {"lags": 12, "base_features_n": 2, "rolling_features": "mean"},
    "F3": {"lags": 20, "base_features_n": 3, "rolling_features": "all"},
}

MODEL_COMPLEXITY_LEVELS: Dict[str, Dict[str, Dict]] = {
    "lstm": {
        "C1": {"num_layers": 1, "initial_units": 16,  "dropout": 0.0, "epochs": 10},
        "C2": {"num_layers": 1, "initial_units": 32,  "dropout": 0.1, "epochs": 20},
        "C3": {"num_layers": 2, "initial_units": 64,  "dropout": 0.2, "epochs": 30},
        "C4": {"num_layers": 2, "initial_units": 128, "dropout": 0.2, "epochs": 50},
        "C5": {"num_layers": 3, "initial_units": 192, "dropout": 0.3, "epochs": 70},
        "C6": {"num_layers": 3, "initial_units": 256, "dropout": 0.3, "epochs": 90},
    },
    "cnn1d": {
        "C1": {"num_blocks": 1, "initial_filters": 16,  "kernel_size": 3, "epochs": 10},
        "C2": {"num_blocks": 2, "initial_filters": 32,  "kernel_size": 3, "epochs": 20},
        "C3": {"num_blocks": 3, "initial_filters": 64,  "kernel_size": 3, "epochs": 30},
        "C4": {"num_blocks": 4, "initial_filters": 96,  "kernel_size": 5, "epochs": 50},
        "C5": {"num_blocks": 5, "initial_filters": 128, "kernel_size": 5, "epochs": 70},
        "C6": {"num_blocks": 6, "initial_filters": 192, "kernel_size": 5, "epochs": 90},
    },
}

# ======================================================================================
# Utilities
# ======================================================================================

def _try_import(module_path: str, class_name: str):
    import importlib
    m = importlib.import_module(module_path)
    return getattr(m, class_name)


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _deep_merge(a: Dict, b: Dict) -> Dict:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _parse_range_or_list(s: str, prefix_filter: Optional[str] = None) -> List:
    s = s.strip()
    if ":" in s:
        a, b, step = [int(x) for x in s.split(":", 2)]
        return list(range(a, b + 1, step))
    vals = [x.strip() for x in s.split(",") if x.strip()]
    if prefix_filter:
        vals = [x for x in vals if x.startswith(prefix_filter)]
    return vals


def algorithm_to_folder(algorithm: str) -> str:
    return TRAINER_MAP[algorithm][2]


def build_run_config(algorithm: str, feat_level: str, comp_level: str, horizon: int, enable_quantization: bool) -> Dict:
    """Baut die finale Config für *einen* Grid-Punkt (Training)."""
    # 1) Basiskonfig laden (z. B. param_lstm_edge oder als Fallback "lstm")
    try:
        base_cfg = load_config_dynamically(algorithm, algorithm)
    except Exception:
        # Minimaler Fallback
        base_cfg = {"model_name": algorithm, "paths": CONFIG_PATH.get("paths", {})}

    # 2) Feature-Komplexität anwenden
    feat_params = FEATURE_COMPLEXITY_LEVELS[feat_level]
    feat_overrides: Dict = {
        "lags": feat_params["lags"],
    }
    # Basisfeature-Liste (falls vorhanden) beschneiden
    base_features_all = base_cfg.get("base_features", [])
    if base_features_all:
        n_feats = min(feat_params["base_features_n"], len(base_features_all))
        feat_overrides["base_features"] = base_features_all[:n_feats]
    # Rolling-Features setzen
    if feat_params["rolling_features"] == "none":
        feat_overrides.update({"include_roll_mean": False, "include_roll_std": False})
    elif feat_params["rolling_features"] == "mean":
        feat_overrides.update({"include_roll_mean": True, "include_roll_std": False})
    else:  # "all"
        feat_overrides.update({"include_roll_mean": True, "include_roll_std": True})

    # 3) Modell-Komplexität anwenden
    model_params = MODEL_COMPLEXITY_LEVELS.get(algorithm, {}).get(comp_level, {})
    model_overrides = {"model_params": model_params}
    model_overrides.update(model_params)  # neuere Trainer lesen top-level Werte

    # 4) Zusammenführen + generische Pfade/Flags
    merged = _deep_merge(base_cfg, feat_overrides)
    merged = _deep_merge(merged, model_overrides)
    merged["horizon"] = horizon

    merged = _deep_merge(merged, {"paths": CONFIG_PATH["paths"]})
    merged = _deep_merge(merged, CONFIG_LOAD_ARTIFACTS)
    merged = _deep_merge(merged, MQTT_CONFIG)
    merged["edge_device"] = True
    merged["enable_edge"] = True

    # 5) Experiment-Ordner anlegen (setzt paths.Models etc.)
    folder_flag = algorithm_to_folder(algorithm)
    merged, _ = PU.setup_experiment(merged, folder_flag, run_type="train")

    # 6) Quantisierungs-Flag merken (für spätere Lösch-Logik)
    merged["enable_quantization"] = enable_quantization
    return merged


def _error_metrics_json_for_run(run_id: str, err_dir: Path) -> Optional[Path]:
    """Liest ErrorMetrics_all_runs.csv und liefert den JSON-Pfad zur run_id (falls vorhanden)."""
    agg_csv = err_dir / "ErrorMetrics_all_runs.csv"
    if not agg_csv.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(agg_csv)
        row = df[df["run_id"] == run_id].tail(1)
        if row.empty:
            return None
        json_path = Path(row.iloc[0]["json_path"]).resolve()
        return json_path if json_path.exists() else None
    except Exception:
        return None


def _discover_predictions_file_from_json(run_id: str, paths: Dict) -> Optional[Path]:
    err_dir = Path(paths.get("Error_Metrics", Path(CONFIG_PATH["paths"]["output"]) / "Error_Metrics"))
    js = _error_metrics_json_for_run(run_id, err_dir)
    if not js:
        return None
    try:
        with open(js, "r", encoding="utf-8") as f:
            data = json.load(f)
        p = data.get("extra_info", {}).get("predictions_file_path")
        if p:
            pth = Path(p)
            if pth.exists():
                return pth
    except Exception:
        pass
    return None


def _fallback_find_step_csv(run_id: str, prediction_data_dir: Path, base_model_stem: Optional[str] = None) -> Optional[Path]:
    """Suche StepPredictions über Dateimuster in Prediction_Data (fallback)."""
    patterns = [f"StepPredictions_{run_id}_*.csv"]
    if base_model_stem:
        patterns.insert(0, f"StepPredictions_{run_id}_*_{base_model_stem}.csv")
    for pat in patterns:
        hits = sorted(prediction_data_dir.glob(pat))
        if hits:
            return hits[-1]
    return None


def summarize_step_csv(step_csv: Path) -> Dict[str, Optional[float]]:
    """Berechnet Ø-Metriken + p95/Jitter (ms) aus StepPredictions."""
    metrics = {
        "avg_total_time_ms": None,
        "avg_inference_time_ms": None,
        "avg_preprocess_time_ms": None,
        "avg_postprocess_time_ms": None,
        "latency_p95_ms": None,
        "latency_jitter_ms": None,
        "avg_cpu_percent": None,
        "avg_ram_percent": None,
    }
    try:
        import pandas as pd
        df = pd.read_csv(step_csv)
        if df.empty:
            return metrics
        # Sekunden → Millisekunden
        for col in ["total_time_s", "inference_time_s", "preprocess_time_s", "postprocess_time_s"]:
            if col in df.columns:
                metrics[f"avg_{col.replace('_s', '_ms')}"] = _safe_float(df[col].mean() * 1000.0)
        if "total_time_s" in df.columns:
            metrics["latency_p95_ms"] = _safe_float(df["total_time_s"].quantile(0.95) * 1000.0)
            metrics["latency_jitter_ms"] = _safe_float(df["total_time_s"].std() * 1000.0)
        if "cpu_percent" in df.columns:
            metrics["avg_cpu_percent"] = _safe_float(df["cpu_percent"].mean())
        if "ram_percent" in df.columns:
            metrics["avg_ram_percent"] = _safe_float(df["ram_percent"].mean())
    except Exception:
        pass
    return metrics


@dataclass
class ExperimentResult:
    algorithm: str
    feature_level: str
    complexity_level: str
    horizon: int
    run_id: Optional[str] = None
    model_variant: Optional[str] = None
    status: str = "pending"
    error_message: Optional[str] = None
    # Trainings-/Modellmetriken
    training_time_s: Optional[float] = None
    model_size_mb: Optional[float] = None
    param_count: Optional[int] = None
    # Inferenzmetriken
    avg_total_time_ms: Optional[float] = None
    avg_inference_time_ms: Optional[float] = None
    avg_preprocess_time_ms: Optional[float] = None
    avg_postprocess_time_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_jitter_ms: Optional[float] = None
    avg_cpu_percent: Optional[float] = None
    avg_ram_percent: Optional[float] = None


# ======================================================================================
# Orchestrierung
# ======================================================================================

def read_metrics_from_training_config(training_config_json: Path) -> Dict[str, Optional[float]]:
    out = {"model_size_mb": None, "param_count": None, "training_time_s": None}
    if not training_config_json.exists():
        return out
    try:
        with open(training_config_json, "r", encoding="utf-8") as f:
            js = json.load(f)
        out["model_size_mb"] = _safe_float(js.get("model_size_MB"))
        out["param_count"] = int(js.get("param_count", 0)) if js.get("param_count") else None
        out["training_time_s"] = _safe_float(js.get("training_time_s"))
    except Exception:
        pass
    return out


def append_summary_row(summary_csv: Path, result: ExperimentResult) -> None:
    res = asdict(result)
    exists = summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(res.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(res)


def run_grid_experiment(
    algorithms: List[str],
    feature_levels: List[str],
    complexity_levels: List[str],
    horizon_values: List[int],
    inference_steps: int,
    loading_strategy: str,
    interval_sec: float,
    keep_artifacts: bool,
    limit_runs: Optional[int],
    enable_quantization: bool,
) -> None:
    out_dir = Path(CONFIG_PATH["paths"]["output"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / SUMMARY_CSV_NAME
    print(f"📈 Ergebnisse werden in {summary_csv} gespeichert.")

    total_runs = len(algorithms) * len(feature_levels) * len(complexity_levels) * len(horizon_values)
    if limit_runs is not None:
        total_runs = min(total_runs, limit_runs)
    print(f"🔬 Starte Grid-Experiment mit bis zu {total_runs} Durchläufen.")

    run_counter = 0
    for algo in algorithms:
        for feat_level in feature_levels:
            for comp_level in complexity_levels:
                for horizon in horizon_values:
                    if limit_runs is not None and run_counter >= limit_runs:
                        print("🏁 Run-Limit erreicht. Pipeline wird beendet.")
                        return
                    run_counter += 1

                    print("-" * 80)
                    log_prefix = f"[{run_counter}/{total_runs}]"
                    print(f"{log_prefix} LAUF: Algo={algo}, Features={feat_level}, Komplexität={comp_level}, Horizont={horizon}")

                    result = ExperimentResult(
                        algorithm=algo, feature_level=feat_level,
                        complexity_level=comp_level, horizon=horizon,
                    )

                    models_dir: Optional[Path] = None
                    try:
                        # 1) Config bauen + Run-Ordner anlegen
                        cfg = build_run_config(algo, feat_level, comp_level, horizon, enable_quantization)
                        run_id = str(cfg.get("run_id"))
                        models_dir = Path(cfg["paths"]["Models"])  # durch setup_experiment gesetzt
                        run_dir = models_dir.parent  # robust statt cfg['paths']['run_dir'] (kann fehlen)
                        print(f"{log_prefix} 📁 Ergebnisordner: {run_dir}")
                        result.run_id = run_id

                        # 2) Training
                        print(f"{log_prefix} Phase 1: Training starten…")
                        module, clsname, folder_flag = TRAINER_MAP[algo]
                        Trainer = _try_import(module, clsname)
                        trainer = Trainer(config=cfg, folder_flag=folder_flag)
                        trainer.run(save_artifacts=True)
                        print(f"{log_prefix} ✅ Training abgeschlossen. Run ID: {run_id}")

                        # 3) Modellvariante festlegen & ggf. TFLite entfernen
                        base_model_filename = (
                            "model.keras" if algo in ["lstm", "cnn1d"] else
                            "model.joblib" if algo == "random_forest" else
                            "model.json"
                        )
                        result.model_variant = base_model_filename

                        if not enable_quantization:
                            print(f"{log_prefix} Aufräumen: Quantisierung ist deaktiviert. Entferne TFLite-Dateien…")
                            for tfl in models_dir.glob("*.tflite"):
                                try:
                                    tfl.unlink()
                                    print(f"{log_prefix} 🧹 Unerwünschte Datei entfernt: {tfl.name}")
                                except Exception as de:
                                    print(f"{log_prefix} ⚠️ Konnte {tfl.name} nicht löschen: {de}")

                        if not (models_dir / base_model_filename).exists():
                            raise FileNotFoundError(
                                f"Das erwartete Basis-Modell '{base_model_filename}' wurde im Ordner nicht gefunden!")

                        # 4) Inferenz per Subprozess
                        print(f"{log_prefix} Phase 2: Inferenz starten für '{base_model_filename}'.")
                        app = Path(getattr(_pwa, "__file__", PROJECT_ROOT / 'pipeline_web_app.py'))
                        cmd = [
                            sys.executable, str(app),
                            "--algorithm", algo, "--load_id", run_id,
                            "--model_filename", base_model_filename, "--no-web",
                            "--inference-steps", str(inference_steps),
                            "--set", f"loading_strategy={loading_strategy}",
                            "--set", f"inference_interval_sec={interval_sec}",
                            "--config-name", cfg.get('model_name', algo)
                        ]
                        import subprocess
                        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
                        if proc.returncode != 0:
                            print(f"{log_prefix} ⚠️ Inferenz-Subprozess endete mit Code {proc.returncode}.")
                            print("--- STDOUT ---\n" + proc.stdout)
                            print("--- STDERR ---\n" + proc.stderr + "\n----------------")
                            raise RuntimeError(f"Inference subprocess failed with code {proc.returncode}")
                        print(f"{log_prefix} ✅ Inferenz abgeschlossen.")

                        # 5) Auswertung
                        print(f"{log_prefix} Phase 3: Ergebnisse auswerten…")
                        # 5a) Trainingsmetriken aus training_config.json
                        tcfg_json = models_dir / "training_config.json"
                        tmetrics = read_metrics_from_training_config(tcfg_json)
                        result.training_time_s = tmetrics["training_time_s"]
                        result.model_size_mb = tmetrics["model_size_mb"]
                        result.param_count = tmetrics["param_count"]

                        # 5b) StepPredictions finden (zuerst über ErrorMetrics JSON, dann Fallback)
                        step_csv = (_discover_predictions_file_from_json(run_id, cfg.get("paths", {}))
                                    or _fallback_find_step_csv(run_id, Path(cfg["paths"]["Prediction_Data"]),
                                                               base_model_stem=Path(base_model_filename).stem))
                        if not step_csv or not Path(step_csv).exists():
                            raise FileNotFoundError(f"Keine StepPredictions CSV gefunden für Run {run_id}")

                        # 5c) CSV zusammenfassen
                        infer_metrics = summarize_step_csv(Path(step_csv))
                        for k, v in infer_metrics.items():
                            setattr(result, k, v)

                        result.status = "success"
                        print(f"{log_prefix} ✅ Auswertung erfolgreich.")

                    except Exception as e:
                        print(f"{log_prefix} ❌ FEHLER im Durchlauf: {e}")
                        print(traceback.format_exc())
                        result.status = "failed"
                        result.error_message = f"{type(e).__name__}: {e}"

                    finally:
                        append_summary_row(summary_csv, result)

                        # 6) Aufräumen (nur wenn NICHT behalten werden soll)
                        if models_dir and models_dir.exists() and not keep_artifacts:
                            print(f"{log_prefix} Phase 4: Aufräumen.")
                            for pattern_list in MODEL_BLOBS_WHITELIST.values():
                                for pattern in pattern_list:
                                    for fp in models_dir.glob(pattern):
                                        try:
                                            fp.unlink()
                                            print(f"🧹 Gelöscht: {fp.name}")
                                        except Exception as del_e:
                                            print(f"⚠️ Fehler beim Löschen von {fp}: {del_e}")
                        print(f"{log_prefix} ✅ Lauf abgeschlossen.")


# ======================================================================================
# CLI
# ======================================================================================

def main():
    p = argparse.ArgumentParser(description="Grid-Experiment-Pipeline für Komplexitätsanalyse")
    p.add_argument("--algorithms", default="lstm,cnn1d", help="Kommagetrennte Liste von Algorithmen.")
    p.add_argument("--features", default="F1,F2,F3", help="Feature-Stufen: Range 'start:stop:step' oder kommagetrennt (F1,F3).")
    p.add_argument("--complexity", default="C1,C2,C3,C4,C5,C6", help="Modell-Komplexität: Range 'start:stop:step' oder kommagetrennt.")
    p.add_argument("--horizon", default="1:19:2", help="Prognosehorizont: Range 'start:stop:step' oder kommagetrennt.")
    p.add_argument("--inference-steps", type=int, default=200, help="Anzahl Inferenzschritte pro Lauf.")
    p.add_argument("--loading-strategy", default="split", choices=["split", "live_mqtt"], help="Datenquelle für die Inferenz.")
    p.add_argument("--interval-sec", type=float, default=0.0, help="Ziel-Inferenzintervall. 0.0 für max. Geschwindigkeit.")
    p.add_argument("--keep-artifacts", action="store_true", help="Artefakte nach dem Lauf NICHT löschen.")
    p.add_argument("--limit-runs", type=int, default=None, help="Maximale Anzahl an Grid-Kombinationen (für Debugging).")
    p.add_argument("--no-quantization", action="store_true", help="Quantisierung komplett deaktivieren (löscht ggf. .tflite nach dem Training).")
    args = p.parse_args()

    algorithms = [a.strip().lower() for a in _parse_range_or_list(args.algorithms)]
    feature_levels = [f.strip().upper() for f in _parse_range_or_list(args.features)]
    complexity_levels = [c.strip().upper() for c in _parse_range_or_list(args.complexity)]
    horizon_values = [int(x) for x in _parse_range_or_list(args.horizon)]

    run_grid_experiment(
        algorithms=algorithms,
        feature_levels=feature_levels,
        complexity_levels=complexity_levels,
        horizon_values=horizon_values,
        inference_steps=args.inference_steps,
        loading_strategy=args.loading_strategy,
        interval_sec=args.interval_sec,
        keep_artifacts=args.keep_artifacts,
        limit_runs=args.limit_runs,
        enable_quantization=(not args.no_quantization),
    )


if __name__ == "__main__":
    main()
