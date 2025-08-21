#!/usr/bin/env python3
from __future__ import annotations
"""
Experiment Pipeline
-------------------

Zweck
  * Führt automatisiert Trainings- und Inferenzläufe über Modelle, Profile (Server/Edge),
    Lags/Horizon-Grids und (falls vorhanden) quantisierte Modellvarianten aus.
  * Startet das initiale Training programmatisch (ohne Web-UI), sammelt die erzeugte run_id
    und ruft anschließend die bestehende Web/Headless-Pipeline (`pipeline_web_app.py`) für die
    Inferenz mit `--load_id` und spezifiziertem `--model_filename` auf.
  * Aggregiert Metriken (Ø Inferenzzeit, Ø Total Time, Ø CPU %, Ø RAM %) je Kombination und
    speichert eine kompakte Übersichtstabelle.
  * Löscht nach jeder Inferenz die verwendeten Modellbinaries (modell.keras, *.tflite, *.joblib),
    um Speicherplatz zu sparen – Artefakt-Metadaten (training_config.json, features.joblib,
    scaler.joblib, Step-Predictions & ErrorMetrics) bleiben erhalten.

Wichtige Hinweise
  * Die spezifischen Konfigurationsprofile (z. B. `param_cnn1d_server`/`param_cnn1d_edge`) sollen
    laut Anforderung im **nächsten Schritt** erstellt/erweitert werden. Diese Pipeline versucht
    deshalb zunächst, solche Profil-Variablen dynamisch zu laden. Falls ein Profilname fehlt,
    wird auf das Basis-Config-Objekt (z. B. `cnn1d`) zurückgefallen und Lags/Horizon/Flags per
    Override gesetzt.
  * Inferenzmodus standardmäßig `live_mqtt` (1 Hz, 60 Schritte), kann aber auch explizit im
    **split**-Modus laufen.
  * Keine Retrainings im Inferenzschritt.

Kompatibel mit:
  - `pipeline_web_app.py` (Training+Inferenz; hier für Inferenz via Subprozess genutzt)
  - Basistrainer in `ML_Algorithms` (Training programmatisch)

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

# ---- Projektpfad sicherstellen ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Annahme: Datei liegt im Ordner "experiment/"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# ---- Imports aus bestehendem Projekt ----
# -- Flexible Imports (Config) --
try:
    from config.config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG  # type: ignore
except ModuleNotFoundError:
    from config_general import CONFIG_PATH, CONFIG_LOAD_ARTIFACTS, MQTT_CONFIG  # type: ignore
# -- Flexible Imports (Pipeline Utils) --
try:
    from ML_Helpfunctions import Pipeline_Utils as PU  # type: ignore
except ModuleNotFoundError:
    try:
        from ML_Helpfunktions import Pipeline_Utils as PU  # type: ignore
    except ModuleNotFoundError:
        import importlib
        PU = importlib.import_module('Pipeline_Utils')  # fallback to root module name

# Für dynamisches Laden von Config-Profilen wiederverwenden
# -- Try to locate pipeline_web_app in multiple places --
import importlib.util as _ilu

def _import_pipeline_web_app():
    candidates = [
        ('pipeline_web_app', PROJECT_ROOT / 'pipeline_web_app.py'),
        ('ML_Algorithms.pipeline_web_app', PROJECT_ROOT / 'ML_Algorithms' / 'pipeline_web_app.py'),
    ]
    # Ensure candidate dirs are importable
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

try:
    _pwa = _import_pipeline_web_app()
    load_config_dynamically = getattr(_pwa, 'load_config_dynamically')
except Exception:
    # Fallback: implement a minimal loader that tries common config modules
    def load_config_dynamically(algo: str, varname: str):
        import importlib
        candidates = [
            f'config_ml_{algo}',
            f'config_{algo}',
            'config_ml_cnn1d', 'config_ml_lstm', 'config_ml_xgboost', 'config_ml_rf',
            'config_general',
        ]
        last_err = None
        for mod in candidates:
            try:
                m = importlib.import_module(mod)
                if hasattr(m, varname):
                    return getattr(m, varname)
                if hasattr(m, algo):  # fallback to base var equals algo
                    return getattr(m, algo)
            except Exception as e:
                last_err = e
                continue
        raise ModuleNotFoundError(f"Could not resolve config var '{varname}' (algo='{algo}'). Last error: {last_err}")

# ---- Trainer-Klassen je Algorithmus (programmatisches Training) ----
TRAINER_MAP = {
    "lstm": ("ML_Algorithms.LSTM.LSTM_train", "LSTMTrainer", "LSTM"),
    "cnn1d": ("ML_Algorithms.CNN1D.cnn1d_train", "CNN1DTrainer", "CNN1D"),
    "random_forest": ("ML_Algorithms.Random_Forest.rf_train", "RandomForestTrainer", "Random_Forest"),
    "xgboost": ("ML_Algorithms.XGBOOST.XGBOOST_train", "XGBoostTrainer", "XGBOOST"),
    # Light XGBoost -> nutzt den XGBoost-Trainer (leichtere Hyperparameter via Config), eigener Ordner-Flag
    "light_xgboost": ("ML_Algorithms.XGBOOST.XGBOOST_train", "XGBoostTrainer", "LIGHT_XGBOOST"),
}

# Default-Config-Variablen pro Profil (werden versucht; sonst Fallback -> Basisvariable == Algorithmus)
# Für light_xgboost wahlweise eigene Profile oder Fallback auf xgboost_*.
DEFAULT_PROFILE_VARS: Dict[str, Dict[str, str]] = {
    "lstm": {"server": "param_lstm_server", "edge": "param_lstm_edge"},
    "cnn1d": {"server": "param_cnn1d_server", "edge": "param_cnn1d_edge"},
    "random_forest": {"server": "random_forest_server", "edge": "random_forest_edge"},
    "xgboost": {"server": "xgboost_server", "edge": "xgboost_edge"},
    "light_xgboost": {"server": "light_xgboost_server", "edge": "light_xgboost_edge"},
}

# Modell-Dateien, die als "groß" gelten und nach der Inferenz entfernt werden dürfen
MODEL_BLOBS_WHITELIST = {
    "keras": ["model.keras"],
    "tflite": ["model_quant_float16.tflite", "model_quant_int8.tflite", "model_quant_int8_full.tflite"],
    "sklearn": ["model.joblib"],  # z. B. RF, evtl. Lasso/SVM später
    "xgb": ["model.json"],        # XGBoost & Light-XGBoost
}

# Datei- und Ordnernamen
SUMMARY_CSV_NAME = "Experiment_Summary.csv"


# ---------------------------
# Hilfs-Datenstrukturen
# ---------------------------
@dataclass
class RunSpec:
    algorithm: str
    profile: str  # "server" | "edge"
    lags: int
    horizon: int
    config_var_used: str  # tatsächlich verwendeter Config-Variablenname
    run_id: str
    model_dir: Path

@dataclass
class InferenceResult:
    algorithm: str
    profile: str
    lags: int
    horizon: int
    model_variant: str  # z. B. model.keras / model_quant_float16.tflite / model.joblib / model.json
    avg_inference_time_ms: Optional[float]
    avg_total_time_ms: Optional[float]
    avg_cpu_percent: Optional[float]
    avg_ram_percent: Optional[float]
    model_size_mb: Optional[float]
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
    """Wandelt 'start:stop:step' oder eine durch Komma getrennte Liste in int-Liste um."""
    spec = spec.strip()
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) == 2:
            start, stop = parts
            step = 1
        else:
            start, stop, step = parts
        return list(range(start, stop + (1 if step > 0 else -1), step))
    # Kommagetrennt
    return [int(x) for x in spec.split(",") if x]


def _safe_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


# ---------------------------
# Konfigurationsaufbau
# ---------------------------

def load_profile_config(algorithm: str, profile: str) -> Tuple[dict, str]:
    """Versucht, ein Profil-Config-Objekt dynamisch zu laden; fällt andernfalls auf Basisvariable zurück.
        Für light_xgboost wird bei Bedarf auf die xgboost-Configs zurückgefallen.
    """
    algo = algorithm.lower()
    varname = DEFAULT_PROFILE_VARS.get(algo, {}).get(profile)
    cfg = None
    used: Optional[str] = None

    # 1) Versuche Profil-Variable (z. B. param_cnn1d_server)
    if varname:
        try:
            cfg = load_config_dynamically(algo, varname)
            used = varname
        except SystemExit:
            cfg = None
        except Exception:
            cfg = None

    # 2) Fallback: Basisvariable == Algorithmus
    if cfg is None:
        try:
            cfg = load_config_dynamically(algo, algo)
            used = algo
        except Exception:
            cfg = None

    # 3) Spezieller Fallback: light_xgboost -> xgboost
    if cfg is None and algo == "light_xgboost":
        try:
            base_var = DEFAULT_PROFILE_VARS.get("xgboost", {}).get(profile, "xgboost")
            cfg = load_config_dynamically("xgboost", base_var)
            used = base_var + " (via light_xgboost)"
        except Exception:
            cfg = load_config_dynamically("xgboost", "xgboost")
            used = "xgboost (via light_xgboost)"

    return cfg, (used or algo)


def algorithm_to_folder(name_or_flag: str) -> str:
    n = (name_or_flag or "").lower()
    if "light_xgboost" in n or "light-xgboost" in n or "light xgboost" in n:
        return "Light_XGBOOST"
    if "lstm" in n:
        return "LSTM"
    if "cnn" in n:
        return "CNN1D"
    if "xgb" in n or "xgboost" in n:
        return "XGBOOST"
    if "rf" in n or "random_forest" in n or "random forest" in n:
        return "Random_Forest"
    return name_or_flag.upper() or "MODEL"


def build_training_config(base_cfg: dict, profile: str, lags: int, horizon: int, folder_flag: str) -> dict:
    """Mergt allgemeine Pfade/Flags, setzt Lags/Horizon, Profile-Flags etc.
        Wichtig: setup_experiment mit dem kanonischen Ordner-Flag ausführen,
        damit die Run-Ordner exakt zum Algorithmus-Ordner passen.
    """
    merged = _deep_merge(base_cfg, {
        "paths": CONFIG_PATH["paths"],
        # keine Web-UI, keine Retrainings in diesem Trainingsteil
        "inference_mode": "load_artifacts_path",
        "edge_device": (profile == "edge"),
        "enable_edge": (profile == "edge"),
        # Exakt vorgegebene Lags/Horizon übernehmen
        "lags": int(lags),
        "horizon": int(horizon),
    })

    # MQTT/Allg. Laufzeitwerte für Einheitlichkeit setzen
    merged = _deep_merge(merged, CONFIG_LOAD_ARTIFACTS)
    merged = _deep_merge(merged, MQTT_CONFIG)

    # Experimentordner erzeugen (train) – *mit* folder_flag
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

    # Trainingszeit in training_config.json mitschreiben (für spätere Auswertung)
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

def list_model_variants(models_dir: Path, algorithm: str) -> List[str]:
    """Findet vorhandene Modellvarianten in einem Models-Ordner und gibt Dateinamen zurück."""
    candidates: List[str] = []
    # Keras / TFLite Varianten
    for p in MODEL_BLOBS_WHITELIST["keras"] + MODEL_BLOBS_WHITELIST["tflite"]:
        if (models_dir / p).exists():
            candidates.append(p)
    # Scikit-Learn
    for p in MODEL_BLOBS_WHITELIST["sklearn"]:
        if (models_dir / p).exists():
            candidates.append(p)
    # XGBoost (inkl. Light)
    for p in MODEL_BLOBS_WHITELIST["xgb"]:
        if (models_dir / p).exists():
            candidates.append(p)

    # Falls keine der bekannten Dateien existiert, alles anzeigen (Robustheit)
    if not candidates:
        candidates = [x.name for x in models_dir.glob("*") if x.is_file()]
    return candidates


def run_inference_via_subprocess(
    algorithm: str,
    run_id: str,
    model_filename: str,
    inference_steps: int,
    loading_strategy: str = "live_mqtt",
    interval_sec: float = 1.0,
    config_name: str | None = None,
) -> int:
    """Ruft pipeline_web_app.py für eine reine Inferenz auf. Gibt den Exitcode zurück."""
    py = sys.executable
    # Resolve pipeline_web_app path robustly (root or ML_Algorithms)
    app_path_candidates: List[Path] = []
    try:
        from types import ModuleType
        if '_pwa' in globals() and isinstance(_pwa, ModuleType):
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
# Auswertung & Speicherung
# ---------------------------

def _discover_predictions_file_from_json(run_id: str, error_metrics_dir: Path) -> Optional[Path]:
    """Liest ErrorMetrics_all_runs.csv, findet JSON für run_id und extrahiert predictions_file_path."""
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
    if hits:
        return hits[-1]
    return None


def summarize_step_csv(step_csv: Path) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Gibt Ø inference_time_ms, Ø total_time_ms, Ø cpu_percent, Ø ram_percent zurück."""
    try:
        import pandas as pd
        df = pd.read_csv(step_csv)
        # Zeiten liegen in Sekunden in der CSV
        inf_ms = _safe_float(df["inference_time_s"].mean() * 1000.0) if "inference_time_s" in df else None
        tot_ms = _safe_float(df["total_time_s"].mean() * 1000.0) if "total_time_s" in df else None
        cpu = _safe_float(df["cpu_percent"].mean()) if "cpu_percent" in df else None
        ram = _safe_float(df["ram_percent"].mean()) if "ram_percent" in df else None
        return inf_ms, tot_ms, cpu, ram
    except Exception:
        return None, None, None, None


def read_model_size_mb(training_config_json: Path) -> Optional[float]:
    if not training_config_json.exists():
        return None
    try:
        with open(training_config_json, "r", encoding="utf-8") as f:
            js = json.load(f)
        return _safe_float(js.get("model_size_MB"))
    except Exception:
        return None


def append_summary_row(summary_csv: Path, row: InferenceResult) -> None:
    header = [
        "algorithm", "profile", "lags", "horizon", "model_variant",
        "avg_inference_time_ms", "avg_total_time_ms", "avg_cpu_percent", "avg_ram_percent",
        "model_size_mb", "run_id",
    ]
    exists = summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(header)
        writer.writerow([
            row.algorithm, row.profile, row.lags, row.horizon, row.model_variant,
            row.avg_inference_time_ms, row.avg_total_time_ms, row.avg_cpu_percent, row.avg_ram_percent,
            row.model_size_mb, row.run_id,
        ])


# ---------------------------
# Aufräumen
# ---------------------------

SCALER_FILE_NAMES = ["scaler.joblib", "y_scaler.joblib"]

def cleanup_scalers(scalers_dir: Path) -> None:
    """Löscht gespeicherte Scaler-Dateien nach dem Lauf. Prediction Data, Trainings-Config usw. bleiben erhalten."""
    if not scalers_dir:
        return
    deleted_any = False
    for fname in SCALER_FILE_NAMES:
        fp = scalers_dir / fname
        if fp.exists():
            try:
                fp.unlink()
                deleted_any = True
                print(f"🧹 Deleted scaler file: {fp}")
            except Exception as e:
                print(f"⚠️ Could not delete scaler file {fp}: {e}")
    if not deleted_any:
        print(f"ℹ️ No scaler files found in {scalers_dir} to delete.")


def cleanup_model_binaries(models_dir: Path) -> None:
    """Löscht große Modellbinaries (bevorzugt Whitelist), belässt Metadaten & CSVs."""
    to_delete = set()
    for p in MODEL_BLOBS_WHITELIST["keras"] + MODEL_BLOBS_WHITELIST["tflite"] + MODEL_BLOBS_WHITELIST["sklearn"] + MODEL_BLOBS_WHITELIST["xgb"]:
        fp = models_dir / p
        if fp.exists():
            to_delete.add(fp)

    for fp in sorted(to_delete):
        try:
            fp.unlink()
            print(f"🧹 Deleted model binary: {fp}")
        except Exception as e:
            print(f"⚠️ Could not delete {fp}: {e}")


# ---------------------------
# Orchestrierung
# ---------------------------

def _run_all_inferences_and_summarize(
    algo: str,
    cfg: dict,
    run_id: str,
    models_dir: Path,
    inference_steps: int,
    loading_strategy: str,
    interval_sec: float,
    summary_csv: Path,
    delete_models_after_inference: bool,
) -> None:
    """Führt für einen Run alle vorhandenen Modellvarianten aus, fasst zusammen, räumt *am Ende* auf."""
    variants = list_model_variants(models_dir, algo)
    if not variants:
        print("⚠️ No model variants found – skipping inference for this run.")
        return

    # Profil (server/edge) ableiten und passenden Config-Variablennamen wählen
    profile_key = "edge" if (cfg.get("edge_device") or cfg.get("enable_edge")) else "server"
    config_name = DEFAULT_PROFILE_VARS.get(algo, {}).get(profile_key, algo)

    for variant in variants:
        rc = run_inference_via_subprocess(
            algorithm=algo,
            run_id=run_id,
            model_filename=variant,
            inference_steps=inference_steps,
            loading_strategy=loading_strategy,
            interval_sec=interval_sec,
            config_name=config_name,
        )
        if rc != 0:
            print(f"⚠️ Inference subprocess returned code {rc} for {variant} (run_id={run_id})")

        # --- Auswertung
        err_dir = Path(cfg["paths"]["Error_Metrics"]) if "paths" in cfg else (Path(CONFIG_PATH["paths"]["output"]) / "Error_Metrics")
        pred_dir = Path(cfg["paths"].get("Prediction_Data"))
        step_csv = _discover_predictions_file_from_json(run_id, err_dir) or _fallback_find_step_csv(run_id, pred_dir)

        avg_inf_ms, avg_total_ms, avg_cpu, avg_ram = (None, None, None, None)
        if step_csv and step_csv.exists():
            avg_inf_ms, avg_total_ms, avg_cpu, avg_ram = summarize_step_csv(step_csv)
        else:
            print("⚠️ StepPredictions CSV not found for run – summary metrics will be empty.")

        # Trainings-Config lesen (Modellgröße)
        training_cfg_json = Path(cfg["paths"]["Models"]) / "training_config.json"
        model_size_mb = read_model_size_mb(training_cfg_json)

        res = InferenceResult(
            algorithm=algo,
            profile="edge" if cfg.get("edge_device") or cfg.get("enable_edge") else "server",
            lags=int(cfg.get("lags", 0)),
            horizon=int(cfg.get("horizon", 0)),
            model_variant=variant,
            avg_inference_time_ms=avg_inf_ms,
            avg_total_time_ms=avg_total_ms,
            avg_cpu_percent=avg_cpu,
            avg_ram_percent=avg_ram,
            model_size_mb=model_size_mb,
            run_id=run_id,
        )
        append_summary_row(summary_csv, res)
        print(f"📈 Summary row appended for {variant} -> {summary_csv}")

    # --- Aufräumen *nach* allen Varianten (wichtiger Fix: Scaler erst am Ende löschen!)
    if delete_models_after_inference:
        cleanup_model_binaries(models_dir)
        try:
            scalers_dir = Path(cfg["paths"]["Scalers"]) if "paths" in cfg else None
            if scalers_dir:
                cleanup_scalers(scalers_dir)
        except Exception as e:
            print(f"⚠️ Could not clean scalers: {e}")


def run_experiments(
    algorithms: List[str],
    profiles: List[str],
    lags_values: List[int],
    horizon_values: List[int],
    inference_steps: int = 20,
    loading_strategy: str = "live_mqtt",
    interval_sec: float = 1.0,
    delete_models_after_inference: bool = True,
    limit_runs: Optional[int] = None,
) -> None:
    out_dir = Path(CONFIG_PATH["paths"]["output"]) / "Error_Metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / SUMMARY_CSV_NAME

    runs_done = 0

    for algorithm in algorithms:
        algo = algorithm.lower()
        assert algo in TRAINER_MAP, f"Unsupported algorithm: {algorithm}"

        for lags in lags_values:
            for horizon in horizon_values:
                if limit_runs is not None and runs_done >= limit_runs:
                    print("Reached run limit – stopping queue.")
                    return

                # --- SERVER train + infer (falls gewünscht)
                if "server" in profiles:
                    base_cfg_s, used_var_s = load_profile_config(algo, "server")
                    folder_flag_s = algorithm_to_folder(algo)
                    cfg_s = build_training_config(base_cfg_s, "server", lags, horizon, folder_flag_s)

                    print(f"\n=== Train {algo} | server | lags={lags} | horizon={horizon} | cfg={used_var_s}")
                    run_id_s, models_dir_s = run_training(algo, cfg_s, folder_flag_s)
                    print(f"✅ Training complete. run_id={run_id_s}")

                    _run_all_inferences_and_summarize(
                        algo, cfg_s, run_id_s, models_dir_s,
                        inference_steps, loading_strategy, interval_sec,
                        summary_csv, delete_models_after_inference
                    )

                # --- EDGE train + infer (falls gewünscht)
                if "edge" in profiles:
                    base_cfg_e, used_var_e = load_profile_config(algo, "edge")
                    folder_flag_e = algorithm_to_folder(algo)
                    cfg_e = build_training_config(base_cfg_e, "edge", lags, horizon, folder_flag_e)

                    print(f"\n=== Train {algo} | edge | lags={lags} | horizon={horizon} | cfg={used_var_e}")
                    run_id_e, models_dir_e = run_training(algo, cfg_e, folder_flag_e)
                    print(f"✅ Training complete. run_id={run_id_e}")

                    _run_all_inferences_and_summarize(
                        algo, cfg_e, run_id_e, models_dir_e,
                        inference_steps, loading_strategy, interval_sec,
                        summary_csv, delete_models_after_inference
                    )

                runs_done += 1


# ---------------------------
# CLI
# ---------------------------

def main():
    p = argparse.ArgumentParser(description="Experiment-Pipeline (Training -> Inferenz) mit Grid über Lags/Horizon")
    p.add_argument("--algorithms", default="cnn1d,lstm,random_forest,xgboost,light_xgboost", help="Kommagetrennte Liste: cnn1d,lstm,random_forest,xgboost,light_xgboost")
    p.add_argument("--profiles", default="server,edge", help="Kommagetrennte Liste: server,edge")
    p.add_argument("--lags", default="1:20:2", help="Range 'start:stop:step' oder kommagetrennt (z. B. 1,3,5)")
    p.add_argument("--horizon", default="1:20:2", help="Range 'start:stop:step' oder kommagetrennt")
    p.add_argument("--inference-steps", type=int, default=60, help="Anzahl Inferenzschritte (1 Hz -> 60 == 1 Minute)")
    p.add_argument("--loading-strategy", default="live_mqtt", choices=["live_mqtt", "split"], help="Datenquelle für die Inferenz")
    p.add_argument("--interval-sec", type=float, default=1.0, help="Ziel-Inferenzintervall in Sekunden")
    p.add_argument("--keep-models", action="store_true", help="Modellbinaries nach Inferenz NICHT löschen")
    p.add_argument("--limit-runs", type=int, default=None, help="Max. Anzahl Experimente (Debug)")

    args = p.parse_args()

    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()]
    lags_vals = _range_from_str(args.lags)
    horizon_vals = _range_from_str(args.horizon)

    run_experiments(
        algorithms=algorithms,
        profiles=profiles,
        lags_values=lags_vals,
        horizon_values=horizon_vals,
        inference_steps=args.inference_steps,
        loading_strategy=args.loading_strategy,
        interval_sec=args.interval_sec,
        delete_models_after_inference=(not args.keep_models),
        limit_runs=args.limit_runs,
    )


if __name__ == "__main__":
    main()