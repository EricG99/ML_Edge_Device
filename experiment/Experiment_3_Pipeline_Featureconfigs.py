#!/usr/bin/env python3
from __future__ import annotations
"""
Experiment-Pipeline (Feature-Configs)
------------------------------------

Zweck
  * Führt Experimente über einen Horizont-Grid **und genau drei Feature-Konfigurationen** aus.
  * Verwendet **ausschließlich Level 'medium'** der bestehenden Modell-Presets
    und ruft Training + Inferenz + Summary aus `experiment_pipeline_multiconfig.py` auf.
  * Ziel: **Inferenzzeiten** der drei Feature-Configs (auf RevPi) vergleichen.

Artefakte
  * Mapping (run_id → Feature-Config) in: `FeatureConfig_RunMeta.csv`
  * Inferenz-Summary: `Experiment_Summary_Server_featureconfigs.csv`

Aufrufbeispiele siehe Modul-Docstring am Ende oder `-h`.
"""

import argparse
import csv
import time
from pathlib import Path
from typing import List, Dict, Optional
import os, subprocess, json
import pandas as pd

# =========================
# 1) ZENTRALE FEATURE-DEFS
# =========================
# Rohspalten (können per CLI überschrieben werden)
DEFAULT_VOLUME_COL = "Group4-2_S6_VolumetricFlowRate"  # Volumenstrom
DEFAULT_PRESS_COL  = "Group4-2_S6_Pressure"            # Druck
DEFAULT_TEMP_COL   = "Group4-2_S6_Temperature"         # Temperatur

# Lags & Rolling Defaults
DEFAULT_LAGS = 20
DEFAULT_ROLLING_WINDOW = 10  # nur wirksam, wenn rolling aktiviert

# Drei feste Konfigurationen (werden nach CLI-Parsing mit den evtl. überschriebenen Spalten befüllt)
def make_feature_configs(volume_col: str, pressure_col: str, temperature_col: str) -> List[Dict]:
    """
    3 Feature-Configs für den RevPi-Vergleich:
      F1: Minimal – nur Volumenstrom, ohne Rolling
      F2: + Druck – ohne Rolling
      F3: + Temperatur – mit Rolling (mean & std) für alle Basis-Features
    """
    return [
        {
            "name": "F1_volumenstrom_only_no_rolling",
            "base_features": [volume_col],
            "rolling_enabled": False,
            "rolling_features": [],
            "rolling_on": [],   # explizit keine Rolling-Spalten
        },
        {
            "name": "F2_volumenstrom_druck_no_rolling",
            "base_features": [volume_col, pressure_col],
            "rolling_enabled": False,
            "rolling_features": [],
            "rolling_on": [],
        },
        {
            "name": "F3_volumenstrom_druck_temp_rolling_mean_std",
            "base_features": [volume_col, pressure_col, temperature_col],
            "rolling_enabled": True,
            "rolling_features": ["mean", "std"],
            "rolling_on": "ALL",  # "ALL" = für alle base_features Rolling erzeugen
        },
    ]

# =========================
# 2) Pipeline-Importe
# =========================
from experiment_pipeline_multiconfig import (
    build_training_config,
    run_training,
    algorithm_to_folder,
    _run_all_inferences_and_summarize,
    _range_from_str,
)

# =========================
# 3) Hilfsfunktionen
# =========================
def append_feature_runmeta(output_dir: Path, row: Dict) -> None:
    """Hängt eine Zeile in FeatureConfig_RunMeta.csv an (wird neu angelegt, falls nicht vorhanden)."""
    outfile = output_dir / "FeatureConfig_RunMeta_V2_test.csv"
    header = [
        "timestamp", "run_id", "algorithm", "level", "horizon", "lags",
        "feature_config_name", "base_features", "rolling_enabled", "rolling_features",
        "rolling_on", "volume_col", "pressure_col", "temperature_col",
    ]
    exists = outfile.exists()
    with open(outfile, "a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        if not exists:
            wr.writerow(header)
        wr.writerow([
            row.get("timestamp"), row.get("run_id"), row.get("algorithm"), row.get("level"),
            row.get("horizon"), row.get("lags"), row.get("feature_config_name"),
            ",".join(row.get("base_features", []) or []), row.get("rolling_enabled"),
            ",".join(row.get("rolling_features", []) or []),
            row.get("rolling_on"),
            row.get("volume_col"), row.get("pressure_col"), row.get("temperature_col"),
        ])

def _normalize_q(label: str) -> str:
    l = (label or "").strip().lower()
    if l in ("quant-8","int8","q8","8","quant8"):
        return "quant-8"
    if l in ("quant-16","fp16","float16","q16","16","quant16"):
        return "quant-16"
    return "no-quant"

def quant_modes_for_algorithm(algo: str, requested: Optional[str] = None) -> List[str]:
    """Quantisierungsmodi je Algorithmus (mit optionalem Override via --quant-modes)."""
    a = (algo or "").lower()
    base = ["no-quant", "quant-16", "quant-8"] if a in ("cnn1d", "lstm") else ["no-quant"]
    if requested:
        wanted = [_normalize_q(x) for x in requested.split(",") if x.strip()]
        filt = [m for m in base if m in wanted]
        if filt:
            return filt
    return base


def _resolve_python_executable() -> str:
    import shutil, sys, os
    if sys.executable and os.path.exists(sys.executable):
        return sys.executable
    for name in ("python3", "python"):
        p = shutil.which(name)
        if p: return p
    return "python"
def _pipeline_script_path() -> str:
    # .../experiment/Experiment_3_Pipeline_Featureconfigs.py → .../ML_Algorithms/pipeline_web_app.py
    here = Path(__file__).resolve().parent
    return str((here.parent / "ML_Algorithms" / "pipeline_web_app.py").resolve())

def _pick_tflite(models_dir: Path) -> Optional[str]:
    """Bevorzuge INT8, sonst FP16; liefere Dateinamen (ohne Pfad) oder None."""
    ints = sorted(models_dir.glob("*int8*.tflite")) + sorted(models_dir.glob("*quant8*.tflite")) + sorted(models_dir.glob("*q8*.tflite"))
    fp16 = sorted(models_dir.glob("*fp16*.tflite")) + sorted(models_dir.glob("*float16*.tflite")) + sorted(models_dir.glob("*quant16*.tflite")) + sorted(models_dir.glob("*16*.tflite"))
    for f in ints + fp16:
        return f.name
    return None

def _find_step_csv(run_dir: Path) -> Optional[str]:
    """Suche eine Preditions-Step-CSV im Run-Ordner."""
    for nm in ("inference_summary.json", "predictions_meta.json"):
        fp = run_dir / nm
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8")) or {}
                step_csv = data.get("step_csv") or data.get("predictions_step_csv")
                if step_csv:
                    cand = Path(step_csv) if os.path.isabs(step_csv) else (run_dir / step_csv)
                    if cand.exists():
                        return str(cand)
            except Exception:
                pass
    for root, _, files in os.walk(str(run_dir)):
        for fn in files:
            low = fn.lower()
            if low.endswith("_predictions_step.csv") or ("predictions" in low and "step" in low and low.endswith(".csv")):
                return str(Path(root) / fn)
    return None

def _summarize_step_csv(step_csv: str) -> Dict[str, float]:
    try:
        df = pd.read_csv(step_csv)
    except Exception:
        return {}
    res = {}
    for c in ("inference_time_ms","total_time_ms","cpu_percent","memory_percent"):
        if c in df.columns:
            res[f"avg_{c}"] = float(pd.to_numeric(df[c], errors="coerce").mean())
    return res

def _append_summary(summary_csv: Path, row: Dict) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["timestamp","run_id","algorithm","feature_config_name","horizon","lags",
            "step_csv","avg_inference_time_ms","avg_total_time_ms","avg_cpu_percent","avg_memory_percent","note"]
    write_header = not summary_csv.exists()
    with open(summary_csv, "a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        if write_header:
            wr.writerow(cols)
        wr.writerow([row.get(k,"") for k in cols])

def _run_tflite_inference_and_summarize(*, algo: str, cfg: Dict, run_id: str, inference_steps: int,
                                        loading_strategy: str, interval_sec: float, summary_csv: Path) -> None:
    base_output = Path(cfg["paths"]["output"]).resolve()
    folder_flag = algorithm_to_folder(algo)
    run_dir = base_output / folder_flag / run_id
    models_dir = run_dir / "Models"

    # .tflite auswählen
    tfl = _pick_tflite(models_dir)
    if not tfl:
        print(f"[WARN] Keine .tflite in {models_dir} gefunden – überspringe Inferenz von {algo} ({run_id}).")
        return

    # pipeline_web_app.py im Subprozess starten (nur Inferenz, headless)
    script = _pipeline_script_path()
    args = [
        sys.executable, script,
        "--algorithm", algo,
        "--load_id", run_id,
        "--model_filename", tfl,
        "--inference-steps", str(int(inference_steps)),
        "--set", f"loading_strategy={loading_strategy}",
        "--set", f"inference_interval_sec={interval_sec}",
        "--no-web",
    ]
    print("[SPAWN] ", " ".join(args))
    rc = subprocess.call(args)
    note = "ok" if rc == 0 else f"rc={rc}"

    # Step-CSV suchen & Kennzahlen schreiben
    step_csv = _find_step_csv(run_dir)
    metrics = _summarize_step_csv(step_csv) if step_csv else {}
    _append_summary(Path(summary_csv), {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "algorithm": algo,
        "feature_config_name": cfg.get("feature_config_name",""),
        "horizon": int(cfg.get("horizon", 0)),
        "lags": int(cfg.get("lags", 0)),
        "step_csv": step_csv or "",
        "avg_inference_time_ms": metrics.get("avg_inference_time_ms", ""),
        "avg_total_time_ms": metrics.get("avg_total_time_ms", ""),
        "avg_cpu_percent": metrics.get("avg_cpu_percent", ""),
        "avg_memory_percent": metrics.get("avg_memory_percent", ""),
        "note": note,
    })


def _select_primary_model_filename(models_dir: str) -> str:
    """Bevorzugt Keras-Modelle; sonst erstes .tflite."""
    p = Path(models_dir)
    for name in ("model.keras", "model.h5"):
        if (p / name).exists():
            return name
    for f in p.glob("*.tflite"):
        return f.name
    # Fallback: irgendeine Datei
    for f in p.iterdir():
        if f.is_file():
            return f.name
    raise FileNotFoundError(f"Keine Modell-Datei in {models_dir} gefunden.")

def _run_inference_subprocess_with_fallback(*, algorithm: str, run_id: str, model_filename: str,
                                            inference_steps: int, loading_strategy: str,
                                            interval_sec: float, no_web: bool = True) -> int:
    """Startet pipeline_web_app.py als Kindprozess mit --auto-quant-fallback."""
    py = _resolve_python_executable()
    script = _pipeline_script_path()
    args = [
        py, script,
        "--algorithm", str(algorithm),
        "--load_id", str(run_id),
        "--model_filename", str(model_filename),
        "--inference-steps", str(inference_steps),
        "--set", f"loading_strategy={loading_strategy}",
        "--set", f"inference_interval_sec={interval_sec}",
        "--auto-quant-fallback",
    ]
    if no_web:
        args.append("--no-web")
    print("[SPAWN] ", " ".join(args))
    return subprocess.call(args)

def _find_step_csv(run_dir: Path) -> Optional[str]:
    """Sucht rekursiv eine Schritt-CSV (robust)."""
    # 1) bekannte JSON-Metadateien prüfen
    for nm in ("inference_summary.json", "predictions_meta.json"):
        fp = run_dir / nm
        if fp.exists():
            try:
                data = json.loads(fp.read_text(encoding="utf-8")) or {}
                step_csv = data.get("step_csv") or data.get("predictions_step_csv")
                if step_csv:
                    cand = Path(step_csv) if os.path.isabs(step_csv) else (run_dir / step_csv)
                    if cand.exists():
                        return str(cand)
            except Exception:
                pass
    # 2) Pattern-Suche
    for root, _, files in os.walk(str(run_dir)):
        for fn in files:
            low = fn.lower()
            if low.endswith("_predictions_step.csv") or "predictions_step" in low:
                return str(Path(root) / fn)
            if low.endswith(".csv") and "predictions" in low and "step" in low:
                return str(Path(root) / fn)
    return None

def _summarize_step_csv(step_csv: str) -> Dict[str, float]:
    """Berechnet einfache Mittelwerte für die wichtigsten Spalten."""
    try:
        df = pd.read_csv(step_csv)
    except Exception:
        return {}
    res = {}
    for c in ("inference_time_ms", "total_time_ms", "cpu_percent", "memory_percent"):
        if c in df.columns:
            res[f"avg_{c}"] = float(pd.to_numeric(df[c], errors="coerce").mean())
    return res

def _append_summary(summary_csv: Path, row: Dict) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.exists()
    cols = ["timestamp","run_id","algorithm","feature_config_name","horizon","lags",
            "step_csv","avg_inference_time_ms","avg_total_time_ms","avg_cpu_percent","avg_memory_percent",
            "note"]
    with open(summary_csv, "a", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        if write_header:
            wr.writerow(cols)
        wr.writerow([row.get(k,"") for k in cols])

def _run_all_inferences_and_summarize_auto_fallback(
    *, algo: str, cfg: Dict, run_id: str, inference_steps: int,
    loading_strategy: str, interval_sec: float, summary_csv: Path, delete_models: bool
) -> None:
    base_output = Path(cfg["paths"]["output"]).resolve()
    folder_flag = algorithm_to_folder(algo)
    run_dir = base_output / folder_flag / run_id
    models_dir = run_dir / "Models"

    # Primäre Modell-Datei wählen (FP32, falls vorhanden)
    try:
        model_filename = _select_primary_model_filename(str(models_dir))
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # Subprozess mit Auto-Fallback starten (FP32 → FP16 → INT8)
    rc = _run_inference_subprocess_with_fallback(
        algorithm=algo,
        run_id=run_id,
        model_filename=model_filename,
        inference_steps=int(inference_steps),
        loading_strategy=str(loading_strategy),
        interval_sec=float(interval_sec),
        no_web=True,
    )

    note = ""
    if rc == 0:
        note = "ok (auto-fallback möglich)"
    elif rc in (137, 9, -9):
        note = "killed/oom"
    else:
        note = f"rc={rc}"

    # Step-CSV finden und kennzahlen berechnen
    step_csv = _find_step_csv(run_dir)
    metrics = _summarize_step_csv(step_csv) if step_csv else {}

    # Summary-Zeile schreiben
    _append_summary(Path(summary_csv), {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "algorithm": algo,
        "feature_config_name": cfg.get("feature_config_name",""),
        "horizon": int(cfg.get("horizon", 0)),
        "lags": int(cfg.get("lags", 0)),
        "step_csv": step_csv or "",
        "avg_inference_time_ms": metrics.get("avg_inference_time_ms", ""),
        "avg_total_time_ms": metrics.get("avg_total_time_ms", ""),
        "avg_cpu_percent": metrics.get("avg_cpu_percent", ""),
        "avg_memory_percent": metrics.get("avg_memory_percent", ""),
        "note": note,
    })

    # Optional: Modelle löschen, wenn gewünscht
    if delete_models:
        try:
            for f in (models_dir.glob("model*.*")):
                try: f.unlink()
                except Exception: pass
        except Exception:
            pass



# =========================
# 4) Hauptlogik
# =========================
def run_feature_experiments(
    algorithms: List[str],
    horizon_values: List[int],
    inference_steps: int,
    loading_strategy: str,
    interval_sec: float,
    keep_models: bool,
    volume_col: str,
    pressure_col: str,
    temperature_col: str,
    lags: int,
    rolling_window_size: int,
    limit_runs: Optional[int] = None,
) -> None:
    runs_done = 0
    feature_cfgs = make_feature_configs(volume_col=volume_col, pressure_col=pressure_col, temperature_col=temperature_col)

    for algorithm in algorithms:
        algo = algorithm.lower().strip()
        if not algo:
            continue

        qmodes = quant_modes_for_algorithm(algo)
        print(f"\n--- Algorithmus: {algo} (Quantisierungsmodi: {qmodes}) ---")

        for horizon in horizon_values:
            for fcfg in feature_cfgs:
                if limit_runs is not None and runs_done >= limit_runs:
                    print("Maximalzahl an Läufen erreicht – Experiment wird beendet.")
                    return

                # 1) Basiskonfig (Level 'medium') aus bestehender Pipeline
                folder_flag = algorithm_to_folder(algo)
                cfg = build_training_config(
                    algorithm=algo,
                    level="medium",
                    horizon=int(horizon),
                    folder_flag=folder_flag,
                    quant_modes=qmodes,
                )

                # 2) Feature-Overrides je Config
                cfg["level_used"] = "medium"
                cfg["lags"] = int(lags)  # fest 20 (oder CLI)
                cfg["rolling_window_size"] = int(rolling_window_size)
                cfg["base_features"] = list(fcfg["base_features"])
                cfg["feature_config_name"] = fcfg["name"]

                # Rolling-Flags (von Eurer Pipeline auswertbar)
                cfg["enable_rolling_features"] = bool(fcfg["rolling_enabled"])
                cfg["rolling_features"] = list(fcfg["rolling_features"])  # ["mean","std"] oder []
                if fcfg.get("rolling_on") == "ALL":
                    cfg["rolling_on"] = list(cfg["base_features"])
                else:
                    cfg["rolling_on"] = list(fcfg.get("rolling_on", []))

                # 3) Training
                print(
                    f"\n=== LAUF {runs_done + 1} | Train: {algo} | Level: medium | Horizon: {horizon} | "
                    f"Features: {fcfg['name']} ==="
                )
                run_id, models_dir = run_training(algo, cfg, folder_flag)
                print(f"✓ Training abgeschlossen. run_id={run_id}")

                # 4) Meta-Infos speichern
                out_dir = Path(cfg["paths"]["output"]).resolve()
                append_feature_runmeta(
                    out_dir,
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "run_id": run_id,
                        "algorithm": algo,
                        "level": "medium",
                        "horizon": int(horizon),
                        "lags": int(cfg.get("lags", 0)),
                        "feature_config_name": fcfg["name"],
                        "base_features": list(fcfg["base_features"]),
                        "rolling_enabled": bool(fcfg["rolling_enabled"]),
                        "rolling_features": list(fcfg["rolling_features"]),
                        "rolling_on": "ALL" if fcfg.get("rolling_on") == "ALL" else ",".join(fcfg.get("rolling_on", [])),
                        "volume_col": volume_col,
                        "pressure_col": pressure_col,
                        "temperature_col": temperature_col,
                    },
                )

                # 5) Inferenz + Summary
                # 5) Inferenz + Summary
                summary_csv = out_dir / "Experiment_Summary_Server_featureconfigs_v2.csv"

                # Nur quantisierte Inferenz?
                if bool(globals().get("__INFER_Q_ONLY__", False)) and algo in ("lstm","cnn1d"):
                    _run_tflite_inference_and_summarize(
                        algo=algo,
                        cfg=cfg,
                        run_id=run_id,
                        inference_steps=int(inference_steps),
                        loading_strategy=loading_strategy,
                        interval_sec=float(interval_sec),
                        summary_csv=summary_csv,
                    )
                else:
                    _run_all_inferences_and_summarize(
                        algo=algo,
                        cfg=cfg,
                        run_id=run_id,
                        models_dir=models_dir,
                        inference_steps=int(inference_steps),
                        loading_strategy=loading_strategy,
                        interval_sec=float(interval_sec),
                        summary_csv=summary_csv,
                        delete_models=(not keep_models),
                    )
                runs_done += 1

# =========================
# 5) CLI
# =========================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Experiment über Horizont-Grid mit 3 Feature-Konfigurationen (Level=medium). "
            "Vergleich der Inferenzzeiten pro Config/Modell/Quantisierung."
        )
    )
    p.add_argument("--algorithms",
                   default="cnn1d,lstm,random_forest,xgboost,light_xgboost,ridge,svm",
                   help="Kommagetrennte Liste der Algorithmen")
    p.add_argument("--horizon",
                   default="1:8:1",
                   help="Range 'start:stop:step' oder kommagetrennt (z.B. 1:6:1 oder 1,2,3)")
    p.add_argument("--inference-steps", type=int, default=60,
                   help="Anzahl Inferenzschritte pro Lauf")
    p.add_argument("--loading-strategy", default="split", choices=["split", "live_mqtt"],
                   help="Datenquelle für Inferenz")
    p.add_argument("--interval-sec", type=float, default=0.0,
                   help="Ziel-Inferenzintervall in Sekunden")
    p.add_argument("--keep-models", action="store_true",
                   help="Modellbinaries nach Inferenz NICHT löschen")
    p.add_argument("--limit-runs", type=int,
                   help="Max. Anzahl an Läufen (Algo×Horizon×3 Feature-Configs)")

    # Spalten (überschreibbar)
    p.add_argument("--volume-col", default=DEFAULT_VOLUME_COL, help="Spaltenname für Volumenstrom")
    p.add_argument("--pressure-col", default=DEFAULT_PRESS_COL, help="Spaltenname für Druck")
    p.add_argument("--temperature-col", default=DEFAULT_TEMP_COL, help="Spaltenname für Temperatur")

    # Lags / Rolling
    p.add_argument("--lags", type=int, default=DEFAULT_LAGS, help="Anzahl Lags (Standard=20)")
    p.add_argument("--rolling-window-size", type=int, default=DEFAULT_ROLLING_WINDOW,
                   help="Fenstergröße für Rolling-Features (falls aktiviert)")
    
    p.add_argument("--auto-quant-fallback", action="store_true",
               help="Inferenz per Subprozess starten: erst no-quant, bei Kill/Fehler automatisch quant-16 und quant-8 versuchen.")

    p.add_argument("--quant-modes", default="",
               help="Kommagetrennt: quant-8,quant-16 (überschreibt Default je Algo; nur für lstm/cnn1d).")
    p.add_argument("--inference-quant-only", action="store_true",
                help="Inferenz nur mit .tflite (quant-8/quant-16); FP32 (.keras/.h5) wird übersprungen.")
    return p

def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    # globaler Schalter für die obige if-Verzweigung
    requested = globals().get("__REQ_QMODES__", "")
    qmodes = quant_modes_for_algorithm(algo, requested=requested)


    algorithms = [a.strip() for a in (args.algorithms or "").split(",") if a.strip()]
    horizon_vals = _range_from_str(args.horizon)

    run_feature_experiments(
        algorithms=algorithms,
        horizon_values=horizon_vals,
        inference_steps=int(args.inference_steps),
        loading_strategy=args.loading_strategy,
        interval_sec=float(args.interval_sec),
        keep_models=bool(args.keep_models),
        volume_col=args.volume_col,
        pressure_col=args.pressure_col,
        temperature_col=args.temperature_col,
        lags=int(args.lags),
        rolling_window_size=int(args.rolling_window_size),
        limit_runs=args.limit_runs,
    )

if __name__ == "__main__":
    main()


"""
Aufrufbeispiele
===============

Windows (PowerShell)
--------------------
# aus dem Projekt-Root
python .\\experiment_pipeline_featureconfigs.py `
  --algorithms "cnn1d,lstm,xgboost,light_xgboost" `
  --horizon "1:6:1" `
  --inference-steps 60 `
  --loading-strategy split `
  --interval-sec 0.0 `
  --volume-col "Group4-2_S6_VolumetricFlowRate" `
  --pressure-col "Group4-2_S6_Pressure" `
  --temperature-col "Group4-2_S6_Temperature"

Linux/macOS (Bash)
------------------
python ./experiment_pipeline_featureconfigs.py \
  --algorithms "cnn1d,lstm,xgboost,light_xgboost" \
  --horizon "1:6:1" \
  --inference-steps 60 \
  --loading-strategy split \
  --interval-sec 0.0 \
  --volume-col "Group4-2_S6_VolumetricFlowRate" \
  --pressure-col "Group4-2_S6_Pressure" \
  --temperature-col "Group4-2_S6_Temperature"

Anmerkungen
-----------
* Inferenzzeiten je Config findest du in `Experiment_Summary_Server_featureconfigs.csv`
  (ggf. nach `feature_config_name` gruppieren).
* run_id↔Config-Mapping in `FeatureConfig_RunMeta.csv`.
"""

