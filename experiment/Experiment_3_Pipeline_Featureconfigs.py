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
    outfile = output_dir / "FeatureConfig_RunMeta.csv"
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

def quant_modes_for_algorithm(algo: str) -> List[str]:
    """Quantisierungsmodi je Algorithmus."""
    a = algo.lower()
    if a in ("cnn1d", "lstm"):
        return ["no-quant", "quant-16", "quant-8"]
    return ["no-quant"]

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
                summary_csv = out_dir / "Experiment_Summary_Server_featureconfigs.csv"
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

    return p

def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

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

