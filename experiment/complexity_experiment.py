#!/usr/bin/env python3
# complexity_experiment.py

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List

"""
Experiment-Pipeline für Modell-Komplexität
-------------------------------------------

Zweck:
  * Führt für verschiedene Algorithmen (LSTM, CNN1D, RF, etc.) Experimente
    mit drei vordefinierten Komplexitätsstufen durch: 'einfach', 'mittel', 'komplex'.
  * Iteriert für jede Modell-Komplexitäts-Kombination über eine Reihe von Horizont-Werten.
  * Verwendet die bestehende `experiment_pipeline.py` als "Worker", um die eigentlichen
    Trainings- und Inferenzläufe durchzuführen.
  * Die Konfigurationen sind direkt in diesem Skript definiert, um eine einfache
    Anpassung zu ermöglichen.

Hinweis:
  Die "komplexen" Konfigurationen sind so gewählt, dass sie anspruchsvoller
  als die "mittleren" sind, aber immer noch eine realistische Chance haben,
  auf einem Edge-Gerät (wie dem RevPi) trainiert zu werden.

Beispielhafter Aufruf:
  python complexity_experiment.py --models lstm cnn1d --horizon 1:5:2
"""

# ---- Basiskonfiguration für alle Edge-Experimente ----
# Diese Werte werden von den spezifischen Modell-Configs übernommen
_COMMON_EDGE_CONFIG = {
    # Artefakt- und Datensatz-Setup
    "dataset": "mqtt_data_filtered.csv",
    "loading_strategy": "split",
    "train_fraction": 0.8,
    "target_feature": "Group4-2_S6_VolumetricFlowRate",
    "base_features": ["Group4-2_S6_VolumetricFlowRate", "Group4-2_S6_MassFlowRate"],
    "time_features": [],

    # Skalierung
    "scale_other_features": True,
    "scale_target": True,
    "scaler_type": "robust",

    # Feature Engineering Flags
    "include_roll_mean": True,
    "include_roll_std": False, # Standardmäßig für Edge deaktiviert

    # Pipeline-Flags
    "edge_device": True,
    "enable_edge": True,
    "validation_fraction": 0.2,
    "early_stopping_patience": 10,
}

# ---- Definition der Modell-Komplexitäten ----
MODEL_CONFIGS = {
    "lstm": {
        "einfach": {
            **_COMMON_EDGE_CONFIG, "model_name": "lstm_einfach",
            "num_layers": 1, "initial_units": 24, "dropout": 0.2,
            "epochs": 50, "batch_size": 64, "optimizer": "adam", "loss": "mse",
        },
        "mittel": {
            **_COMMON_EDGE_CONFIG, "model_name": "lstm_mittel",
            "num_layers": 1, "initial_units": 45, "dropout": 0.33,
            "epochs": 52, "batch_size": 64, "optimizer": "nadam", "loss": "mse",
        },
        "komplex": {
            **_COMMON_EDGE_CONFIG, "model_name": "lstm_komplex",
            "num_layers": 2, "initial_units": 32, "dropout": 0.35, # Layer-Größen: 32 -> 16
            "epochs": 60, "batch_size": 32, "optimizer": "nadam", "loss": "mse",
        },
    },
    "cnn1d": {
        "einfach": {
            **_COMMON_EDGE_CONFIG, "model_name": "cnn1d_einfach",
            "cnn_blocks": 1, "cnn_base_filters": 32, "cnn_kernel_size": 3, "cnn_dropout": 0.1,
            "epochs": 40, "batch_size": 64, "optimizer": "adam", "loss": "huber",
        },
        "mittel": {
            **_COMMON_EDGE_CONFIG, "model_name": "cnn1d_mittel",
            "cnn_blocks": 1, "cnn_base_filters": 53, "cnn_kernel_size": 4, "cnn_dropout": 0.06,
            "epochs": 45, "batch_size": 64, "optimizer": "adam", "loss": "huber",
        },
        "komplex": {
            **_COMMON_EDGE_CONFIG, "model_name": "cnn1d_komplex",
            "cnn_blocks": 2, "cnn_base_filters": 64, "cnn_kernel_size": 5, "cnn_dropout": 0.15,
            "epochs": 50, "batch_size": 32, "optimizer": "adam", "loss": "huber",
        },
    },
    "random_forest": {
        "einfach": {
            **_COMMON_EDGE_CONFIG, "model_name": "rf_einfach",
            "n_estimators": 100, "max_depth": 4, "min_samples_leaf": 10, "n_jobs": -1, "random_state": 42,
        },
        "mittel": {
            **_COMMON_EDGE_CONFIG, "model_name": "rf_mittel",
            "n_estimators": 202, "max_depth": 5, "min_samples_leaf": 7, "n_jobs": -1, "random_state": 42,
        },
        "komplex": {
            **_COMMON_EDGE_CONFIG, "model_name": "rf_komplex",
            "n_estimators": 300, "max_depth": 7, "min_samples_leaf": 4, "n_jobs": -1, "random_state": 42,
        },
    },
    "xgboost": {
        "einfach": {
            **_COMMON_EDGE_CONFIG, "model_name": "xgb_einfach",
            "n_estimators": 200, "max_depth": 2, "learning_rate": 0.02, "subsample": 0.7, "n_jobs": -1, "random_state": 42,
        },
        "mittel": {
            **_COMMON_EDGE_CONFIG, "model_name": "xgb_mittel",
            "n_estimators": 381, "max_depth": 3, "learning_rate": 0.013, "subsample": 0.6, "n_jobs": -1, "random_state": 42,
        },
        "komplex": {
            **_COMMON_EDGE_CONFIG, "model_name": "xgb_komplex",
            "n_estimators": 500, "max_depth": 4, "learning_rate": 0.01, "subsample": 0.8, "n_jobs": -1, "random_state": 42,
        },
    },
    "light_xgboost": { # Basiert auf LightGBM, nutzt aber XGBoost-Trainer
        "einfach": {
            **_COMMON_EDGE_CONFIG, "model_name": "lgbm_einfach",
            "n_estimators": 150, "num_leaves": 10, "learning_rate": 0.02, "n_jobs": -1, "random_state": 42,
        },
        "mittel": {
            **_COMMON_EDGE_CONFIG, "model_name": "lgbm_mittel",
            "n_estimators": 270, "num_leaves": 19, "learning_rate": 0.017, "n_jobs": -1, "random_state": 42,
        },
        "komplex": {
            **_COMMON_EDGE_CONFIG, "model_name": "lgbm_komplex",
            "n_estimators": 400, "num_leaves": 31, "learning_rate": 0.01, "n_jobs": -1, "random_state": 42,
        },
    },
}

# ---- HILFSFUNKTIONEN ----
def _range_from_str(spec: str) -> List[int]:
    """Wandelt 'start:stop:step' oder eine durch Komma getrennte Liste in int-Liste um."""
    spec = spec.strip()
    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        start, stop, step = (parts[0], parts[1], 1) if len(parts) == 2 else parts
        return list(range(start, stop + 1, step))
    return [int(x) for x in spec.split(",") if x]

# ---- PIPELINE-LOGIK ----
def run_single_experiment(
    model_name: str,
    complexity: str,
    horizon: int,
    lags: int,
    inference_steps: int,
    quant_modes: List[str]
):
    """Führt einen einzelnen Lauf der experiment_pipeline.py als Subprozess aus."""
    print("\n" + "="*80)
    print(f"== Starte: {model_name.upper()} | Komplexität: {complexity.upper()} | Horizont: {horizon}")
    print("="*80 + "\n")

    # Pfad zur Worker-Pipeline
    script_dir = Path(__file__).resolve().parent
    pipeline_script = script_dir / "experiment_pipeline.py"
    if not pipeline_script.exists():
        print(f"❌ FEHLER: Worker-Skript nicht gefunden unter {pipeline_script}")
        sys.exit(1)

    # Lade die Basiskonfiguration für diesen Lauf
    config = MODEL_CONFIGS[model_name][complexity]
    
    # Baue den Befehl für den Subprozess
    command = [
        sys.executable, str(pipeline_script),
        "--algorithms", model_name,
        "--profiles", "edge",
        "--lags", str(lags),
        "--horizon", str(horizon),
        "--inference-steps", str(inference_steps),
    ]

    # Füge Quantisierungsmodus hinzu
    command.extend(["--quant-mode", *quant_modes])

    # Überschreibe die Default-Konfiguration der Worker-Pipeline mit unseren
    # spezifischen Parametern für diesen Lauf.
    # HINWEIS: 'model_params' wird hier bewusst nicht gesetzt, da die Trainer
    # auf die Top-Level-Parameter zugreifen (z.B. n_estimators).
    params_to_set = {k: v for k, v in config.items() if not isinstance(v, (dict, list))}
    for key, value in params_to_set.items():
        command.extend(["--set", f"{key}={value}"])

    try:
        # Führe den Befehl aus und warte auf das Ergebnis
        subprocess.run(command, check=True)
        print(f"\n--- ✅ ERFOLG: {model_name.upper()} | {complexity.upper()} | Horizont: {horizon} ---\n")
    except subprocess.CalledProcessError as e:
        print(f"\n--- ❌ FEHLER: {model_name.upper()} | {complexity.upper()} | Horizont: {horizon} ---")
        print(f"--- ❌ Exit-Code: {e.returncode}. Breche die gesamte Pipeline ab. ---")
        sys.exit(1)


# ---- HAUPTPROGRAMM ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment-Pipeline für Modell-Komplexität.")
    parser.add_argument(
        "--models", nargs='+', required=True,
        choices=MODEL_CONFIGS.keys(),
        help="Liste der zu testenden Modelle."
    )
    parser.add_argument(
        "--horizon", default="1:5:2",
        help="Horizont-Werte als 'start:stop:step' oder '1,3,5'."
    )
    parser.add_argument(
        "--lags", type=int, default=15,
        help="Ein fester Lag-Wert für alle Experimente."
    )
    parser.add_argument(
        "--inference-steps", type=int, default=50,
        help="Anzahl der Inferenz-Schritte pro Lauf."
    )
    parser.add_argument(
        "--quant-mode", nargs='+', default=["quant-16"],
        choices=["no-quant", "quant-16", "quant-8"],
        help="Quantisierungsmodus(e) für alle Läufe."
    )
    args = parser.parse_args()

    horizon_values = _range_from_str(args.horizon)

    # --- Start der Experiment-Schleife ---
    total_runs = len(args.models) * 3 * len(horizon_values)
    current_run = 0

    for model in args.models:
        for complexity_level in ["einfach", "mittel", "komplex"]:
            for h_val in horizon_values:
                current_run += 1
                print(f"\n\n>>> Starte Gesamtlauf {current_run} von {total_runs} <<<")
                run_single_experiment(
                    model_name=model,
                    complexity=complexity_level,
                    horizon=h_val,
                    lags=args.lags,
                    inference_steps=args.inference_steps,
                    quant_modes=args.quant_mode
                )

    print("\n" + "="*80)
    print("🎉 Alle konfigurierten Komplexitäts-Experimente wurden ausgeführt. 🎉")
    print("="*80 + "\n")