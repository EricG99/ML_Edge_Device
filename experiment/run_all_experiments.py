import subprocess
import sys
from pathlib import Path

# --- Konfiguration der Experimente ---
# Modelle gemäß deinen Logs
EXPERIMENTS_TO_RUN = [
    {"model": "light_xgboost", "profile": "edge"},
    {"model": "lstm",          "profile": "edge"},
    {"model": "cnn1d",         "profile": "edge"},
    {"model": "random_forest", "profile": "edge"},
    {"model": "xgboost",       "profile": "edge"},
]

# --- Gemeinsame Parameter für alle Läufe ---
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = SCRIPT_DIR / "experiment_pipeline.py"

# Vorgaben laut Wunsch
LAGS = "1:16:3"
HORIZON = "1:16:3"
LOADING_STRATEGY = "live_mqtt" #live_mqtt
INFERENCE_STEPS = "60"

# Quantisierung (wie gehabt)
# Mögliche Werte: "no-quant", "quant-16", "quant-8"
QUANT_MODES = ["no-quant", "quant-16", "quant-8"]

def run_single_experiment(model_name: str, profile: str):
    config_name = f"{model_name.upper()} mit Profil '{profile}'"
    print("\n" + "="*70)
    print(f"== Starte Experiment für: {config_name}")
    print("="*70 + "\n")

    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--algorithms", model_name,
        "--profiles", profile,
        "--lags", LAGS,
        "--horizon", HORIZON,
        "--loading-strategy", LOADING_STRATEGY,
        "--inference-steps", INFERENCE_STEPS,
    ]

    if QUANT_MODES:
        command.append("--quant-mode")
        command.extend(QUANT_MODES)

    try:
        subprocess.run(command, check=True)
        print(f"\n--- ✅ Experiment für {config_name} erfolgreich abgeschlossen. ---\n")
    except subprocess.CalledProcessError as e:
        print(f"\n--- ❌ FEHLER: Experiment für {config_name} ist mit Exit-Code {e.returncode} fehlgeschlagen. ---")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n--- ❌ FEHLER: Das Skript '{PIPELINE_SCRIPT}' konnte nicht gefunden werden. ---")
        sys.exit(1)

if __name__ == "__main__":
    if not PIPELINE_SCRIPT.is_file():
        print(f"--- ❌ FEHLER: Die Zieldatei '{PIPELINE_SCRIPT}' existiert nicht. ---")
        sys.exit(1)

    for exp in EXPERIMENTS_TO_RUN:
        run_single_experiment(model_name=exp["model"], profile=exp["profile"])

    print("\n" + "="*70)
    print("== Alle konfigurierten Experimente wurden ausgeführt. ==")
    print("="*70 + "\n")
