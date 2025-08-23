import subprocess
import sys
from pathlib import Path

# --- Konfiguration der Experimente ---
MODELS_TO_RUN = [

    "lstm",
    "random_forest",
    "xgboost",
    "light_xgboost",
    "cnn1d",
]

# --- Gemeinsame Parameter für alle Läufe ---
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = SCRIPT_DIR / "experiment_pipeline.py"

LAGS = "1:16:3"
HORIZON = "1:16:3"
PROFILE = "edge"
LOADING_STRATEGY = "live_mqtt"
INFERENCE_STEPS = "60"

# NEU: Steuern Sie hier die Quantisierungsmodi für alle Experimente
# Mögliche Werte: "no-quant", "quant-16", "quant-8"
# Sie können auch mehrere angeben, z.B. ["quant-16", "quant-8"]
QUANT_MODES = ["quant-16"]


def run_single_experiment(model_name: str):
    """
    Erstellt den Befehl für ein einzelnes Experiment und führt ihn als separaten Prozess aus.
    """
    print("\n" + "="*70)
    print(f"== Starte Experiment für das Modell: {model_name.upper()}")
    print("="*70 + "\n")

    command = [
        sys.executable,
        str(PIPELINE_SCRIPT),
        "--algorithms", model_name,
        "--profiles", PROFILE,
        "--lags", LAGS,
        "--horizon", HORIZON,
        "--loading-strategy", LOADING_STRATEGY,
        "--inference-steps", INFERENCE_STEPS
    ]
    
    # NEU: Fügt das --quant-mode Argument hinzu, falls definiert
    if QUANT_MODES:
        command.append("--quant-mode")
        command.extend(QUANT_MODES)

    try:
        subprocess.run(command, check=True)
        print(f"\n--- ✅ Experiment für {model_name} erfolgreich abgeschlossen. ---\n")
    except subprocess.CalledProcessError as e:
        print(f"\n--- ❌ FEHLER: Experiment für {model_name} ist mit Exit-Code {e.returncode} fehlgeschlagen. ---")
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n--- ❌ FEHLER: Das Skript '{PIPELINE_SCRIPT}' konnte nicht gefunden werden. ---")
        sys.exit(1)


if __name__ == "__main__":
    if not PIPELINE_SCRIPT.is_file():
        print(f"--- ❌ FEHLER: Die Zieldatei '{PIPELINE_SCRIPT}' existiert nicht. ---")
        sys.exit(1)
        
    for model in MODELS_TO_RUN:
        run_single_experiment(model)

    print("\n" + "="*70)
    print("== Alle konfigurierten Experimente wurden ausgeführt. ==")
    print("="*70 + "\n")