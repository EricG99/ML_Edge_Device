import subprocess
import sys
from pathlib import Path # KORREKTUR: pathlib für robuste Pfade importieren

# --- Konfiguration der Experimente ---
MODELS_TO_RUN = [
    "random_forest",
    "cnn1d",
    "lstm",

    "xgboost",
    "light_xgboost"
]

# --- Gemeinsame Parameter für alle Läufe ---

# KORREKTUR: Den Pfad zum Skript dynamisch und robust ermitteln
# Das Skript findet nun immer die "experiment_pipeline.py", die im selben Ordner liegt.
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = SCRIPT_DIR / "experiment_pipeline.py"

LAGS = "1:15:2"  # Lags von 1 bis 15 in 2er-Schritten
HORIZON = "1:15:2"  # Horizon von 1 bis 15 in 2er-Schritten
PROFILE = "edge"
LOADING_STRATEGY = "live_mqtt"
INFERENCE_STEPS = "50"


def run_single_experiment(model_name: str):
    """
    Erstellt den Befehl für ein einzelnes Experiment und führt ihn als separaten Prozess aus.
    """
    print("\n" + "="*70)
    print(f"== Starte Experiment für das Modell: {model_name.upper()}")
    print("="*70 + "\n")

    # Baut den Befehl zusammen
    command = [
        sys.executable,  # Stellt sicher, dass der gleiche Python-Interpreter verwendet wird
        str(PIPELINE_SCRIPT), # KORREKTUR: Den vollen Pfad verwenden
        "--algorithms", model_name,
        "--profiles", PROFILE,
        "--lags", LAGS,
        "--horizon", HORIZON,
        "--loading-strategy", LOADING_STRATEGY,
        "--inference-steps", INFERENCE_STEPS
    ]

    try:
        # Führt den Befehl aus und wartet, bis er abgeschlossen ist.
        # check=True sorgt dafür, dass das Skript bei einem Fehler im Unterprozess abbricht.
        subprocess.run(command, check=True)
        print(f"\n--- ✅ Experiment für {model_name} erfolgreich abgeschlossen. ---\n")
    except subprocess.CalledProcessError as e:
        print(f"\n--- ❌ FEHLER: Experiment für {model_name} ist mit Exit-Code {e.returncode} fehlgeschlagen. ---")
        # Um trotz Fehler weiterzumachen, kommentieren Sie die nächste Zeile aus
        sys.exit(1)
    except FileNotFoundError:
        print(f"\n--- ❌ FEHLER: Das Skript '{PIPELINE_SCRIPT}' konnte nicht gefunden werden. ---")
        print("Stellen Sie sicher, dass sich beide Skripte im selben Ordner befinden.")
        sys.exit(1)


if __name__ == "__main__":
    # KORREKTUR: Sicherstellen, dass die Zieldatei existiert, bevor die Schleife startet
    if not PIPELINE_SCRIPT.is_file():
        print(f"--- ❌ FEHLER: Die Zieldatei '{PIPELINE_SCRIPT}' existiert nicht. ---")
        sys.exit(1)
        
    for model in MODELS_TO_RUN:
        run_single_experiment(model)

    print("\n" + "="*70)
    print("== Alle konfigurierten Experimente wurden ausgeführt. ==")
    print("="*70 + "\n")