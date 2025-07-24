import subprocess
import shlex
import os

# --- Konfiguration ---
REMOTE_USER = "pi"
REMOTE_HOST = "RevPi100364121487"

# Basis-Pfade auf dem lokalen PC und dem RevPi
LOCAL_BASE_DIR = r"C:\DEV\RevPi_ML\ML_Edge_Device\Output\RandomForest\2025-07-22_160652_7540_train"
REMOTE_BASE_DIR = "/home/pi/ML_Edge_Device/Output/RandomForest/2025-07-22_160652_7540_train"

# HIER SIND DIE NEUEN DATEIEN HINZUGEFÜGT
# Format: (Lokaler Unterordner, Dateiname, Ziel-Unterordner)
DATEIEN_ZUM_SENDEN = [
    ("Models", "model.joblib", "Models"),
    ("Models", "features.joblib", "Models"),
    ("Models", "training_config.json", "Models"),
    ("Scalers", "scaler.joblib", "Scalers") 
]
# --------------------

def run_command(command):
    """Führt einen Shell-Befehl aus und prüft auf Fehler."""
    try:
        print(f"▶️  Führe aus: {' '.join(command)}")
        subprocess.run(command, check=True, shell=False)
        print("✅ Befehl erfolgreich ausgeführt.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Fehler bei der Ausführung des Befehls: {e}")
        exit(1)
    except FileNotFoundError:
        print(f"❌ Fehler: Der Befehl '{command[0]}' wurde nicht gefunden.")
        print("Stellen Sie sicher, dass OpenSSH auf Ihrem System installiert und im PATH ist.")
        exit(1)

def sende_datei(local_base, remote_base, datei_info):
    """Erstellt den Zielordner und kopiert eine einzelne Datei."""
    local_subdir, filename, remote_subdir = datei_info
    
    source_file = os.path.join(local_base, local_subdir, filename)
    remote_full_path = f"{remote_base}/{remote_subdir}"
    remote_destination = f"{REMOTE_USER}@{REMOTE_HOST}:{remote_full_path}/"
    
    print(f"\n--- Verarbeite: {filename} ---")

    print(f"Schritt 1: Erstelle Zielordner '{remote_full_path}'")
    mkdir_command = [
        "ssh",
        f"{REMOTE_USER}@{REMOTE_HOST}",
        f"mkdir -p {shlex.quote(remote_full_path)}"
    ]
    run_command(mkdir_command)

    print(f"Schritt 2: Sende '{filename}'")
    scp_command = [
        "scp",
        source_file,
        remote_destination
    ]
    run_command(scp_command)

def main():
    """Hauptfunktion, die den Transfer für alle definierten Dateien startet."""
    for datei_info in DATEIEN_ZUM_SENDEN:
        sende_datei(LOCAL_BASE_DIR, REMOTE_BASE_DIR, datei_info)

    print("\n🎉 Alle Kopiervorgänge erfolgreich abgeschlossen!")

if __name__ == "__main__":
    main()