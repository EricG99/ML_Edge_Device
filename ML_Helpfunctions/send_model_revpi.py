import subprocess
import shlex

# --- Konfiguration ---
SOURCE_FILE = r"C:\DEV\RevPi_ML\ML_Edge_Device\Output\RandomForest\2025-07-22_160652_7540_train\Models\model.joblib"
REMOTE_USER = "pi"
REMOTE_HOST = "RevPi100364121487"

# HIER IST DIE ÄNDERUNG: Verwende den absoluten Pfad statt '~'
REMOTE_BASE_DIR = "/home/pi/ML_Edge_Device/Output/RandomForest" 

NEW_FOLDER = "2025-07-22_160652_7540_train/Models"
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

def main():
    """Hauptfunktion zum Erstellen des Ordners und Kopieren der Datei."""
    
    remote_full_path = f"{REMOTE_BASE_DIR}/{NEW_FOLDER}"
    remote_destination = f"{REMOTE_USER}@{REMOTE_HOST}:{remote_full_path}/"

    print("\n--- Schritt 1: Erstelle Zielordner auf dem RevPi ---")
    mkdir_command = [
        "ssh",
        f"{REMOTE_USER}@{REMOTE_HOST}",
        f"mkdir -p {shlex.quote(remote_full_path)}"
    ]
    run_command(mkdir_command)

    print("\n--- Schritt 2: Sende die Datei via SCP ---")
    scp_command = [
        "scp",
        SOURCE_FILE,
        remote_destination
    ]
    run_command(scp_command)

    print("\n🎉 Kopiervorgang erfolgreich abgeschlossen!")


if __name__ == "__main__":
    main()