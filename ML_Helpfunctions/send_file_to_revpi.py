import argparse
import os
from paramiko import SSHClient
from scp import SCPClient
import getpass

def send_file(local_path, remote_path, hostname, username, password=None):
    """
    Stellt eine SSH-Verbindung zu einem Host her und sendet eine Datei per SCP.

    Args:
        local_path (str): Der Pfad zur lokalen Datei, die gesendet werden soll.
        remote_path (str): Der Zielpfad auf dem Remote-Gerät.
        hostname (str): Der Hostname oder die IP-Adresse des RevPi.
        username (str): Der Benutzername für die SSH-Verbindung.
        password (str, optional): Das Passwort für die SSH-Verbindung. 
                                  Wenn nicht angegeben, wird es abgefragt.
    """
    if not os.path.exists(local_path):
        print(f"Fehler: Die Quelldatei '{local_path}' wurde nicht gefunden.")
        return

    try:
        # Erstellt ein SSH-Client-Objekt
        ssh = SSHClient()
        # Lädt bekannte Hosts (optional, aber empfohlen)
        ssh.load_system_host_keys()
        # Fügt neue Hosts automatisch hinzu (vereinfacht, aber weniger sicher)
        # Für höhere Sicherheit verwenden Sie stattdessen ssh.load_system_host_keys()
        # und stellen Sie sicher, dass der Host-Schlüssel bekannt ist.
        ssh.set_missing_host_key_policy(ssh.WarningPolicy())

        print(f"Verbinde mit {hostname}...")
        
        # Stellt die Verbindung her
        if password:
            ssh.connect(hostname, username=username, password=password)
        else:
            # Fragt nach dem Passwort, wenn keines übergeben wurde
            password = getpass.getpass(f"Passwort für {username}@{hostname}: ")
            ssh.connect(hostname, username=username, password=password)

        print("Verbindung hergestellt.")

        # Verwendet SCPClient, um die Datei zu übertragen
        with SCPClient(ssh.get_transport()) as scp:
            print(f"Übertrage '{local_path}' nach '{hostname}:{remote_path}'...")
            scp.put(local_path, remote_path)
            print("Dateiübertragung erfolgreich abgeschlossen.")

    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
    finally:
        # Schließt die SSH-Verbindung
        if 'ssh' in locals() and ssh.get_transport() and ssh.get_transport().is_active():
            ssh.close()
            print("Verbindung geschlossen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sendet eine Datei per SSH an einen RevPi.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # parser.add_argument("input", help="Der Pfad zur lokalen Datei, die gesendet werden soll.")
    # parser.add_argument("output", help="Der Zielpfad auf dem RevPi.")
    # parser.add_argument("--host", required=True, help="Der Hostname oder die IP-Adresse des RevPi.")
    # parser.add_argument("--user", required=True, help="Der Benutzername für die SSH-Verbindung.")
    # parser.add_argument("--password", help="Das Passwort für die SSH-Verbindung. \nWenn nicht angegeben, wird es interaktiv abgefragt.", nargs='?')

    # args = parser.parse_args()

    send_file(args.input, args.output, args.host, args.user, args.password)