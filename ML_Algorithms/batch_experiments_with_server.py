# batch_experiments_with_server.py
import subprocess, sys, time, json, os, shlex
from pathlib import Path
import urllib.request

PY = sys.executable
ROOT = Path(__file__).resolve().parent
PORT = 5002

EXPERIMENTS = [
    {"algorithm": "lstm", "config": "lstm", "name": "lstm_small"},
    {"algorithm": "lstm", "config": "lstm", "name": "lstm_medium"},
    {"algorithm": "rf",   "config": "rf",   "name": "rf_default"},
]

def run(cmd, cwd=None):
    print(f"\n$ {cmd}")
    p = subprocess.run(shlex.split(cmd), cwd=cwd, capture_output=True, text=True)
    print(p.stdout)
    if p.returncode != 0:
        print(p.stderr, file=sys.stderr)
        raise SystemExit(p.returncode)
    return p.stdout

def wait_healthz(url, timeout_s=30):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("Flask-Server nicht erreichbar.")

def main():
    # 1) Flask-Server EINMAL starten (persistenter Prozess)
    server = subprocess.Popen(
        [PY, "pipeline_web_app.py", "--web-only", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=ROOT
    )
    try:
        wait_healthz(f"http://127.0.0.1:{PORT}/healthz", timeout_s=30)
        print("✅ Flask-Server ist bereit.")

        # 2) Für jedes Experiment: Training (mit Web aus), dann Inferenz (mit Web aus)
        for i, exp in enumerate(EXPERIMENTS, 1):
            algo = exp["algorithm"]; cfg = exp["config"]
            print(f"\n=== Run {i}/{len(EXPERIMENTS)}: {exp['name']} ({algo}/{cfg}) ===")

            # Training (erzeugt Run-ID und Artefakte); Web bleibt als separater Prozess an
            out = run(f'{PY} pipeline_web_app.py --retrain --algorithm {algo} --config-name {cfg} --no-web', cwd=ROOT)

            # Run-ID aus der Trainer-Logausgabe extrahieren
            run_id = None
            for line in out.splitlines():
                if "Run ID:" in line:
                    run_id = line.split("Run ID:")[-1].strip()
            if not run_id:
                raise RuntimeError("Run ID nicht gefunden.")

            # Inferenz (headless), definierte Schrittzahl, lädt Artefakte per --load_id
            run(f'{PY} pipeline_web_app.py --algorithm {algo} --load_id {run_id} --no-web --inference-steps 200', cwd=ROOT)

            # (Optional) hier kannst du direkt Metriken/CSV einlesen/auswerten

    finally:
        # 3) Server am Ende sauber stoppen
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

if __name__ == "__main__":
    main()
