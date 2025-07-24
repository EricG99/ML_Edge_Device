# config/config_general.py
from pathlib import Path
from datetime import datetime

# --- Pfade ---
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"
ARTIFACTS_OUTPUT_PATH = OUTPUT_DIR 

# Diese Struktur bleibt für den Zugriff auf Unterordner nützlich
CONFIG_PATH = {
    "paths": {
        "base": BASE_DIR,
        "input": INPUT_DIR,
        "output": OUTPUT_DIR,
        "input_data": INPUT_DIR / "Input_Data",
        "artifacts_output": OUTPUT_DIR,
        "output_error_metrics": OUTPUT_DIR / "Error_Metrics"
    }
}

# --- MQTT-Konfiguration ---
MQTT_CONFIG = {
    'MQTT_BROKER_IP': "192.168.0.101",
    'MQTT_PORT': 1883,
    'MQTT_TOPIC': "sim/data/20240341/S6"
}

# --- Konfiguration zum Laden von Artefakten ---
# Die Logik, welcher Run geladen wird, sollte zur Laufzeit entschieden werden,
# aber die statischen Namen für den "Fast-Mode" können hier bleiben.
CONFIG_LOAD_ARTIFACTS = {
    "inference_mode": "load_artifacts_path", # "load_artifacts_fast" oder "load_artifacts_path"
    "inference_steps" : 500,
    "inference_interval_sec": 1,
    "artifacts_base_path": ARTIFACTS_OUTPUT_PATH,

    "loading_strategy": "live_mqtt", #"split", "separate_csv", "live_mqtt"
    
    # Statische Namen für den 'fast' mode
    "model_path_static": "trained_rf_model.joblib",
    "scaler_path_static": "trained_rf_scaler.joblib",
    "features_path_static": "trained_rf_features.joblib",
}

# --- Hilfsfunktionen ---
def generate_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}"

def initialize_directories():
    for path in CONFIG_PATH["paths"].values():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

initialize_directories()