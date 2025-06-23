import sys
import importlib
import threading
import time
import logging
from flask import Flask, render_template, jsonify, request

# --- Logging-Konfiguration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App Initialisierung ---
app = Flask(__name__, template_folder='templates')

# --- Globale Daten-Container ---
pipeline_artifacts = {
    "status": "initializing",
    "error_message": None
}
prediction_data = {
    "step_data": [],
    "inference_step": -1,
    "total_steps": 0,
    "horizon": 1
}

# --- Dynamischer Import des ML-Pipeline-Skripts ---
ml_pipeline = None
try:
    if len(sys.argv) < 2:
        raise ImportError("Es wurde kein Name für ein ML-Pipeline-Skript übergeben.")
    
    pipeline_name = sys.argv[1]
    # Annahme: Die Skripte befinden sich in einem Unterordner 'ml_pipelines'
    ml_pipeline = importlib.import_module(f"ML_Flask_Pipeline.{pipeline_name}")
    logging.info(f"ML-Pipeline '{pipeline_name}' erfolgreich geladen.")

except ImportError as e:
    logging.error(f"Fehler beim Laden der ML-Pipeline: {e}")
    pipeline_artifacts["status"] = "error"
    pipeline_artifacts["error_message"] = str(e)


# --- Hintergrundprozesse (rufen die importierte ML-Logik auf) ---

def run_training_background():
    """Führt das Training unter Verwendung der geladenen ML-Pipeline aus."""
    global pipeline_artifacts, prediction_data
    if not ml_pipeline: return
    
    pipeline_artifacts["status"] = "training"
    logging.info(f"Phase 1: Modelltraining mit '{sys.argv[1]}' wird gestartet...")
    try:
        artifacts = ml_pipeline.setup_and_train()
        pipeline_artifacts.update(artifacts)
        pipeline_artifacts["status"] = "ready_for_inference"
        prediction_data["total_steps"] = artifacts.get("total_steps", 0)
        prediction_data["horizon"] = artifacts.get("config", {}).get("horizon", 1)
        logging.info("Phase 1: Modelltraining erfolgreich abgeschlossen.")
    except Exception as e:
        logging.error(f"Fehler während des Modelltrainings: {e}", exc_info=True)
        pipeline_artifacts["status"] = "error"
        pipeline_artifacts["error_message"] = str(e)

def run_inference_background():
    """Führt die Inferenz Schritt für Schritt mit der geladenen ML-Pipeline aus."""
    global prediction_data, pipeline_artifacts
    if not ml_pipeline: return

    try:
        prediction_data["step_data"] = []
        for i in range(prediction_data["total_steps"]):
            # Aktualisiere den Status SOFORT
            prediction_data["inference_step"] = i
            
            # Führe Inferenz durch
            step_result = ml_pipeline.run_inference_step(pipeline_artifacts, i)
            prediction_data["step_data"].append(step_result)
            
            # Kurze Pause für UI-Updates
            time.sleep(0.05)
            
            # Debug-Logging
            if i % 10 == 0:
                logging.info(f"Inferenzschritt {i}/{prediction_data['total_steps']} abgeschlossen")
        
        # Finaler Statusupdate
        pipeline_artifacts["status"] = "finished"
        prediction_data["inference_step"] = prediction_data["total_steps"]
        logging.info("Phase 2: Live-Inferenz erfolgreich abgeschlossen.")
        
    except Exception as e:
        logging.error(f"Fehler während der Live-Inferenz: {e}", exc_info=True)
        pipeline_artifacts["status"] = "error"
        pipeline_artifacts["error_message"] = str(e)


# --- Flask Routen (API für das Frontend) ---

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/api/status')
def get_status():
    return jsonify({
        "status": pipeline_artifacts["status"],
        "error_message": pipeline_artifacts["error_message"],
        "total_steps": prediction_data["total_steps"],
        "horizon": prediction_data.get("horizon", 1)
    })

@app.route('/api/run_inference', methods=['POST'])
def run_inference_endpoint():
    if pipeline_artifacts.get("status") != "ready_for_inference":
        return jsonify({"error": "Modell nicht bereit für Inferenz."}), 400
    pipeline_artifacts["status"] = "inference_running"
    prediction_data["inference_step"] = -1
    threading.Thread(target=run_inference_background).start()
    return jsonify({"status": "success", "message": "Live-Inferenz gestartet."})

@app.route('/api/data')
def get_data():
    step = request.args.get('step', type=int)
    if step > prediction_data["inference_step"]:
        return jsonify({"status": "waiting"}), 202
    if step >= len(prediction_data["step_data"]):
        return jsonify({"error": "Schritt nicht gefunden."}), 404
    return jsonify(prediction_data["step_data"][step])

# --- Startpunkt für dieses Skript ---
if __name__ == '__main__':
    if ml_pipeline:
        training_thread = threading.Thread(target=run_training_background)
        training_thread.start()
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    else:
        print("Flask-Anwendung konnte wegen eines Importfehlers nicht gestartet werden.")
