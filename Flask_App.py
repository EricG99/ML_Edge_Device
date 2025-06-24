# Flask_App.py
import threading
import time
import logging
from flask import Flask, render_template, jsonify, request

# --- Logging-Konfiguration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Flask App Instanz (wird von initialize_flask_app gesetzt) ---
app = None

# --- Globale Daten-Container (für den Zustand der ML-Pipeline) ---
# Diese werden von den ML-Pipeline-Funktionen über den gemeinsamen Zustand aktualisiert
_pipeline_artifacts = {
    "status": "initializing",
    "error_message": None
}
_prediction_data = {
    "step_data": [],
    "inference_step": -1,
    "total_steps": 0,
    "horizon": 1
}

# --- Referenzen zu den ML-Pipeline-Funktionen (werden von initialize_flask_app gesetzt) ---
_setup_and_train_func = None
_run_inference_step_func = None

def initialize_flask_app(setup_and_train_func, run_inference_step_func):
    """
    Initialisiert die Flask-Anwendung und registriert die ML-Pipeline-Funktionen.
    Diese Funktion sollte vom spezifischen ML-Pipeline-Skript aufgerufen werden.

    Args:
        setup_and_train_func (callable): Die Funktion der ML-Pipeline zum Einrichten und Trainieren des Modells.
                                          Sie sollte ein Dictionary von Artefakten zurückgeben.
        run_inference_step_func (callable): Die Funktion der ML-Pipeline zur Ausführung eines Inferenzschritts.
                                            Sie sollte ein Dictionary mit den Ergebnissen des Schritts zurückgeben.
    """
    global app, _setup_and_train_func, _run_inference_step_func

    app = Flask(__name__, template_folder='templates')
    _setup_and_train_func = setup_and_train_func
    _run_inference_step_func = run_inference_step_func

    # Routen registrieren, nachdem die App initialisiert wurde
    _register_routes()

    # Starte das Training in einem Hintergrund-Thread sofort nach der App-Initialisierung,
    # wenn ML-Funktionen bereitgestellt wurden.
    if _setup_and_train_func:
        logging.info("Starte das anfängliche Modelltraining im Hintergrund.")
        training_thread = threading.Thread(target=_run_training_background)
        training_thread.start()
    else:
        _pipeline_artifacts["status"] = "error"
        _pipeline_artifacts["error_message"] = "ML-Pipeline-Funktionen wurden der Flask-App nicht bereitgestellt."
        logging.error("ML-Pipeline-Funktionen wurden nicht bereitgestellt. Die App wird nicht korrekt funktionieren.")

def _run_training_background():
    """Führt das Training unter Verwendung der geladenen ML-Pipeline aus."""
    global _pipeline_artifacts, _prediction_data
    if not _setup_and_train_func:
        logging.error("setup_and_train_func ist nicht gesetzt. Training kann nicht gestartet werden.")
        return

    _pipeline_artifacts["status"] = "training"
    logging.info("Phase 1: Modelltraining wird gestartet...")
    try:
        # Rufe die bereitgestellte setup_and_train Funktion auf
        artifacts = _setup_and_train_func()
        _pipeline_artifacts.update(artifacts)
        _pipeline_artifacts["status"] = "ready_for_inference"
        _prediction_data["total_steps"] = artifacts.get("total_steps", 0)
        _prediction_data["horizon"] = artifacts.get("config", {}).get("horizon", 1)
        logging.info("Phase 1: Modelltraining erfolgreich abgeschlossen.")
    except Exception as e:
        logging.error(f"Fehler während des Modelltrainings: {e}", exc_info=True)
        _pipeline_artifacts["status"] = "error"
        _pipeline_artifacts["error_message"] = str(e)

def _run_inference_background():
    """Führt die Inferenz Schritt für Schritt mit der geladenen ML-Pipeline aus."""
    global _prediction_data, _pipeline_artifacts
    if not _run_inference_step_func:
        logging.error("run_inference_step_func ist nicht gesetzt. Inferenz kann nicht gestartet werden.")
        return

    try:
        _prediction_data["step_data"] = []
        for i in range(_prediction_data["total_steps"]):
            # Aktualisiere den Status SOFORT
            _prediction_data["inference_step"] = i

            # Führe Inferenz durch
            # Rufe die bereitgestellte run_inference_step Funktion auf
            step_result = _run_inference_step_func(_pipeline_artifacts, i)
            _prediction_data["step_data"].append(step_result)

            # Kurze Pause für UI-Updates
            time.sleep(0.05)

            # Debug-Logging
            if i % 10 == 0:
                logging.info(f"Inferenzschritt {i}/{_prediction_data['total_steps']} abgeschlossen")

        # Finaler Statusupdate
        _pipeline_artifacts["status"] = "finished"
        _prediction_data["inference_step"] = _prediction_data["total_steps"]
        logging.info("Phase 2: Live-Inferenz erfolgreich abgeschlossen.")

    except Exception as e:
        logging.error(f"Fehler während der Live-Inferenz: {e}", exc_info=True)
        _pipeline_artifacts["status"] = "error"
        _pipeline_artifacts["error_message"] = str(e)

def _register_routes():
    """Registriert alle Flask-Routen mit der App-Instanz."""
    if app is None:
        logging.error("Flask-App-Instanz ist nicht initialisiert. Routen können nicht registriert werden.")
        return

    @app.route('/')
    def index():
        return render_template('dashboard.html')

    @app.route('/api/status')
    def get_status():
        return jsonify({
            "status": _pipeline_artifacts["status"],
            "error_message": _pipeline_artifacts["error_message"],
            "total_steps": _prediction_data["total_steps"],
            "horizon": _prediction_data.get("horizon", 1)
        })

    @app.route('/api/run_inference', methods=['POST'])
    def run_inference_endpoint():
        if _pipeline_artifacts.get("status") != "ready_for_inference":
            return jsonify({"error": "Modell nicht bereit für Inferenz."}), 400
        _pipeline_artifacts["status"] = "inference_running"
        _prediction_data["inference_step"] = -1 # Set to -1 to indicate inference has just started
        threading.Thread(target=_run_inference_background).start()
        return jsonify({"status": "success", "message": "Live-Inferenz gestartet."})

    @app.route('/api/data')
    def get_data():
        step = request.args.get('step', type=int)
        if step is None:
            return jsonify({"error": "Abfrageparameter 'step' ist erforderlich."}), 400
        if step > _prediction_data["inference_step"]:
            # Data for this step is not yet available
            return jsonify({"status": "waiting"}), 202
        if step >= len(_prediction_data["step_data"]):
            # Data for this step is not found, implies an out-of-bounds request
            return jsonify({"error": "Schritt nicht gefunden oder noch nicht berechnet."}), 404
        return jsonify(_prediction_data["step_data"][step])

def run_flask_server(host='0.0.0.0', port=5001, debug=False, use_reloader=False):
    """Startet den Flask-Entwicklungsserver."""
    if app:
        logging.info(f"Flask-Server startet auf http://{host}:{port}")
        app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)
    else:
        logging.error("Flask-App nicht initialisiert. Rufen Sie zuerst initialize_flask_app auf.")


