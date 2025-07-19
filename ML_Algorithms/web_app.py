# web_app.py
import logging
from flask import Flask, render_template, jsonify, request
import threading

# Quieter logging for the web server
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

def create_app(pipeline_state, prediction_data, start_inference_callback):
    """
    Creates and configures the Flask application.

    Args:
        pipeline_state (dict): Shared dictionary for the pipeline's status.
        prediction_data (dict): Shared dictionary for storing prediction results.
        start_inference_callback (function): A function to call to start the inference thread.

    Returns:
        Flask: The configured Flask app instance.
    """
    app = Flask(__name__, template_folder='templates')
    
    # Store shared objects and callback in app config for access in routes
    app.config['PIPELINE_STATE'] = pipeline_state
    app.config['PREDICTION_DATA'] = prediction_data
    app.config['START_INFERENCE_CALLBACK'] = start_inference_callback
    
    @app.route('/')
    def index():
        return render_template('dashboard.html')

    @app.route('/api/status')
    def get_status():
        return jsonify(app.config['PIPELINE_STATE'])

    @app.route('/api/data')
    def get_data():
        step = request.args.get('step', type=int, default=0)
        pred_data = app.config['PREDICTION_DATA']
        
        # Prüfen, ob neue Daten für den angeforderten Schritt verfügbar sind
        if step > pred_data["inference_step"]:
            # Immer 200 OK zurückgeben, aber den Status in der JSON-Antwort mitteilen
            return jsonify({"status": "waiting"}), 200
        
        # Daten sind vorhanden, sende sie mit dem Status "success"
        return jsonify({
            "status": "success",
            "data": pred_data["step_data"][step]
        }), 200

    @app.route('/api/run_inference', methods=['POST'])
    def run_inference_endpoint():
        pipeline_state = app.config['PIPELINE_STATE']
        if pipeline_state.get("status") not in ["ready_for_inference", "finished"]:
            return jsonify({"error": "Not ready for inference."}), 400
        
        # Call the provided callback function to start the background inference thread
        start_callback = app.config['START_INFERENCE_CALLBACK']
        thread = threading.Thread(target=start_callback, daemon=True)
        thread.start()
        
        return jsonify({"status": "Inference process initiated."})

    return app