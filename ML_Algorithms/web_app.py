# web_app.py (angepasst)
import logging
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
# NEU: Sicherstellen, dass Pandas Timestamps korrekt behandelt werden
try:
    from pandas import Timestamp
except ImportError:
    Timestamp = datetime

log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

def create_app(app_config, pipeline_state, predictions_list, lock, *, template_name='dashboard_retrain.html', start_inference_callback=None):
    """
    Create and configure the Flask web application that serves the pipeline UI & APIs.
    """
    app = Flask(__name__, template_folder='templates')
    app.config['APP_CONFIG'] = app_config
    app.config['LOCK'] = lock
    app.config['PIPELINE_STATE'] = pipeline_state
    app.config['PREDICTIONS'] = predictions_list
    app.config['START_INFERENCE_CALLBACK'] = start_inference_callback

    @app.route('/')
    def index():
        return render_template(template_name, config=app_config)

    @app.route('/api/status')
    def get_status():
        with app.config['LOCK']:
            return jsonify(app.config['PIPELINE_STATE'].copy())

    @app.route('/api/data')
    def get_data():
        """
        Liefert den Eintrag für den angefragten Schritt (inkl. abgeleiteter Felder
        für 1-Schritt-Prognose und Zukunftsprognose). Verwendet IMMER
        'inference_interval_sec' für die Zeitstempel-Berechnung.
        """
        step_index = request.args.get('step', type=int, default=0)

        with app.config['LOCK']:
            preds = app.config['PREDICTIONS']
            state = app.config['PIPELINE_STATE']

            if step_index < len(preds):
                entry = dict(preds[step_index])  # shallow copy

                # Datetime normalisieren
                dt = entry.get('datetime')
                if isinstance(dt, (datetime, Timestamp)):
                    entry['datetime'] = dt.isoformat()
                elif hasattr(dt, 'to_pydatetime'):
                    entry['datetime'] = dt.to_pydatetime().isoformat()

                # Zukunfts-Prognosefeld auftrennen
                full_forecast = entry.get("future_forecast", [])
                if full_forecast:
                    # KORREKTUR: Immer inference_interval_sec verwenden
                    interval_sec = app.config['APP_CONFIG'].get("inference_interval_sec", 1.0)
                    base_ts = Timestamp(entry['datetime']) if not isinstance(dt, (datetime, Timestamp)) else dt

                    # 1-Schritt-Prognose
                    entry['prediction_1_step'] = full_forecast[0]
                    entry['prediction_1_step_date'] = (base_ts + timedelta(seconds=interval_sec)).isoformat()

                    # Restliche Zukunft
                    remaining_forecast = full_forecast[1:]
                    entry['future_forecast'] = remaining_forecast
                    entry['future_forecast_dates'] = [
                        (base_ts + timedelta(seconds=(i + 2) * interval_sec)).isoformat()
                        for i in range(len(remaining_forecast))
                    ]
                else:
                    entry['prediction_1_step'] = None
                    entry['prediction_1_step_date'] = None
                    entry['future_forecast'] = []
                    entry['future_forecast_dates'] = []

                return jsonify({"status": "success", "data": entry})
            else:
                return jsonify({"status": "waiting"})


    @app.route('/api/control', methods=['POST'])
    def control_pipeline():
        action = request.json.get('action')
        with app.config['LOCK']:
            if action == 'pause':
                app.config['PIPELINE_STATE']['is_paused'] = True
            elif action == 'resume':
                app.config['PIPELINE_STATE']['is_paused'] = False
            out = {"status": "ok", "is_paused": app.config['PIPELINE_STATE']['is_paused']}
        logging.info("Control action received: %s -> paused=%s", action, out["is_paused"])
        return jsonify(out)

    if start_inference_callback is not None:
        @app.route('/api/run_inference', methods=['POST'])
        def run_inference_endpoint():
            with app.config['LOCK']:
                status = app.config['PIPELINE_STATE'].get("status")
            if status not in ["ready_for_inference", "finished"]:
                return jsonify({"error": "Not ready for inference."}), 400
            try:
                start_inference_callback()
            except Exception as e:
                logging.exception("Failed to start inference")
                return jsonify({"error": str(e)}), 500
            return jsonify({"status": "Inference process initiated."})
    return app