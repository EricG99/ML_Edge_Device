# web_app.py
import logging
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta

log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

def create_app(app_config, pipeline_state, predictions_list, lock, *, template_name='dashboard_retrain.html', start_inference_callback=None):
    """
    Create and configure the Flask web application that serves the pipeline UI & APIs.

    Args:
        app_config (dict): Effective configuration dict used by the pipeline.
        pipeline_state (dict): Shared mutable state with keys like status, is_paused, etc.
        predictions_list (list[dict]): Shared list of per-step prediction entries.
        lock (threading.Lock): A threading lock protecting shared state.
        template_name (str): Template to render at '/' (default: dashboard_retrain.html).
        start_inference_callback (callable|None): Optional function to start inference in background.

    Returns:
        Flask: Configured Flask app.
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
        step_index = request.args.get('step', type=int, default=0)

        with app.config['LOCK']:
            preds = app.config['PREDICTIONS']
            state = app.config['PIPELINE_STATE']

            if step_index < len(preds):
                entry = dict(preds[step_index])  # shallow copy
                # Normalize datetime to ISO string
                dt = entry.get('datetime')
                if isinstance(dt, (datetime,  )):
                    entry['datetime'] = dt.isoformat()
                elif hasattr(dt, 'to_pydatetime'):
                    entry['datetime'] = dt.to_pydatetime().isoformat()

                # Add rolling forecast dates aligned to inference interval
                interval_key = "inference_cycle_sec" if state.get("mode") == "retraining" else "inference_interval_sec"
                interval_sec = app.config['APP_CONFIG'].get(interval_key, 1.0)
                rf_values = entry.get("rolling_forecast", [])
                try:
                    base_ts = datetime.fromisoformat(entry['datetime'])
                    entry['rolling_forecast_dates'] = [
                        (base_ts + timedelta(seconds=i * interval_sec)).isoformat()
                        for i in range(len(rf_values))
                    ]
                except Exception:
                    entry['rolling_forecast_dates'] = []

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

    # Optional endpoint to trigger background inference (for manual mode)
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
