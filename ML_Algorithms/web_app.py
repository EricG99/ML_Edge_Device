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

# In web_app.py, innerhalb create_app(...):

    @app.route('/api/data')
    def get_data():
        """
        Liefert den nächsten verfügbaren Schritt für das Dashboard.
        Bricht Forecast in (1-Schritt + restliche Zukunft) auf und liefert CPU/RAM mit.
        """
        # Optional: gezielt einen Schritt holen (?step=N)
        try:
            step_idx = int(request.args.get('step')) if 'step' in request.args else None
        except Exception:
            step_idx = None

        with app.config['LOCK']:
            preds = app.config['PREDICTIONS']
            state = app.config['PIPELINE_STATE']
            cfg = app.config['APP_CONFIG']

            # Schritt selektieren
            if step_idx is not None:
                entry = preds[step_idx] if 0 <= step_idx < len(preds) else None
            else:
                entry = preds[-1] if preds else None

        if entry is None:
            return jsonify({"status": "waiting"})

        # --- Datum normalisieren ---
        from datetime import datetime, timedelta
        try:
            from pandas import Timestamp
        except Exception:
            Timestamp = datetime

        dt = entry.get('datetime')
        if dt is not None and not isinstance(dt, (datetime, Timestamp)):
            # ISO-String -> Timestamp
            try:
                dt = Timestamp(dt)
            except Exception:
                pass
        # Für JSON: ISO-String ausgeben
        if isinstance(dt, (datetime, Timestamp)):
            entry['datetime'] = (dt.to_pydatetime() if hasattr(dt, 'to_pydatetime') else dt).isoformat()

        # --- Forecast aufsplitten: (h1) + (h2..hH) ---
        full_forecast = entry.get("future_forecast", []) or entry.get("rolling_forecast", []) or []
        horizon = int(cfg.get("horizon", 1))
        interval_sec = float(cfg.get("inference_interval_sec", 1.0))

        if isinstance(dt, (datetime, Timestamp)):
            base_ts = dt if isinstance(dt, Timestamp) else Timestamp(dt)
        else:
            base_ts = None

        if full_forecast and base_ts is not None:
            pred_1 = full_forecast[0]
            pred_1_date = (base_ts + timedelta(seconds=interval_sec)).isoformat()

            rest = full_forecast[1:horizon]
            rest_dates = [
                (base_ts + timedelta(seconds=(i + 2) * interval_sec)).isoformat()
                for i in range(len(rest))
            ]
        else:
            pred_1, pred_1_date, rest, rest_dates = None, None, [], []

        # --- CPU/RAM: Schritt-Werte bevorzugen, sonst live messen ---
        # CPU:
        cpu_from_step = entry.get("cpu_load")
        if cpu_from_step is None:
            cpu_from_step = entry.get("cpu_percent")
        if cpu_from_step is None:
            from ML_Helpfunctions import Pipeline_Utils
            cpu_from_step = Pipeline_Utils.get_cpu_usage()
        try:
            cpu_from_step = float(cpu_from_step) if cpu_from_step is not None else None
        except Exception:
            cpu_from_step = None

        # RAM:
        ram_from_step = entry.get("ram_usage")
        if ram_from_step is None and entry.get("ram_mb") is not None:
            # In MB: in GB/Percent umrechnen so gut es geht (Percent unbekannt)
            try:
                used_gb = float(entry["ram_mb"]) / 1024.0
                ram_from_step = {"total_gb": "N/A", "used_gb": round(used_gb, 2), "percent": None}
            except Exception:
                ram_from_step = None

        if ram_from_step is None:
            from ML_Helpfunctions import Pipeline_Utils
            ram_from_step = Pipeline_Utils.get_memory_usage()

        # Antwort-Objekt zusammenbauen (nur Felder, die das Frontend nutzt)
        out = dict(entry)  # Kopie
        out["prediction_1_step"] = pred_1
        out["prediction_1_step_date"] = pred_1_date
        out["future_forecast"] = rest
        out["future_forecast_dates"] = rest_dates
        out["cpu_load"] = cpu_from_step
        out["ram"] = ram_from_step

        return jsonify({"status": "success", "data": out})


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