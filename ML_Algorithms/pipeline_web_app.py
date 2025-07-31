import time
import logging
import argparse
import sys
import threading
import socket
import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request
from copy import deepcopy
from datetime import datetime, timedelta
import tensorflow as tf

# --- Systempfad-Setup ---
# Stellt sicher, dass das Skript von überall aus korrekt auf die Projektdateien zugreifen kann
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Allgemeine Anwendungsimporte ---
from config.config_general import CONFIG_PATH, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient
from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
from ML_Helpfunctions import Feature_Engeneering as fe

# --- Dynamische Importe für Algorithmen ---
# Diese werden je nach ausgewähltem Modus geladen
from ML_Algorithms.LSTM.LSTM_train import LSTMTrainer
from ML_Algorithms.LSTM.LSTM_inference import LSTMInference
from ML_Algorithms.Random_Forest.rf_inference import RFInference

# --- Globale Konfiguration für das Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

# --- Globale Zustandsvariablen und Sperrobjekte ---
PIPELINE_STATE = {
    "status": "initializing",
    "error_message": None,
    "retraining_status": "idle",
    "cycle_count": 0,
    "steps_in_cycle": 0,
    "total_steps": 0,
    "total_cycles": 0,
    "is_paused": False,
    "is_finished": False,
}
shared_resource_lock = threading.Lock()

# --- Globale, dynamisch geladene Objekte ---
inference_processor = None
mqtt_client = None
# Speziell für den Retraining-Modus
shared_model = {"model": None, "scaler": None, "features": None, "config": None}
retraining_data_list = []
all_predictions = []

# --- Gemeinsame Hilfsfunktionen ---

def get_local_ip():
    """Ermittelt die lokale IP-Adresse des Geräts, um den Zugriffslink anzuzeigen."""
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = "127.0.0.1"
    finally:
        if s:
            s.close()
    return ip_address

# --- Standard-Inferenzmodus-Funktionen (aus pipeline_web_app.py) ---

def prepare_standard_inference(config: dict, algorithm: str):
    """Lädt Artefakte und initialisiert den Inferenz-Prozessor für den Standardmodus."""
    global inference_processor
    with shared_resource_lock:
        PIPELINE_STATE["status"] = "loading_artifacts"
        logging.info(f"Phase 1: Lade Artefakte für Algorithmus '{algorithm}'...")
        try:
            if algorithm == 'random_forest':
                processor_class = RFInference
                FOLDER_FLAG = "RandomForest"
            elif algorithm == 'lstm':
                processor_class = LSTMInference
                FOLDER_FLAG = "LSTM"
            else:
                raise ValueError(f"Unbekannter Algorithmus: {algorithm}")

            proc = processor_class(
                config=config,
                broker_ip=config['MQTT_BROKER_IP'], port=config['MQTT_PORT'], topic=config['MQTT_TOPIC'],
                folder_flag=FOLDER_FLAG
            )
            proc.load_artifacts()
            inference_processor = proc
            PIPELINE_STATE.update({
                "status": "ready_for_inference",
                "horizon": config.get("horizon", 1),
                "inference_interval_sec": config.get("inference_interval_sec", 1.0),
                "total_steps": inference_processor.inference_steps
            })
            logging.info("✅ Phase 1: Modell & Artefakte geladen. Bereit für Inferenz.")
        except Exception as e:
            logging.error(f"Fehler während der Vorbereitung: {e}", exc_info=True)
            PIPELINE_STATE["status"] = "error"
            PIPELINE_STATE["error_message"] = str(e)

def run_standard_inference_background_task(config):
    """Orchestriert den Inferenzprozess im Standardmodus."""
    global inference_processor, mqtt_client
    with shared_resource_lock:
        if PIPELINE_STATE["status"] == "inference_running":
            logging.warning("Inferenz läuft bereits.")
            return
        PIPELINE_STATE["status"] = "inference_running"
        logging.info("Starte Inferenz-Hintergrundaufgabe...")
        mqtt_client = MqttInferenceClient(
            broker_ip=config['MQTT_BROKER_IP'], port=config['MQTT_PORT'], topic=config['MQTT_TOPIC'],
            on_message_callback=inference_processor.update_latest_data
        )

    inference_thread = threading.Thread(target=inference_processor._live_inference_loop, daemon=True)
    mqtt_client.run()
    inference_thread.start()
    logging.info("🚀 Inferenzprozess und MQTT-Client gestartet.")
    inference_thread.join()

    logging.info("Inferenzschleife beendet. Bereinige Ressourcen...")
    if mqtt_client:
        mqtt_client.client.loop_stop()
        mqtt_client.client.disconnect()
    inference_processor._save_results()
    PIPELINE_STATE["status"] = "finished"
    logging.info("✅ Inferenz-Hintergrundaufgabe abgeschlossen.")

def start_standard_inference_flow(config):
    """Wrapper zum Starten der Standard-Inferenz in einem neuen Thread."""
    if inference_processor:
        inference_processor.step_counter = 0
        inference_processor.results_buffer.clear()
    thread = threading.Thread(target=run_standard_inference_background_task, args=(config,), daemon=True)
    thread.start()

# --- Retraining-Modus-Funktionen (aus pipeline_web_app_retraining.py) ---

def initial_training(config: dict):
    """Führt das initiale Training für den Retraining-Modus durch."""
    global shared_model
    logging.info("--- PHASE 1: Initiales Training startet (Retraining-Modus) ---")
    trainer = LSTMTrainer(config=config, folder_flag="LSTM_retrain")
    model, scaler, features = trainer.run(save_artifacts=False)
    with shared_resource_lock:
        shared_model.update({"model": model, "scaler": scaler, "features": features, "config": config})
        PIPELINE_STATE["status"] = "inference_running"
    logging.info("--- PHASE 1: Initiales Training abgeschlossen. Starte Inferenz-Manager. ---")

def retraining_thread_task(current_model, scaler, features, retraining_data, config):
    """Führt das Nachtraining in einem separaten Thread aus."""
    global shared_model
    logging.info("--- RETRAINING THREAD: Startet Nachtraining. ---")
    PIPELINE_STATE["retraining_status"] = "training"
    try:
        retraining_data_featured, _ = fe.add_all_features(retraining_data, config)
        retraining_data_featured = retraining_data_featured.dropna()
        scaled_data = scaler.transform(retraining_data_featured[features])
        X_retrain, y_retrain = LoadPrepareData.convert_data_to_sliding_window(
            scaled_data, lag_horizon=config["lags"], forecast_horizon=config["horizon"]
        )
        if len(X_retrain) == 0:
            logging.warning("RETRAINING THREAD: Nicht genügend Daten für Nachtraining.")
            return

        model_to_retrain = tf.keras.models.clone_model(current_model)
        model_to_retrain.set_weights(current_model.get_weights())
        model_to_retrain.compile(optimizer=config.get("optimizer", "adam"), loss=config.get("loss", "mse"))
        model_to_retrain.fit(X_retrain, y_retrain, epochs=config.get("retraining_epochs", 5), batch_size=config.get("batch_size", 32), verbose=0)
        
        with shared_resource_lock:
            shared_model["model"] = model_to_retrain
            logging.info("--- RETRAINING THREAD: Modellaustausch erfolgreich! ---")
    except Exception as e:
        logging.error(f"RETRAINING THREAD: Fehler: {e}", exc_info=True)
    finally:
        PIPELINE_STATE["retraining_status"] = "idle"

def run_rolling_forecast(model, scaler, start_window, config, feature_list):
    """Erzeugt eine rollierende Vorhersage für den definierten Horizont."""
    horizon = config.get("horizon", 1)
    if horizon <= 1: return [], []
    future_preds_unscaled = []
    current_window = deepcopy(start_window)
    for _ in range(horizon - 1):
        prediction_scaled = model.predict(current_window, verbose=0)
        target_index = feature_list.index(config['base_features'][0])
        prediction_unscaled = PipelineUtils.safe_inverse_transform(
            scaler, prediction_scaled.reshape(1, -1), target_index
        ).flatten()[0]
        future_preds_unscaled.append(prediction_unscaled)
        next_step_features_scaled = current_window[0, -1, :].copy()
        next_step_features_scaled[target_index] = prediction_scaled[0, 0]
        new_window_step = np.expand_dims(next_step_features_scaled, axis=0)
        current_window = np.append(current_window[:, 1:, :], np.expand_dims(new_window_step, axis=0), axis=1)
    return future_preds_unscaled

def inference_and_retraining_manager(config: dict):
    """Haupt-Controller für Inferenz, Datensammlung und Retraining."""
    global inference_processor, all_predictions, retraining_data_list, mqtt_client
    while PIPELINE_STATE["status"] == "initializing":
        time.sleep(1)
    logging.info("--- PHASE 2: Inferenz-Manager startet (Retraining-Modus). ---")
    
    max_cycles = config.get("retraining_cycles", 5)
    steps_per_cycle = config.get("retraining_interval_steps", 40)
    target_interval_sec = config.get("inference_cycle_sec", 1.0)
    
    PIPELINE_STATE.update({"total_cycles": max_cycles, "total_steps": steps_per_cycle})

    inference_processor = LSTMInference(config, "", 0, "", "LSTM_retrain")
    with shared_resource_lock:
        inference_processor.model = shared_model["model"]
        inference_processor.scaler = shared_model["scaler"]
        inference_processor.feature_list = shared_model["features"]
    
    mqtt_client = MqttInferenceClient(
        broker_ip=config['MQTT_BROKER_IP'], port=config['MQTT_PORT'], topic=config['MQTT_TOPIC'],
        on_message_callback=inference_processor.update_latest_data
    )
    mqtt_client.run()

    for cycle in range(max_cycles):
        PIPELINE_STATE.update({"cycle_count": cycle + 1, "retraining_status": "collecting"})
        logging.info(f"--- Zyklus {cycle + 1}/{max_cycles}: Starte Datensammlung. ---")

        for step in range(steps_per_cycle):
            start_cycle_time = time.perf_counter()
            while PIPELINE_STATE["is_paused"]: time.sleep(0.5)
            PIPELINE_STATE["steps_in_cycle"] = step + 1
            
            if inference_processor.latest_payload is None:
                logging.warning("Keine neuen MQTT-Daten. Warte...")
                time.sleep(target_interval_sec)
                continue

            with shared_resource_lock:
                inference_processor.model = shared_model["model"]
            
            input_data, timestamp, true_value = inference_processor._prepare_input_data()
            prediction_unscaled = None
            if input_data is not None:
                prediction_scaled, model_inference_time_ms = PipelineUtils.run_timed_inference(model=inference_processor.model, input_data=input_data)
                target_index = inference_processor.feature_list.index(inference_processor.target_feature)
                prediction_unscaled = PipelineUtils.safe_inverse_transform(
                    scaler=inference_processor.scaler, array=prediction_scaled.reshape(1, -1), target_index=target_index
                ).flatten()[0]
                rolling_preds = run_rolling_forecast(inference_processor.model, inference_processor.scaler, input_data, config, inference_processor.feature_list)
                
                total_processing_time_ms = (time.perf_counter() - start_cycle_time) * 1000
                cpu_load = PipelineUtils.get_cpu_usage()

            if prediction_unscaled is not None:
                log_msg = (
                    f"Step [{step+1}/{steps_per_cycle}] Pred: {prediction_unscaled:.2f}, True: {true_value:.2f}, "
                    f"Model Time: {model_inference_time_ms:.2f}ms, Total Time: {total_processing_time_ms:.2f}ms, CPU: {cpu_load:.1f}%"
                )
                logging.info(log_msg)

                if (total_processing_time_ms / 1000) > target_interval_sec:
                    logging.warning(f"ZEITÜBERSCHREITUNG! Verarbeitung ({total_processing_time_ms:.0f}ms) dauerte länger als das Zielintervall ({target_interval_sec*1000:.0f}ms).")

                prediction_entry = {
                    "datetime": timestamp, "prediction": prediction_unscaled, "true_value": true_value,
                    "rolling_forecast": rolling_preds, "cpu_load": PipelineUtils.get_cpu_usage(),
                    "model_inference_time_ms": model_inference_time_ms,
                    "total_processing_time_ms": (time.perf_counter() - start_cycle_time) * 1000
                }


                all_predictions.append(prediction_entry)
                retraining_data_list.append(inference_processor.latest_payload)

            inference_processor.latest_payload = None
            sleep_duration = target_interval_sec - (time.perf_counter() - start_cycle_time)
            if sleep_duration > 0: time.sleep(sleep_duration)

        logging.info(f"--- Zyklus {cycle + 1}: Datensammlung abgeschlossen. Starte Nachtraining. ---")
        with shared_resource_lock:
            model_copy = shared_model["model"]
            scaler_copy = deepcopy(shared_model["scaler"])
            features_copy = shared_model["features"][:]
            config_copy = deepcopy(shared_model["config"])
            retraining_data_buffer = pd.DataFrame(retraining_data_list)
            retraining_data_buffer['datetime'] = pd.to_datetime(retraining_data_buffer['datetime'])
            retraining_data_buffer = retraining_data_buffer.set_index('datetime')

            retraining_thread = threading.Thread(target=retraining_thread_task, args=(model_copy, scaler_copy, features_copy, retraining_data_buffer.copy(), config_copy), name=f"RetrainingThread-{cycle+1}")
            retraining_thread.start()
            retraining_data_list.clear()

    if 'retraining_thread' in locals() and retraining_thread.is_alive():
        retraining_thread.join()

    logging.info("--- PHASE 3: Maximale Zyklen erreicht. Pipeline beendet. ---")
    PIPELINE_STATE.update({"status": "finished", "is_finished": True, "is_paused": True})
    if mqtt_client:
        mqtt_client.client.loop_stop()
        mqtt_client.client.disconnect()

    if all_predictions:
        config, paths = PipelineUtils.setup_experiment(config, "LSTM_retrain_final", run_type='inference')
        pd.DataFrame(all_predictions).to_csv(os.path.join(paths.get("Prediction_Data"), f"final_predictions_{config['run_id']}.csv"), index=False)
        logging.info("Finale Vorhersagen gespeichert.")

# --- Hauptfunktion und Web-App ---

def main():
    """Hauptfunktion zum Starten der Pipeline und der Web-App."""
    parser = argparse.ArgumentParser(description="Vereinheitlichte ML-Pipeline mit optionalem Retraining und Web-UI")
    parser.add_argument('--retraining', action=argparse.BooleanOptionalAction, default=True, help="Aktiviert den Retraining-Modus (Standard: aktiviert).")
    parser.add_argument('--algorithm', type=str, default='lstm', choices=['random_forest', 'lstm'], help="Algorithmus für den Standard-Inferenzmodus.")
    parser.add_argument("--load_id", type=str, help="Optionale Run ID zum Laden von Artefakten.")
    parser.add_argument("--model_filename", type=str, help="Optionaler Dateiname des Modells.")
    args = parser.parse_args()

    # --- Konfiguration basierend auf den Argumenten erstellen ---
    if args.retraining:
        from config.config_ml_lstm import param_lstm_test
        config = param_lstm_test.copy()
        args.algorithm = 'lstm' # Retraining ist nur für LSTM implementiert
        logging.info("--- MODUS: Retraining aktiviert ---")
    else:
        if args.algorithm == 'random_forest':
            from config.config_ml_rf import param_rf_test
            config = param_rf_test.copy()
        elif args.algorithm == 'lstm':
            from config.config_ml_lstm import param_lstm_test
            config = param_lstm_test.copy()
        logging.info(f"--- MODUS: Standard-Inferenz | ALGORITHMUS: {args.algorithm} ---")
    
    config.update(CONFIG_LOAD_ARTIFACTS)
    config.update(MQTT_CONFIG)
    config['paths'] = CONFIG_PATH['paths']
    if args.load_id:
        config['load_id'] = args.load_id
        config['inference_mode'] = 'load_artifacts_path'
    if args.model_filename:
        config['model_filename'] = args.model_filename

    # --- Starte den entsprechenden Pipeline-Flow ---
    if args.retraining:
        config['inference_cycle_sec'] = config.get('inference_cycle_sec', 1.0)
        threading.Thread(target=initial_training, args=(config,), name="InitialTrainingThread", daemon=True).start()
        threading.Thread(target=inference_and_retraining_manager, args=(config,), name="ManagerThread", daemon=True).start()
        port = 5002
    else:
        config['inference_steps'] = config.get('inference_steps', 100)
        threading.Thread(target=prepare_standard_inference, args=(config, args.algorithm), daemon=True).start()
        port = 5001
        
    # --- Flask-App erstellen und konfigurieren ---
    template_folder = os.path.join(project_root, 'ML_Algorithms', 'templates')
    app = Flask(__name__, template_folder=template_folder)

    @app.route('/')
    def index():
        template = 'dashboard_retrain.html' if args.retraining else 'dashboard.html'
        return render_template(template, config=config)

    @app.route('/api/status')
    def get_status():
        status_data = PIPELINE_STATE.copy()
        if not args.retraining and inference_processor:
            status_data['current_step'] = inference_processor.step_counter
        return jsonify(status_data)

    @app.route('/api/data')
    def get_data():
        if args.retraining:
            if not all_predictions: return jsonify({"status": "waiting"})
            latest_data = deepcopy(all_predictions[-1])
            if isinstance(latest_data['datetime'], (datetime, pd.Timestamp)):
                latest_data['datetime'] = latest_data['datetime'].isoformat()
            interval_sec = config.get("inference_cycle_sec", 1.0)
            future_dates = [(datetime.fromisoformat(latest_data['datetime']) + timedelta(seconds=(i+1) * interval_sec)).isoformat() for i in range(len(latest_data.get("rolling_forecast", [])))]
            latest_data['rolling_forecast_dates'] = future_dates
            return jsonify({"status": "success", "data": latest_data})
        else: # Standard-Inferenz
            step = request.args.get('step', type=int, default=0)
            if not inference_processor or step >= len(inference_processor.results_buffer):
                return jsonify({"status": "waiting"}), 200
            raw_data = inference_processor.results_buffer[step]
            predictions_list = [raw_data[f"prediction_step_{i+1}"] for i in range(config.get("horizon", 1)) if f"prediction_step_{i+1}" in raw_data]
            timestamp_obj = raw_data["datetime"]
            interval_sec = config.get("inference_interval_sec", 1.0)
            future_dates = [(timestamp_obj + pd.Timedelta(seconds=i * interval_sec)).isoformat() for i in range(len(predictions_list))]
            output_data = {
                "date": timestamp_obj.isoformat(),
                "true_value": raw_data["true_value"],
                "predicted_value_step_1": predictions_list[0] if predictions_list else None,
                "future_forecast": {"dates": future_dates, "values": predictions_list},
                "cpu_load": raw_data["cpu_load_percent"],
                "inference_time_ms": raw_data["inference_time_ms"]
            }
            return jsonify({"status": "success", "data": output_data}), 200

    @app.route('/api/run_inference', methods=['POST'])
    def run_inference_endpoint():
        if args.retraining:
             return jsonify({"error": "Aktion im Retraining-Modus nicht verfügbar."}), 400
        if PIPELINE_STATE.get("status") not in ["ready_for_inference", "finished"]:
            return jsonify({"error": "Nicht bereit für Inferenz."}), 400
        start_standard_inference_flow(config)
        return jsonify({"status": "Inferenzprozess gestartet."})

    @app.route('/api/control', methods=['POST'])
    def control_pipeline():
        if not args.retraining:
            return jsonify({"error": "Aktion nur im Retraining-Modus verfügbar."}), 400
        action = request.json.get('action')
        with shared_resource_lock:
            if action == 'pause': PIPELINE_STATE['is_paused'] = True
            elif action == 'resume': PIPELINE_STATE['is_paused'] = False
        return jsonify({"status": "ok", "is_paused": PIPELINE_STATE['is_paused']})

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    local_ip = get_local_ip()
    logging.info(f"\n🚀 Webserver startet. Öffnen Sie http://{local_ip}:{port} in Ihrem Browser.")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()