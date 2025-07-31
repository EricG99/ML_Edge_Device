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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Anwendungsimporte ---
from config.config_general import CONFIG_PATH, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
from ML_Helpfunctions import Feature_Engeneering as fe
from ML_Helpfunctions import LSTM_Utils

# --- Trainer- und Inferenz-Klassen ---
from ML_Algorithms.LSTM.LSTM_train import LSTMTrainer
from ML_Algorithms.LSTM.LSTM_inference import LSTMInference

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

# --- Globale Objekte für das Modell und die Daten ---
shared_model = { "model": None, "scaler": None, "features": None, "config": None }
inference_processor = None
retraining_data_list = [] 
all_predictions = []


def get_local_ip():
    """Ermittelt die lokale IP-Adresse des Geräts."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        if 's' in locals() and s:
            s.close()
    return ip

def initial_training(config: dict) -> None:
    """Führt das initiale Training des Modells durch."""
    global shared_model
    logging.info("--- PHASE 1: Initiales Training startet ---")
    trainer = LSTMTrainer(config=config, folder_flag="LSTM_retrain")
    model, scaler, features = trainer.run(save_artifacts=False)
    
    with shared_resource_lock:
        shared_model["model"] = model
        shared_model["scaler"] = scaler
        shared_model["features"] = features
        shared_model["config"] = config
        PIPELINE_STATE["status"] = "inference_running"
    logging.info("--- PHASE 1: Initiales Training abgeschlossen. Starte Inferenz automatisch. ---")


def retraining_thread_task(current_model, scaler, features, retraining_data, config):
    """Führt das Nachtraining in einem separaten Thread aus."""
    global shared_model
    logging.info("--- RETRAINING THREAD: Startet Nachtraining. ---")
    PIPELINE_STATE["retraining_status"] = "training"
    try:
        retraining_data_featured, _ = fe.add_all_features(retraining_data, config)
        
        # *** KORREKTUR: Sicherere Operation ohne 'inplace=True' ***
        # Erstellt eine neue Kopie des DataFrames, die nur die gültigen Zeilen enthält.
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
    if horizon <= 1:
        return [], []

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
    global inference_processor, all_predictions, retraining_data_list
        
    while PIPELINE_STATE["status"] == "initializing":
        time.sleep(1)
        
    logging.info("--- PHASE 2: Inferenz-Manager startet. ---")
    
    max_cycles = config.get("retraining_cycles", 5)
    steps_per_cycle = config.get("retraining_interval_steps", 40)
    target_interval_sec = config.get("inference_cycle_sec", 1.0) 
    
    PIPELINE_STATE["total_cycles"] = max_cycles
    PIPELINE_STATE["total_steps"] = steps_per_cycle

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
        PIPELINE_STATE["cycle_count"] = cycle + 1
        PIPELINE_STATE["retraining_status"] = "collecting"
        logging.info(f"--- Zyklus {cycle + 1}/{max_cycles}: Starte Datensammlung. ---")

        for step in range(steps_per_cycle):
            start_cycle_time = time.perf_counter()

            while PIPELINE_STATE["is_paused"]:
                time.sleep(0.5)

            PIPELINE_STATE["steps_in_cycle"] = step + 1
            
            if inference_processor.latest_payload is None:
                logging.warning("Keine neuen MQTT-Daten verfügbar. Warte auf nächsten Zyklus.")
                time.sleep(target_interval_sec)
                continue

            with shared_resource_lock:
                inference_processor.model = shared_model["model"]
            
            input_data, timestamp, true_value = inference_processor._prepare_input_data()
            
            prediction_unscaled = None
            model_inference_time_ms = 0
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
                    "cpu_load": cpu_load,
                    "total_processing_time_ms": total_processing_time_ms,
                    "model_inference_time_ms": model_inference_time_ms,
                    "rolling_forecast": rolling_preds
                }
                all_predictions.append(prediction_entry)
                
                retraining_data_list.append(inference_processor.latest_payload)


            inference_processor.latest_payload = None

            cycle_duration = time.perf_counter() - start_cycle_time
            sleep_duration = target_interval_sec - cycle_duration
            if sleep_duration > 0:
                time.sleep(sleep_duration)

        logging.info(f"--- Zyklus {cycle + 1}: Datensammlung abgeschlossen. Starte Nachtraining. ---")
        with shared_resource_lock:
            model_copy, scaler_copy, features_copy, config_copy = (
                shared_model["model"], deepcopy(shared_model["scaler"]), 
                shared_model["features"][:], deepcopy(shared_model["config"])
            )
            retraining_data_buffer = pd.DataFrame(retraining_data_list)
            retraining_data_buffer['datetime'] = pd.to_datetime(retraining_data_buffer['datetime'])
            retraining_data_buffer = retraining_data_buffer.set_index('datetime')

            retraining_thread = threading.Thread(target=retraining_thread_task, args=(model_copy, scaler_copy, features_copy, retraining_data_buffer.copy(), config_copy), name=f"RetrainingThread-{cycle+1}")
            retraining_thread.start()
            retraining_data_list = [] # Reset der Liste

    if 'retraining_thread' in locals() and retraining_thread.is_alive():
        retraining_thread.join()

    logging.info("--- PHASE 3: Maximale Zyklen erreicht. Pipeline beendet. ---")
    PIPELINE_STATE.update({"status": "finished", "is_finished": True, "is_paused": True})
    mqtt_client.client.loop_stop()
    mqtt_client.client.disconnect()

    if all_predictions:
        config, paths = PipelineUtils.setup_experiment(config, "LSTM_retrain_final", run_type='inference')
        results_df = pd.DataFrame(all_predictions)
        results_df.to_csv(os.path.join(paths.get("Prediction_Data"), f"final_predictions_{config['run_id']}.csv"), index=False)
        logging.info(f"Finale Vorhersagen gespeichert.")


def main():
    """Hauptfunktion zum Starten der Pipeline und der Web-App."""
    parser = argparse.ArgumentParser(description="LSTM Pipeline with Incremental Retraining and Web UI")
    args = parser.parse_args()

    from config.config_ml_lstm import param_lstm_test
    config = param_lstm_test.copy()
    config.update(CONFIG_LOAD_ARTIFACTS)
    config.update(MQTT_CONFIG)
    config['paths'] = CONFIG_PATH['paths']
    config['inference_cycle_sec'] = config.get('inference_cycle_sec', 1.0) 
    
    threading.Thread(target=initial_training, args=(config,), name="InitialTrainingThread", daemon=True).start()
    threading.Thread(target=inference_and_retraining_manager, args=(config,), name="ManagerThread", daemon=True).start()

    app = Flask(__name__, template_folder=os.path.join(project_root, 'ML_Algorithms', 'templates'))

    @app.route('/')
    def index():
        return render_template('dashboard_retrain.html', config=config)

    @app.route('/api/status')
    def get_status():
        return jsonify(PIPELINE_STATE)

    @app.route('/api/data')
    def get_data():
        if not all_predictions:
            return jsonify({"status": "waiting"})
        
        latest_data = deepcopy(all_predictions[-1])
        
        if isinstance(latest_data['datetime'], (datetime, pd.Timestamp)):
            latest_data['datetime'] = latest_data['datetime'].isoformat()
        
        interval_sec = config.get("inference_cycle_sec", 1.0)
        future_dates = [(datetime.fromisoformat(latest_data['datetime']) + timedelta(seconds=(i+1) * interval_sec)).isoformat() for i in range(len(latest_data.get("rolling_forecast", [])))]
        latest_data['rolling_forecast_dates'] = future_dates

        return jsonify({"status": "success", "data": latest_data})

    @app.route('/api/control', methods=['POST'])
    def control_pipeline():
        """Steuert den Pausen- und Startzustand der Pipeline."""
        action = request.json.get('action')
        with shared_resource_lock:
            if action == 'pause':
                PIPELINE_STATE['is_paused'] = True
                logging.info("UI-Aktion: Pipeline pausiert.")
            elif action == 'resume':
                PIPELINE_STATE['is_paused'] = False
                logging.info("UI-Aktion: Pipeline fortgesetzt.")
        return jsonify({"status": "ok", "is_paused": PIPELINE_STATE['is_paused']})

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    local_ip = get_local_ip()
    logging.info(f"🚀 Web server starting. Open http://{local_ip}:5002 in your browser.")
    app.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
