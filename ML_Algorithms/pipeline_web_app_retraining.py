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
all_predictions = []
# --- PERFORMANCE-OPTIMIERUNG: Liste anstelle von DataFrame für die Datensammlung ---
retraining_data_list = []

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

def initial_phase(config: dict, algorithm: str):
    """
    GENERALISIERT: Führt entweder das initiale Training durch oder lädt ein existierendes Modell.
    """
    global shared_model
    
    # --- Logik für dynamische Klassen-Auswahl ---
    if algorithm == 'lstm':
        from ML_Algorithms.LSTM.LSTM_train import LSTMTrainer
        from ML_Algorithms.LSTM.LSTM_inference import LSTMInference
        TrainerClass = LSTMTrainer
        InferenceClass = LSTMInference
        folder_flag = "LSTM_retrain"
    elif algorithm == 'random_forest':
        # Annahme: Es existieren äquivalente RF-Klassen
        from ML_Algorithms.Random_Forest.rf_train import RFTrainer
        from ML_Algorithms.Random_Forest.rf_inference import RFInference
        TrainerClass = RFTrainer
        InferenceClass = RFInference
        folder_flag = "RF_retrain"
    else:
        raise ValueError(f"Unbekanntes Algorithm: {algorithm}")

    # --- Entscheidung: Trainieren oder Laden ---
    if config.get('load_id'):
        logging.info(f"--- PHASE 1: Überspringe Training. Lade Artefakte von run_id: {config['load_id']} ---")
        try:
            # Lade-Prozessor initialisieren und Artefakte laden
            loader_proc = InferenceClass(config, "", 0, "", folder_flag)
            loader_proc.load_artifacts()
            model, scaler, features = loader_proc.model, loader_proc.scaler, loader_proc.feature_list
            logging.info("--- PHASE 1: Artefakte erfolgreich geladen. ---")
        except Exception as e:
            logging.error(f"Fehler beim Laden der Artefakte: {e}", exc_info=True)
            PIPELINE_STATE.update({"status": "error", "error_message": str(e)})
            return
    else:
        logging.info(f"--- PHASE 1: Kein 'load_id' gefunden. Starte initiales Training für {algorithm}. ---")
        trainer = TrainerClass(config=config, folder_flag=folder_flag)
        model, scaler, features = trainer.run(save_artifacts=True) # Speichern ist hier sinnvoll
        logging.info("--- PHASE 1: Initiales Training abgeschlossen. ---")

    # --- Globales Modell aktualisieren ---
    with shared_resource_lock:
        shared_model["model"] = model
        shared_model["scaler"] = scaler
        shared_model["features"] = features
        shared_model["config"] = config
        PIPELINE_STATE["status"] = "inference_running"


def retraining_thread_task(current_model, scaler, features, retraining_data, config):
    """Führt das Nachtraining in einem separaten Thread aus."""
    global shared_model
    logging.info("--- RETRAINING THREAD: Startet Nachtraining. ---")
    PIPELINE_STATE["retraining_status"] = "training"
    
    # HINWEIS: Die folgende Logik ist spezifisch für Keras/LSTM-Modelle.
    # Für andere Modelle (z.B. Scikit-learn) müsste hier eine andere Retraining-Strategie implementiert werden.
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

        # Klonen und erneutes Kompilieren ist für Keras-Modelle erforderlich
        model_to_retrain = tf.keras.models.clone_model(current_model)
        model_to_retrain.set_weights(current_model.get_weights())
        model_to_retrain.compile(optimizer=config.get("optimizer", "adam"), loss=config.get("loss", "mse"))
        
        # Inkrementelles Training
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
    # Diese Funktion ist generisch und muss nicht geändert werden
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


def inference_and_retraining_manager(config: dict, algorithm: str):
    """Haupt-Controller für Inferenz, Datensammlung und Retraining."""
    global inference_processor, all_predictions, retraining_data_list
    
    while PIPELINE_STATE["status"] == "initializing":
        time.sleep(1)
    if PIPELINE_STATE["status"] == "error":
        logging.error("Pipeline startet nicht wegen eines Fehlers in der Initialisierungsphase.")
        return
        
    logging.info("--- PHASE 2: Inferenz-Manager startet. ---")
    
    max_cycles = config.get("retraining_cycles", 5)
    steps_per_cycle = config.get("retraining_interval_steps", 40)
    target_interval_sec = config.get("inference_cycle_sec", 1.0) 
    
    PIPELINE_STATE.update({"total_cycles": max_cycles, "total_steps": steps_per_cycle})

    # --- Dynamische Auswahl der Inferenz-Klasse ---
    if algorithm == 'lstm':
        from ML_Algorithms.LSTM.LSTM_inference import LSTMInference
        InferenceClass = LSTMInference
        folder_flag = "LSTM_retrain"
    else: # Fügen Sie hier weitere 'elif' für andere Algorithmen hinzu
        from ML_Algorithms.Random_Forest.rf_inference import RFInference
        InferenceClass = RFInference
        folder_flag = "RF_retrain"

    inference_processor = InferenceClass(config, "", 0, "", folder_flag)
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
            while PIPELINE_STATE["is_paused"]: time.sleep(0.5)
            PIPELINE_STATE["steps_in_cycle"] = step + 1
            
            if inference_processor.latest_payload is None:
                logging.warning("Keine neuen MQTT-Daten verfügbar. Warte auf nächsten Zyklus.")
                time.sleep(target_interval_sec)
                continue

            with shared_resource_lock:
                inference_processor.model = shared_model["model"]
            
            input_data, timestamp, true_value = inference_processor._prepare_input_data()
            
            prediction_unscaled, model_inference_time_ms = None, 0
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
                # Log- und Speicherlogik...
                # ... (unverändert)
                prediction_entry = { "datetime": timestamp, "prediction": prediction_unscaled, "true_value": true_value, "cpu_load": cpu_load, "total_processing_time_ms": total_processing_time_ms, "model_inference_time_ms": model_inference_time_ms, "rolling_forecast": rolling_preds }
                all_predictions.append(prediction_entry)
                
                # --- PERFORMANCE-OPTIMIERUNG: An die Liste anhängen anstatt pd.concat ---
                retraining_data_list.append(inference_processor.latest_payload)

            inference_processor.latest_payload = None
            cycle_duration = time.perf_counter() - start_cycle_time
            sleep_duration = target_interval_sec - cycle_duration
            if sleep_duration > 0: time.sleep(sleep_duration)

        logging.info(f"--- Zyklus {cycle + 1}: Datensammlung abgeschlossen. Starte Nachtraining. ---")
        
        # --- PERFORMANCE-OPTIMIERUNG: DataFrame einmalig erstellen ---
        retraining_data_buffer = pd.DataFrame(retraining_data_list)
        retraining_data_buffer['datetime'] = pd.to_datetime(retraining_data_buffer['datetime'])
        retraining_data_buffer = retraining_data_buffer.set_index('datetime')
        
        with shared_resource_lock:
            model_copy, scaler_copy, features_copy, config_copy = (shared_model["model"], deepcopy(shared_model["scaler"]), shared_model["features"][:], deepcopy(shared_model["config"]))
        
        retraining_thread = threading.Thread(target=retraining_thread_task, args=(model_copy, scaler_copy, features_copy, retraining_data_buffer.copy(), config_copy), name=f"RetrainingThread-{cycle+1}")
        retraining_thread.start()
        
        retraining_data_list = [] # Liste für den nächsten Zyklus zurücksetzen

    # --- Cleanup-Phase ---
    # ... (unverändert)
    if 'retraining_thread' in locals() and retraining_thread.is_alive(): retraining_thread.join()
    logging.info("--- PHASE 3: Maximale Zyklen erreicht. Pipeline beendet. ---")
    PIPELINE_STATE.update({"status": "finished", "is_finished": True, "is_paused": True})
    mqtt_client.client.loop_stop()
    mqtt_client.client.disconnect()
    # ... (Speichern der Ergebnisse)


def main():
    """Hauptfunktion zum Starten der Pipeline und der Web-App."""
    parser = argparse.ArgumentParser(description="ML Pipeline with Incremental Retraining and Web UI")
    # --- GENERALISIERT: Kommandozeilen-Argumente ---
    parser.add_argument('--algorithm', type=str, required=True, choices=['lstm', 'random_forest'])
    parser.add_argument("--load_id", type=str, help="Optional: Die ID eines früheren Laufs, um Artefakte zu laden und Training zu überspringen.")
    args = parser.parse_args()

    # --- GENERALISIERT: Dynamische Konfigurations-Erstellung ---
    if args.algorithm == 'lstm':
        from config.config_ml_lstm import param_lstm_test
        config = param_lstm_test.copy()
    elif args.algorithm == 'random_forest':
        from config.config_ml_random_forest import param_rf_test
        config = param_rf_test.copy()
    else:
        sys.exit(f"Unbekannter Algorithmus: {args.algorithm}")
        
    config.update(CONFIG_LOAD_ARTIFACTS)
    config.update(MQTT_CONFIG)
    config['paths'] = CONFIG_PATH['paths']
    config['inference_cycle_sec'] = config.get('inference_cycle_sec', 1.0) 
    if args.load_id:
        config['load_id'] = args.load_id

    # --- Starte die generalisierten Threads ---
    threading.Thread(target=initial_phase, args=(config, args.algorithm), name="InitialPhaseThread", daemon=True).start()
    threading.Thread(target=inference_and_retraining_manager, args=(config, args.algorithm), name="ManagerThread", daemon=True).start()

    # --- Flask App (unverändert) ---
    app = Flask(__name__, template_folder=os.path.join(project_root, 'ML_Algorithms', 'templates'))
    @app.route('/')
    def index(): return render_template('dashboard_retrain.html', config=config)
    @app.route('/api/status')
    def get_status(): return jsonify(PIPELINE_STATE)
    @app.route('/api/data')
    def get_data():
        if not all_predictions: return jsonify({"status": "waiting"})
        latest_data = deepcopy(all_predictions[-1])
        if isinstance(latest_data['datetime'], (datetime, pd.Timestamp)): latest_data['datetime'] = latest_data['datetime'].isoformat()
        interval_sec = config.get("inference_cycle_sec", 1.0)
        future_dates = [(datetime.fromisoformat(latest_data['datetime']) + timedelta(seconds=(i+1) * interval_sec)).isoformat() for i in range(len(latest_data.get("rolling_forecast", [])))]
        latest_data['rolling_forecast_dates'] = future_dates
        return jsonify({"status": "success", "data": latest_data})
    @app.route('/api/control', methods=['POST'])
    def control_pipeline():
        action = request.json.get('action')
        with shared_resource_lock:
            if action == 'pause': PIPELINE_STATE['is_paused'] = True
            elif action == 'resume': PIPELINE_STATE['is_paused'] = False
        return jsonify({"status": "ok", "is_paused": PIPELINE_STATE['is_paused']})

    log = logging.getLogger('werkzeug'); log.setLevel(logging.WARNING)
    local_ip = get_local_ip()
    logging.info(f"🚀 Web server starting. Open http://{local_ip}:5002 in your browser.")
    app.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()