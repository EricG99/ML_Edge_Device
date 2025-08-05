# pipeline_web_app.py
#
# BEISPIEL-AUFRUFE:
# -----------------
# Modus: Retraining (Initiales Training + kontinuierliches Nachtraining)
# > python pipeline_web_app.py --retraining --algorithm lstm
# > python pipeline_web_app.py --retraining --algorithm random_forest
#
# Modus: No-Retraining (Einmaliges Training + lange Inferenzphase ohne Nachtraining)
# > python pipeline_web_app.py --no-retraining --algorithm lstm
#
# Modus: No-Retraining (Laden eines existierenden Modells + lange Inferenzphase)
# > python pipeline_web_app.py --no-retraining --algorithm lstm --load_id <IHRE_RUN_ID>
#

import time
import logging
import argparse
import sys
import threading
import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request
from copy import deepcopy
from datetime import datetime, timedelta
import tensorflow as tf

# Benötigte Imports für das In-Memory-Retraining
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# --- Systempfad-Setup ---
# Stellt sicher, dass die übergeordneten Projektverzeichnisse für Importe erreichbar sind
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from config.config_general import CONFIG_PATH, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS
    from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
    from ML_Helpfunctions.MQTT_Client import MqttInferenceClient
    from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
    from ML_Helpfunctions import Feature_Engeneering as fe
except ImportError as e:
    print(f"Fehler beim Importieren der Basis-Module: {e}")
    print("Stellen Sie sicher, dass die Verzeichnisstruktur korrekt ist und die __init__.py Dateien vorhanden sind.")
    sys.exit(1)

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
    "mode": "unknown"
}
shared_resource_lock = threading.Lock()

# --- Globale, dynamisch geladene Objekte ---
shared_model = {"model": None, "scaler": None, "features": None, "config": None, "initial_training_data": None}
all_predictions = [] # Speichert alle Vorhersagen für die Web-App

# =============================================================================
# CORE LOGIC: TRAINING, RETRAINING, AND INFERENCE
# =============================================================================

def initial_training(config: dict, trainer_class, folder_flag: str):
    """
    Führt das initiale Training für alle Modi durch, die es benötigen.
    Die trainierten Artefakte werden in `shared_model` gespeichert.
    """
    global shared_model
    logging.info(f"--- PHASE 1: Initiales Training für {folder_flag} startet ---")
    try:
        trainer = trainer_class(config=config, folder_flag=folder_flag)
        pipeline = trainer._setup_pipeline()
        initial_data_df, _ = pipeline._load_data(mode='train')
        model, scaler, features = trainer.run(save_artifacts=True) # Speichert Artefakte für Nachvollziehbarkeit

        with shared_resource_lock:
            shared_model.update({
                "model": model, "scaler": scaler, "features": features,
                "config": config, "initial_training_data": initial_data_df
            })
        logging.info("--- PHASE 1: Initiales Training abgeschlossen. ---")
        with shared_resource_lock:
            PIPELINE_STATE["status"] = "ready_for_inference"
    except Exception as e:
        logging.error(f"Fehler während des initialen Trainings: {e}", exc_info=True)
        with shared_resource_lock:
            PIPELINE_STATE["status"] = "error"
            PIPELINE_STATE["error_message"] = str(e)


def retraining_thread_task(retraining_data_df: pd.DataFrame, algorithm: str):
    """
    Führt das Nachtraining in einem separaten Thread aus.
    Random Forest wird komplett im Speicher neu trainiert, um Datei-I/O zu vermeiden.
    """
    global shared_model
    logging.info(f"--- RETRAINING THREAD ({algorithm}): Startet Nachtraining. ---")
    with shared_resource_lock:
        PIPELINE_STATE["retraining_status"] = "training"

    try:
        with shared_resource_lock:
            # Notwendige Objekte aus dem globalen Zustand holen und kopieren
            config = deepcopy(shared_model['config'])
            initial_data = shared_model['initial_training_data']

        if algorithm == 'lstm':
            # Die Logik für LSTM (inkrementelles Training) bleibt unverändert.
            with shared_resource_lock:
                 # Wichtig: Klonen des Modells für Thread-Sicherheit bei Keras
                if hasattr(shared_model['model'], 'clone_model'):
                    current_model = tf.keras.models.clone_model(shared_model['model'])
                    current_model.set_weights(shared_model['model'].get_weights())
                else:
                    current_model = deepcopy(shared_model['model'])

            logging.info("LSTM-Nachtraining (inkrementell) wird durchgeführt...")
            retraining_data_featured, _ = fe.add_all_features(retraining_data_df, config)
            retraining_data_featured = retraining_data_featured.dropna()

            if not retraining_data_featured.empty:
                scaled_data = shared_model['scaler'].transform(retraining_data_featured[shared_model['features']])
                X_retrain, y_retrain = LoadPrepareData.convert_data_to_sliding_window(
                    scaled_data, lag_horizon=config["lags"], forecast_horizon=config["horizon"]
                )
                if len(X_retrain) > 0:
                    current_model.compile(optimizer=config.get("optimizer", "adam"), loss=config.get("loss", "mse"))
                    current_model.fit(X_retrain, y_retrain, epochs=config.get("retraining_epochs", 5), batch_size=config.get("batch_size", 32), verbose=0)
                    with shared_resource_lock:
                        shared_model["model"] = current_model
                        logging.info("--- RETRAINING THREAD (LSTM): Modellaustausch erfolgreich! ---")

        elif algorithm == 'random_forest':
            # DEFINITIVE LÖSUNG: Vollständiges Neutraining für Random Forest im Speicher.
            logging.info("Kombiniere alte und neue Daten für RF-Nachtraining...")
            combined_data = pd.concat([initial_data, retraining_data_df]).drop_duplicates().sort_index()

            # 1. Feature Engineering auf den kombinierten Daten
            logging.info("Führe Feature Engineering für kombinierten Datensatz durch...")
            combined_df_featured, features_dict = fe.add_all_features(combined_data, config)
            new_features = features_dict["all"]
            combined_df_featured.dropna(inplace=True)

            # 2. Daten für Training vorbereiten (X und y)
            X_retrain = combined_df_featured[new_features]
            y_retrain = combined_df_featured[config["base_features"][0]]

            # 3. Scaler neu anpassen und Daten transformieren
            logging.info("Passe Scaler neu an die kombinierten Daten an...")
            scaler_class = RobustScaler if config.get("scaler_type", "minmax") == "robust" else MinMaxScaler
            new_scaler = scaler_class()
            X_retrain_scaled = new_scaler.fit_transform(X_retrain)

            # 4. Neues Modell trainieren
            logging.info("Trainiere neues Random Forest Modell...")
            model_params = config.get("model_params", {})
            new_model = RandomForestRegressor(**model_params)
            new_model.fit(X_retrain_scaled, y_retrain.values)
            
            # 5. Globales Modell, Scaler und Features atomar austauschen
            with shared_resource_lock:
                shared_model.update({
                    "model": new_model, 
                    "scaler": new_scaler, 
                    "features": new_features
                })
                logging.info("--- RETRAINING THREAD (RF): Modellaustausch erfolgreich! ---")

    except Exception as e:
        logging.error(f"RETRAINING THREAD: Fehler: {e}", exc_info=True)
    finally:
        with shared_resource_lock:
            PIPELINE_STATE["retraining_status"] = "idle"


def inference_manager(config: dict, inference_class, folder_flag: str, algorithm: str, mode: str):
    """
    Einheitlicher Inferenz-Manager. Nutzt die direkte Multi-Output-Strategie für alle Modelle.
    VERBESSERTE VERSION MIT TRY...FINALLY FÜR ROBUSTES AUFRÄUMEN.
    """
    global all_predictions
    retraining_data_list = []
    mqtt_client = None  # WICHTIG: Variable am Anfang initialisieren

    try:
        # Warten, bis die Initialisierung (Training oder Laden) abgeschlossen ist
        while PIPELINE_STATE["status"] == "initializing":
            time.sleep(1)
        if PIPELINE_STATE["status"] == "error":
            logging.error("Inferenz-Manager startet nicht, da ein Fehler bei der Initialisierung aufgetreten ist.")
            return  # Frühzeitiges Beenden

        # Status auf "inference_running" setzen, damit das Frontend Daten abruft
        with shared_resource_lock:
            PIPELINE_STATE["status"] = "inference_running"
        logging.info(f"--- INFERENCE MANAGER: Startet im Modus '{mode}' ---")
        
        # Konfiguration der Zyklen und Schritte basierend auf dem Modus
        if mode == "retraining":
            max_cycles = config.get("retraining_cycles", 2)
            steps_per_cycle = config.get("retraining_interval_steps", 10)
            target_interval_sec = config.get("inference_cycle_sec", 1.0)
        else:  # no_retraining
            max_cycles = 1
            steps_per_cycle = config.get("inference_steps", 500)
            target_interval_sec = config.get("inference_interval_sec", 1.0)

        with shared_resource_lock:
            PIPELINE_STATE.update({"total_cycles": max_cycles, "total_steps": steps_per_cycle, "mode": mode})

        # Inferenz-Prozessor initialisieren
        inference_processor = inference_class(config, "", 0, "", folder_flag)
        
        # Modell-Setup
        model_is_trained_in_memory = shared_model.get("model") is not None

        if model_is_trained_in_memory:
            logging.info("Verwende das neu trainierte Modell aus dem initialen Lauf.")
            with shared_resource_lock:
                inference_processor.model = shared_model["model"]
                inference_processor.scaler = shared_model["scaler"]
                inference_processor.feature_list = shared_model["features"]
                inference_processor.target_feature = shared_model["config"]["base_features"][0]
        elif config.get('load_id'):
            try:
                logging.info(f"Lade Artefakte von Run ID: {config['load_id']}")
                inference_processor.load_artifacts()
            except Exception as e:
                logging.error(f"Fehler beim Laden der Artefakte: {e}", exc_info=True)
                with shared_resource_lock: PIPELINE_STATE.update({"status": "error", "error_message": str(e)})
                return # Frühzeitiges Beenden
        else:
            logging.error("Kein trainiertes Modell und keine load_id zum Laden gefunden. Inferenz kann nicht starten.")
            with shared_resource_lock: PIPELINE_STATE.update({"status": "error", "error_message": "Kein Modell verfügbar."})
            return # Frühzeitiges Beenden

        # MQTT Client starten
        mqtt_client = MqttInferenceClient(
            broker_ip=config['MQTT_BROKER_IP'], port=config['MQTT_PORT'], topic=config['MQTT_TOPIC'],
            on_message_callback=inference_processor.update_latest_data
        )
        mqtt_client.run()

        # Hauptschleife für Inferenz und Retraining
        for cycle in range(max_cycles):
            with shared_resource_lock: PIPELINE_STATE.update({"cycle_count": cycle + 1, "retraining_status": "collecting" if mode == "retraining" else "idle"})
            logging.info(f"--- Zyklus {cycle + 1}/{max_cycles}: Starte Inferenz. ---")

            for step in range(steps_per_cycle):
                start_cycle_time = time.perf_counter()
                while PIPELINE_STATE["is_paused"]: time.sleep(0.5)
                if PIPELINE_STATE["is_finished"]: break

                with shared_resource_lock: PIPELINE_STATE["steps_in_cycle"] = step + 1
                if inference_processor.latest_payload is None:
                    time.sleep(target_interval_sec)
                    continue

                with shared_resource_lock:
                    if model_is_trained_in_memory:
                        inference_processor.model = shared_model.get("model")
                
                input_data, timestamp, true_value = inference_processor._prepare_input_data()
                
                if input_data is not None:
                    prediction_scaled, model_inference_time_ms = PipelineUtils.run_timed_inference(
                        model=inference_processor.model, input_data=input_data
                    )
                    
                    try:
                        target_index = inference_processor.feature_list.index(inference_processor.target_feature)
                    except (ValueError, AttributeError):
                        target_index = 0
                    
                    predictions_unscaled_all = PipelineUtils.safe_inverse_transform(
                        scaler=inference_processor.scaler, array=prediction_scaled.reshape(1, -1), target_index=target_index
                    ).flatten()

                    total_processing_time_ms = (time.perf_counter() - start_cycle_time) * 1000
                    prediction_entry = {
                        "datetime": timestamp,
                        "prediction": predictions_unscaled_all[0] if len(predictions_unscaled_all) > 0 else None,
                        "true_value": true_value,
                        "rolling_forecast": predictions_unscaled_all.tolist(),
                        "cpu_load": PipelineUtils.get_cpu_usage(),
                        "ram_usage": PipelineUtils.get_memory_usage(),
                        "model_inference_time_ms": model_inference_time_ms,
                        "total_processing_time_ms": total_processing_time_ms
                    }
                    with shared_resource_lock:
                        all_predictions.append(prediction_entry)
                    
                    pred_str = f"{prediction_entry['prediction']:.2f}" if prediction_entry['prediction'] is not None else "N/A"
                    true_str = f"{prediction_entry['true_value']:.2f}" if prediction_entry['true_value'] is not None else "N/A"
                    log_msg = (
                        f"Step [{step+1}/{steps_per_cycle}] | "
                        f"Time: {timestamp.strftime('%H:%M:%S')} | "
                        f"Pred: {pred_str} | "
                        f"True: {true_str} | "
                        f"Infer Time: {model_inference_time_ms:.2f}ms | "
                        f"Total Time: {total_processing_time_ms:.2f}ms"
                    )
                    logging.info(log_msg)

                    if mode == "retraining":
                        retraining_data_list.append(inference_processor.latest_payload)
                
                inference_processor.latest_payload = None
                sleep_duration = target_interval_sec - (time.perf_counter() - start_cycle_time)
                if sleep_duration > 0: time.sleep(sleep_duration)
            
            if PIPELINE_STATE["is_finished"]: break

            # Retraining nach jedem Zyklus (nur im Retraining-Modus)
            if mode == "retraining" and cycle < max_cycles - 1:
                logging.info(f"--- Zyklus {cycle + 1}: Datensammlung abgeschlossen. Starte Nachtraining. ---")
                retraining_data_buffer = pd.DataFrame(retraining_data_list)
                retraining_data_buffer.columns = retraining_data_buffer.columns.str.lower()
                retraining_data_buffer['datetime'] = pd.to_datetime(retraining_data_buffer['datetime'])
                retraining_data_buffer = retraining_data_buffer.set_index('datetime')
                
                retraining_thread = threading.Thread(
                    target=retraining_thread_task, args=(retraining_data_buffer.copy(), algorithm), name=f"RetrainingThread-{cycle+1}"
                )
                retraining_thread.start()
                retraining_data_list.clear()

    finally:
        # Dieser Block wird IMMER ausgeführt, egal ob der try-Block erfolgreich war,
        # einen Fehler hatte oder durch 'return' verlassen wurde.
        logging.info("--- Pipeline beendet. Aufräumen und Speichern. ---")
        with shared_resource_lock: 
            PIPELINE_STATE.update({"status": "finished", "is_finished": True})
        
        if mqtt_client:
            logging.info("Trenne MQTT-Client...")
            try:
                # Korrektes, zweistufiges Beenden über das interne .client-Objekt
                mqtt_client.client.loop_stop()
                mqtt_client.client.disconnect()
                logging.info("MQTT-Client erfolgreich getrennt.")
            except Exception as e:
                logging.error(f"Fehler beim Trennen des MQTT-Clients: {e}", exc_info=True)

        if all_predictions:
            # === ÜBERARBEITETE SPEICHERLOGIK ===
            # Speichert die Vorhersagen im Ordner des ursprünglichen Laufs, anstatt einen neuen "_final" Ordner zu erstellen.
            logging.info("Speichere finale Vorhersagen...")
            
            # Die `config` Variable in diesem Scope enthält bereits die korrekten, versionierten Pfade vom initialen Lauf.
            prediction_dir = config.get("paths", {}).get("Prediction_Data")
            run_id = config.get("run_id", "final_run")

            if prediction_dir and os.path.isdir(os.path.dirname(prediction_dir)):
                # Sicherstellen, dass das Zielverzeichnis existiert
                os.makedirs(prediction_dir, exist_ok=True)
                final_pred_path = os.path.join(prediction_dir, f"final_predictions_{run_id}.csv")
                pd.DataFrame(all_predictions).to_csv(final_pred_path, index=False)
                logging.info(f"Finale Vorhersagen gespeichert unter: {final_pred_path}")
            else:
                logging.warning(f"Pfad 'Prediction_Data' nicht in der Konfiguration für Run '{run_id}' gefunden. Speichern der Vorhersagen übersprungen.")


# =============================================================================
# FLASK WEB APPLICATION
# =============================================================================

def create_flask_app(app_config):
    """Erstellt und konfiguriert die Flask-Anwendung."""
    template_folder = os.path.join(project_root, 'ML_Algorithms', 'templates')
    if not os.path.exists(template_folder):
        template_folder = os.path.join(os.path.dirname(__file__), 'templates')
        
    app = Flask(__name__, template_folder=template_folder)

    @app.route('/')
    def index():
        return render_template('dashboard_retrain.html', config=app_config)

    @app.route('/api/status')
    def get_status():
        with shared_resource_lock:
            return jsonify(PIPELINE_STATE.copy())

    @app.route('/api/data')
    def get_data():
        # Das Frontend fragt nach einem bestimmten Schritt, z.B. /api/data?step=5
        step_index = request.args.get('step', type=int, default=0)
        
        with shared_resource_lock:
            # Prüfen, ob die Daten für den angeforderten Schritt bereits existieren
            if step_index < len(all_predictions):
                # Ja, Daten sind da. Sende sie.
                data_for_step = deepcopy(all_predictions[step_index])
                
                if isinstance(data_for_step['datetime'], (datetime, pd.Timestamp)):
                    data_for_step['datetime'] = data_for_step['datetime'].isoformat()
                
                # Die Logik für die rollierende Prognose muss hier auch ausgeführt werden
                interval_key = "inference_cycle_sec" if PIPELINE_STATE["mode"] == "retraining" else "inference_interval_sec"
                interval_sec = app_config.get(interval_key, 1.0)
                rolling_forecast_values = data_for_step.get("rolling_forecast", [])
                rolling_forecast_dates = [
                    (datetime.fromisoformat(data_for_step['datetime']) + timedelta(seconds=(i) * interval_sec)).isoformat()
                    for i in range(len(rolling_forecast_values))
                ]
                data_for_step['rolling_forecast_dates'] = rolling_forecast_dates
                
                return jsonify({"status": "success", "data": data_for_step})
            else:
                # Nein, der Inferenz-Thread ist noch nicht so weit. Frontend soll warten.
                return jsonify({"status": "waiting"})

    @app.route('/api/control', methods=['POST'])
    def control_pipeline():
        action = request.json.get('action')
        with shared_resource_lock:
            if action == 'pause': PIPELINE_STATE['is_paused'] = True
            elif action == 'resume': PIPELINE_STATE['is_paused'] = False
            logging.info(f"Steuerungsaktion '{action}' empfangen. Pausiert: {PIPELINE_STATE['is_paused']}")
        return jsonify({"status": "ok", "is_paused": PIPELINE_STATE['is_paused']})

    return app

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vereinheitlichte ML-Pipeline mit Web-UI")
    parser.add_argument('--retraining', action=argparse.BooleanOptionalAction, default=False, help="Aktiviert den Retraining-Modus.")
    parser.add_argument('--algorithm', type=str, default='lstm', choices=['random_forest', 'lstm'], help="Zu verwendender Algorithmus.")
    parser.add_argument("--load_id", type=str, help="Optionale Run ID zum Laden von Artefakten anstelle von Training.")
    args = parser.parse_args()

    # Konfiguration basierend auf Algorithmus laden
    if args.algorithm == 'random_forest':
        from config.config_ml_rf import param_rf_test
        config = param_rf_test.copy()
        from ML_Algorithms.Random_Forest.rf_train import RandomForestTrainer
        from ML_Algorithms.Random_Forest.rf_inference import RFInference
        trainer_class = RandomForestTrainer
        inference_class = RFInference
        folder_flag = "RandomForest"
    else: # lstm
        from config.config_ml_lstm import param_lstm_test
        config = param_lstm_test.copy()
        from ML_Algorithms.LSTM.LSTM_train import LSTMTrainer
        from ML_Algorithms.LSTM.LSTM_inference import LSTMInference
        trainer_class = LSTMTrainer
        inference_class = LSTMInference
        folder_flag = "LSTM"
    
    config.update(CONFIG_LOAD_ARTIFACTS)
    config.update(MQTT_CONFIG)
    config['paths'] = CONFIG_PATH['paths']
    if args.load_id: config['load_id'] = args.load_id

    # Modus bestimmen
    mode = "retraining" if args.retraining else "no_retraining"
    port = 5002 if mode == "retraining" else 5001
    
    log_msg = f"--- MODUS: {mode.replace('_', ' ')} | ALGORITHMUS: {args.algorithm} ---"
    if args.load_id:
        log_msg += f" | Lade Modell von Run ID: {args.load_id}"
    logging.info(log_msg)

    # Initialisierungsthread starten (entweder Training oder Laden)
    if args.load_id:
        # Wenn eine ID gegeben ist, wird nicht trainiert.
        PIPELINE_STATE["status"] = "ready_for_inference"
    else:
        # === KORREKTUR HIER ===
        # Der Experiment-Name ist jetzt nur noch der Algorithmus-Name.
        exp_name = folder_flag 
        _, paths = PipelineUtils.setup_experiment(config, exp_name, run_type='train')
        config['paths'] = paths
        threading.Thread(
            target=initial_training, args=(config, trainer_class, folder_flag),
            name="InitialTrainingThread", daemon=True
        ).start()

    # Inferenz-Manager-Thread starten
    threading.Thread(
        target=inference_manager, args=(config, inference_class, folder_flag, args.algorithm, mode),
        name="InferenceManagerThread", daemon=True
    ).start()

    # Flask-App starten
    app = create_flask_app(config)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    local_ip = PipelineUtils.get_local_ip()
    logging.info(f"\n🚀 Webserver startet. Öffnen Sie http://{local_ip}:{port} in Ihrem Browser.")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)