# pipeline_web_app.py
#
# BEISPIEL-AUFRUFE NEU:
# -----------------
# > python pipeline_web_app.py --algorithm lstm --config-name param_lstm_test --retraining
# > python pipeline_web_app.py --algorithm random_forest --config-name param_rf_test --no-retraining
# > python pipeline_web_app.py --algorithm lstm --config-name param_lstm_production --no-retraining --load_id <IHRE_RUN_ID>
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
import importlib 

# Benötigte Imports für das In-Memory-Retraining
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler

# --- Systempfad-Setup ---
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
# HILFSFUNKTION
# =============================================================================
def load_config_dynamically(algorithm: str, config_name: str) -> dict:
    """
    Lädt dynamisch eine Konfigurationsvariable aus einem Modul.
    """
    try:
        module_path = f"config.config_ml_{algorithm}"
        config_module = importlib.import_module(module_path)
        config_dict = getattr(config_module, config_name)
        logging.info(f"Konfiguration '{config_name}' erfolgreich aus '{module_path}' geladen.")
        return deepcopy(config_dict)
    except (ImportError, AttributeError) as e:
        logging.error(f"Fehler beim dynamischen Laden der Konfiguration '{config_name}' aus '{module_path}': {e}", exc_info=True)
        sys.exit(1)


# =============================================================================
# CORE LOGIC: TRAINING & RETRAINING
# =============================================================================

def initial_training(config: dict, trainer_class, folder_flag: str):
    """
    Führt das initiale Training durch und speichert die Artefakte im Speicher.
    """
    global shared_model
    logging.info(f"--- PHASE 1: Initiales Training für {folder_flag} startet ---")
    try:
        trainer = trainer_class(config=config, folder_flag=folder_flag)
        # HINWEIS: Die `run`-Methode ruft intern die Datenpipeline auf.
        # Wir speichern hier eine Kopie der initialen Trainingsdaten für das RF-Nachtraining.
        data_pipeline_for_retraining = trainer._setup_pipeline()
        initial_data_df, _ = data_pipeline_for_retraining._load_data(mode='train')
        
        model, scaler, features = trainer.run(save_artifacts=True)

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
    """
    global shared_model
    logging.info(f"--- RETRAINING THREAD ({algorithm}): Startet Nachtraining. ---")
    with shared_resource_lock:
        PIPELINE_STATE["retraining_status"] = "training"

    try:
        with shared_resource_lock:
            config = deepcopy(shared_model['config'])
            initial_data = shared_model['initial_training_data']

        if algorithm == 'lstm':
            # Inkrementelles Training für LSTM
            with shared_resource_lock:
                current_model = tf.keras.models.clone_model(shared_model['model'])
                current_model.set_weights(shared_model['model'].get_weights())

            retraining_data_featured, _ = fe.add_all_features(retraining_data_df, config)
            retraining_data_featured.dropna(inplace=True)

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
            # Vollständiges Neutraining für Random Forest
            logging.info("Kombiniere alte und neue Daten für RF-Nachtraining...")
            combined_data = pd.concat([initial_data, retraining_data_df]).drop_duplicates().sort_index()

            combined_df_featured, features_dict = fe.add_all_features(combined_data, config)
            combined_df_featured.dropna(inplace=True)
            
            target_column = config["base_features"][0]
            x_features = [col for col in features_dict["all"] if col != target_column]
            
            X_retrain_df = combined_df_featured[x_features]
            y_retrain = combined_df_featured[target_column]

            logging.info("Passe Scaler neu an und trainiere neues RF-Modell...")
            new_scaler = (RobustScaler if config.get("scaler_type") == "robust" else MinMaxScaler)()
            X_retrain_scaled = new_scaler.fit_transform(X_retrain_df)

            new_model = RandomForestRegressor(**config.get("model_params", {}))
            new_model.fit(X_retrain_scaled, y_retrain.values)
            
            with shared_resource_lock:
                shared_model.update({
                    "model": new_model, "scaler": new_scaler, "features": x_features
                })
                logging.info("--- RETRAINING THREAD (RF): Modellaustausch erfolgreich! ---")

    except Exception as e:
        logging.error(f"RETRAINING THREAD: Fehler: {e}", exc_info=True)
    finally:
        with shared_resource_lock:
            PIPELINE_STATE["retraining_status"] = "idle"


# =============================================================================
# INFERENCE MANAGER
# =============================================================================
def inference_manager(config: dict, inference_class, folder_flag: str, algorithm: str, mode: str):
    """
    Einheitlicher Inferenz-Manager mit korrekter, modell-spezifischer Logik.
    """
    global all_predictions
    retraining_data_list = []
    mqtt_client = None
    inference_processor = None

    try:
        while PIPELINE_STATE["status"] == "initializing":
            time.sleep(1)
        if PIPELINE_STATE["status"] == "error":
            logging.error("Inferenz-Manager startet nicht wegen Initialisierungsfehler.")
            return

        with shared_resource_lock:
            PIPELINE_STATE["status"] = "inference_running"
        logging.info(f"--- INFERENCE MANAGER: Startet im Modus '{mode}' für Algorithmus '{algorithm}' ---")
        
        inference_processor = inference_class(config, "", 0, "", folder_flag)
        
        if shared_model.get("model") is not None:
            logging.info("Verwende das neu trainierte Modell aus dem initialen Lauf.")
            with shared_resource_lock:
                inference_processor.model = shared_model["model"]
                inference_processor.scaler = shared_model["scaler"]
                inference_processor.feature_list = shared_model["features"]
                inference_processor.training_config = shared_model["config"]
                inference_processor.target_feature = shared_model["config"]["base_features"][0]
        elif config.get('load_id'):
            logging.info(f"Lade Artefakte von Run ID: {config['load_id']}")
            inference_processor.load_artifacts()
        else:
            logging.error("Kein trainiertes Modell und keine load_id zum Laden gefunden.")
            with shared_resource_lock: PIPELINE_STATE.update({"status": "error", "error_message": "Kein Modell verfügbar."})
            return

        loading_strategy = config.get("loading_strategy", "live_mqtt")
        data_source = []
        
        if loading_strategy == "split":
            test_df = LoadPrepareData.load_test_data_by_fraction(config, make_date_as_index=False)
            if not test_df.empty: data_source = test_df.to_dict("records")
        else: # live_mqtt
            mqtt_client = MqttInferenceClient(
                broker_ip=config['MQTT_BROKER_IP'], port=config['MQTT_PORT'], topic=config['MQTT_TOPIC'],
                on_message_callback=inference_processor.update_latest_data
            )
            mqtt_client.run()

        max_cycles = config.get("retraining_cycles", 2) if mode == "retraining" else 1
        steps_per_cycle = config.get("retraining_interval_steps", 200) if mode == "retraining" else (len(data_source) if loading_strategy == "split" else config.get("inference_steps", 500))
        target_interval_sec = config.get("inference_interval_sec", 1.0)
        with shared_resource_lock:
            PIPELINE_STATE.update({"total_cycles": max_cycles, "total_steps": steps_per_cycle, "mode": mode})

        current_data_index = 0
        for cycle in range(max_cycles):
            with shared_resource_lock: PIPELINE_STATE.update({"cycle_count": cycle + 1, "retraining_status": "collecting" if mode == "retraining" else "idle"})
            logging.info(f"--- Zyklus {cycle + 1}/{max_cycles}: Starte Inferenz. ---")

            for step in range(steps_per_cycle):
                start_cycle_time = time.perf_counter()
                while PIPELINE_STATE["is_paused"]: time.sleep(0.5)
                if PIPELINE_STATE["is_finished"]: break

                if loading_strategy == "split":
                    if current_data_index < len(data_source):
                        inference_processor.latest_payload = data_source[current_data_index]
                        current_data_index += 1
                    else: break
                
                with shared_resource_lock: PIPELINE_STATE["steps_in_cycle"] = step + 1
                if inference_processor.latest_payload is None:
                    if loading_strategy == "live_mqtt": time.sleep(target_interval_sec)
                    continue

                with shared_resource_lock:
                    if shared_model.get("model") is not None:
                        inference_processor.model = shared_model["model"]
                        inference_processor.scaler = shared_model["scaler"]
                        inference_processor.feature_list = shared_model["features"]
                
                input_data, timestamp, true_value = inference_processor._prepare_input_data()
                
                if input_data is not None:
                    prediction_raw, model_inference_time_ms = PipelineUtils.run_timed_inference(model=inference_processor.model, input_data=input_data)
                    
                    if algorithm == 'lstm':
                        try:
                            target_index = inference_processor.feature_list.index(inference_processor.target_feature)
                            predictions_unscaled_all = PipelineUtils.safe_inverse_transform(
                                scaler=inference_processor.scaler, 
                                array=prediction_raw.reshape(1, -1), 
                                target_index=target_index
                            ).flatten()
                        except ValueError:
                            logging.error(f"FATAL: Zielvariable '{inference_processor.target_feature}' nicht in LSTM-Feature-Liste gefunden.")
                            predictions_unscaled_all = np.array([0])
                    else: # algorithm == 'random_forest'
                        predictions_unscaled_all = prediction_raw.flatten()
                    
                    # <<< NEUE ZEILE FÜR KONSOLEN-AUSGABE >>>
                    true_value_str = f"{true_value:.4f}" if isinstance(true_value, (int, float)) else "N/A"
                    logging.info(f">>> VORHERSAGE: {predictions_unscaled_all[0]:.4f} (Wahrer Wert: {true_value_str})")
                    # <<< ENDE NEUE ZEILE >>>

                    total_processing_time_ms = (time.perf_counter() - start_cycle_time) * 1000
                    prediction_entry = {
                        "datetime": timestamp, "prediction": predictions_unscaled_all[0], "true_value": true_value,
                        "rolling_forecast": predictions_unscaled_all.tolist(), "cpu_load": PipelineUtils.get_cpu_usage(),
                        "ram_usage": PipelineUtils.get_memory_usage(), "model_inference_time_ms": model_inference_time_ms,
                        "total_processing_time_ms": total_processing_time_ms
                    }
                    with shared_resource_lock:
                        all_predictions.append(prediction_entry)
                    
                    if mode == "retraining":
                        retraining_data_list.append(inference_processor.latest_payload)
                
                inference_processor.latest_payload = None
                sleep_duration = target_interval_sec - (time.perf_counter() - start_cycle_time)
                if sleep_duration > 0: time.sleep(sleep_duration)

            if PIPELINE_STATE["is_finished"]: break

            if mode == "retraining" and cycle < max_cycles - 1:
                logging.info(f"--- Zyklus {cycle + 1}: Starte Nachtraining. ---")
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
        logging.info("--- Pipeline beendet. Aufräumen und Speichern. ---")
        with shared_resource_lock: PIPELINE_STATE.update({"status": "finished", "is_finished": True})
        
        if mqtt_client:
            mqtt_client.stop()
        
        if all_predictions:
            logging.info("🔄 Speichere finale Vorhersagen...")
            df = pd.DataFrame(all_predictions)
            if "ram_usage" in df.columns and not df["ram_usage"].dropna().empty:
                try:
                    ram_df = pd.json_normalize(df["ram_usage"].dropna())
                    ram_df.columns = [f"ram_{col}" for col in ram_df.columns]
                    df = df.drop(columns=["ram_usage"]).join(ram_df)
                except Exception as e:
                    logging.warning(f"Konnte RAM-Daten nicht entpacken: {e}")
            if "rolling_forecast" in df.columns and "true_value" in df.columns:
                try:
                    final_config, final_paths = PipelineUtils.setup_experiment(config, folder_flag, run_type='inference')
                    df_valid = df.dropna(subset=['rolling_forecast', 'true_value']).copy()
                    if not df_valid.empty:
                        y_pred = np.stack(df_valid["rolling_forecast"].to_numpy())
                        y_true = np.stack(df_valid["true_value"].to_numpy())
                        horizon = y_pred.shape[1] if y_pred.ndim > 1 else 1
                        if y_true.ndim == 1 and horizon > 1: y_true = np.tile(y_true.reshape(-1, 1), reps=(1, horizon))
                        
                        pred_path = os.path.join(final_paths["Prediction_Data"], f"prediction_final_{final_config.get('run_id')}.csv")
                        
                        metrics = PipelineUtils.evaluate_all_metrics(y_true, y_pred, horizon=horizon)
                        PipelineUtils.save_metrics_summary(metrics, final_config, (inference_processor.training_config if hasattr(inference_processor, 'training_config') else {}), final_paths)
                        logging.info(f"✅ Vorhersagen und Metriken gespeichert.")
                except Exception as e:
                    logging.error(f"Fehler beim Speichern der erweiterten Vorhersagen: {e}", exc_info=True)


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
    def index(): return render_template('dashboard_retrain.html', config=app_config)

    @app.route('/api/status')
    def get_status():
        with shared_resource_lock: return jsonify(PIPELINE_STATE.copy())

    @app.route('/api/data')
    def get_data():
        step_index = request.args.get('step', type=int, default=0)
        with shared_resource_lock:
            if step_index < len(all_predictions):
                data_for_step = deepcopy(all_predictions[step_index])
                if isinstance(data_for_step['datetime'], (datetime, pd.Timestamp)):
                    data_for_step['datetime'] = data_for_step['datetime'].isoformat()
                
                interval_sec = app_config.get("inference_interval_sec", 1.0)
                rolling_forecast_values = data_for_step.get("rolling_forecast", [])
                base_time = datetime.fromisoformat(data_for_step['datetime'])
                data_for_step['rolling_forecast_dates'] = [
                    (base_time + timedelta(seconds=(i) * interval_sec)).isoformat()
                    for i in range(len(rolling_forecast_values))
                ]
                return jsonify({"status": "success", "data": data_for_step})
            else:
                return jsonify({"status": "waiting"})

    @app.route('/api/control', methods=['POST'])
    def control_pipeline():
        action = request.json.get('action')
        with shared_resource_lock:
            if action == 'pause': PIPELINE_STATE['is_paused'] = True
            elif action == 'resume': PIPELINE_STATE['is_paused'] = False
        return jsonify({"status": "ok", "is_paused": PIPELINE_STATE['is_paused']})

    return app


# =============================================================================
# HAUPTLOGIK
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vereinheitlichte ML-Pipeline mit Web-UI")
    parser.add_argument('--algorithm', type=str, required=True, choices=['random_forest', 'lstm'], help="Zu verwendender Algorithmus.")
    parser.add_argument('--config-name', type=str, required=True, help="Name der Konfigurationsvariable.")
    parser.add_argument('--retraining', action=argparse.BooleanOptionalAction, default=False, help="Aktiviert den Retraining-Modus.")
    parser.add_argument("--load_id", type=str, help="Optionale Run ID zum Laden von Artefakten.")
    parser.add_argument("--model_filename", type=str, help="Optional: Name der zu ladenden Modelldatei.")
    args = parser.parse_args()

    config = load_config_dynamically(args.algorithm, args.config_name)

    if args.algorithm == 'random_forest':
        from ML_Algorithms.Random_Forest.rf_train import RandomForestTrainer
        from ML_Algorithms.Random_Forest.rf_inference import RFInference
        trainer_class, inference_class, folder_flag = RandomForestTrainer, RFInference, "RandomForest"
    else: # lstm
        from ML_Algorithms.LSTM.LSTM_train import LSTMTrainer
        from ML_Algorithms.LSTM.LSTM_inference import LSTMInference
        trainer_class, inference_class, folder_flag = LSTMTrainer, LSTMInference, "LSTM"
    
    config.update(CONFIG_LOAD_ARTIFACTS); config.update(MQTT_CONFIG)
    config['paths'] = CONFIG_PATH['paths']; config['algorithm'] = args.algorithm

    mode = "retraining" if args.retraining else "no_retraining"
    port = 5002 if mode == "retraining" else 5001
    
    if args.model_filename: config['model_filename'] = args.model_filename
    if args.load_id:
        config['load_id'] = args.load_id
        config['inference_mode'] = 'load_artifacts_path'
    else:
        final_config, paths = PipelineUtils.setup_experiment(config, folder_flag, run_type='train')
        config.update(final_config); config['paths'] = paths

    if not args.load_id:
        threading.Thread(target=initial_training, args=(config, trainer_class, folder_flag), name="InitialTrainingThread", daemon=True).start()
    else:
        with shared_resource_lock: PIPELINE_STATE["status"] = "ready_for_inference"

    threading.Thread(target=inference_manager, args=(config, inference_class, folder_flag, args.algorithm, mode), name="InferenceManagerThread", daemon=True).start()

    app = create_flask_app(config)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    local_ip = PipelineUtils.get_local_ip()
    logging.info(f"\n🚀 Webserver startet. Öffnen Sie http://{local_ip}:{port} in Ihrem Browser.")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)