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
import importlib  # NEU: Für dynamische Imports

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
logger = logging.getLogger("pipeline")

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
all_predictions = []  # Speichert alle Vorhersagen für die Web-App


# =============================================================================
# NEUE HILFSFUNKTION
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


def _run_inference_unified(model, input_data: np.ndarray):
    """
    Führt eine Vorhersage sowohl für Keras-Modelle als auch TFLite-Interpreter aus.
    Gibt (prediction_array, inference_time_ms) zurück.
    """
    start = time.perf_counter()
    # TFLite-Interpreter?
    if hasattr(model, "get_input_details") and hasattr(model, "set_tensor") and hasattr(model, "invoke"):
        interpreter = model
        try:
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Form anpassen, wenn nötig
            if tuple(input_details[0]["shape"]) != tuple(input_data.shape):
                interpreter.resize_tensor_input(input_details[0]["index"], input_data.shape, strict=False)
                interpreter.allocate_tensors()
                input_details = interpreter.get_input_details()
                output_details = interpreter.get_output_details()

            interpreter.set_tensor(input_details[0]["index"], input_data.astype(np.float32))
            interpreter.invoke()
            pred = interpreter.get_tensor(output_details[0]["index"])
        except Exception as e:
            logging.error(f"TFLite Inferenz fehlgeschlagen: {e}", exc_info=True)
            raise
    else:
        # Keras / Sklearn (hat .predict)
        try:
            # Keras: verbose=0 unterdrückt "1/1"-Balken
            if hasattr(model, "predict"):
                try:
                    pred = model.predict(input_data, verbose=0)
                except TypeError:
                    pred = model.predict(input_data)
            else:
                # Fallback für sklearn-artige Modelle
                pred = model.predict(input_data)
        except Exception as e:
            logging.error(f"Keras/Sklearn Inferenz fehlgeschlagen: {e}", exc_info=True)
            raise

    dur_ms = (time.perf_counter() - start) * 1000.0
    return np.asarray(pred), dur_ms


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
        model, scaler, features = trainer.run(save_artifacts=True)  # Speichert Artefakte für Nachvollziehbarkeit

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
            with shared_resource_lock:
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
                    current_model.fit(X_retrain, y_retrain, epochs=config.get("retraining_epochs", 5),
                                      batch_size=config.get("batch_size", 32), verbose=0)
                    with shared_resource_lock:
                        shared_model["model"] = current_model
                        logging.info("--- RETRAINING THREAD (LSTM): Modellaustausch erfolgreich! ---")

        elif algorithm == 'random_forest':
            # Vollständiges Neutraining für Random Forest im Speicher.
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


# In: pipeline_web_app.py

def inference_manager(config: dict, inference_class, folder_flag: str, algorithm: str, mode: str):
    """
    Einheitlicher Inferenz-Manager.
    Die Kernlogik für Zyklen, Retraining und Pausen ist jetzt für beide Ladestrategien ('split' und 'live_mqtt') identisch.
    Diese Funktion orchestriert nur noch den Ablauf und delegiert die Inferenzlogik.
    """
    global all_predictions
    retraining_data_list = []
    mqtt_client = None
    inference_processor = None

    try:
        # --- PHASE 1: INITIALISIERUNG ---
        # Warten, bis das initiale Training abgeschlossen ist oder Artefakte geladen werden können
        while PIPELINE_STATE["status"] == "initializing":
            time.sleep(1)
        if PIPELINE_STATE["status"] == "error":
            logging.error("Inferenz-Manager startet nicht, da ein Fehler bei der Initialisierung aufgetreten ist.")
            return

        with shared_resource_lock:
            PIPELINE_STATE["status"] = "inference_running"
        logging.info(f"--- INFERENCE MANAGER: Startet im Modus '{mode}' ---")

        # Inferenz-Prozessor initialisieren
        inference_processor = inference_class(config, folder_flag=folder_flag)

        # Modell laden: Entweder aus dem initialen Training oder von einer Run ID
        model_was_trained_in_memory = shared_model.get("model") is not None
        if model_was_trained_in_memory:
            logging.info("Verwende das neu trainierte Modell aus dem initialen Lauf.")
            inference_processor.set_artifacts_from_memory(shared_model)
        elif config.get('load_id'):
            logging.info(f"Lade Artefakte von Run ID: {config['load_id']}")
            inference_processor.load_artifacts()
        else:
            raise RuntimeError("Kein trainiertes Modell und keine load_id zum Laden gefunden.")

        # --- PHASE 2: DATENQUELLE & SCHLEIFENPARAMETER ---
        loading_strategy = config.get("loading_strategy", "live_mqtt")
        data_source_iterator = inference_processor.get_data_source_iterator()

        if mode == "retraining":
            max_cycles = config.get("retraining_cycles", 2)
            steps_per_cycle = config.get("retraining_interval_steps", 200)
        else:
            max_cycles = 1
            steps_per_cycle = config.get("inference_steps", 500)
        
        # Falls es eine fixe Datenquelle ist, begrenzen wir die Schritte
        if hasattr(data_source_iterator, '__len__'):
             steps_per_cycle = min(steps_per_cycle, len(data_source_iterator))

        target_interval_sec = config.get("inference_interval_sec", 1.0)
        with shared_resource_lock:
            PIPELINE_STATE.update({"total_cycles": max_cycles, "total_steps": steps_per_cycle, "mode": mode})

        # --- PHASE 3: VEREINHEITLICHTE HAUPTSCHLEIFE ---
        for cycle in range(max_cycles):
            with shared_resource_lock:
                PIPELINE_STATE.update({"cycle_count": cycle + 1,
                                       "retraining_status": "collecting" if mode == "retraining" else "idle"})
            logging.info(f"--- Zyklus {cycle + 1}/{max_cycles}: Starte Inferenz. ---")

            for step, payload in enumerate(data_source_iterator(steps_per_cycle)):
                start_cycle_time = time.perf_counter()
                
                # Pausen- und Beenden-Logik
                while PIPELINE_STATE["is_paused"]:
                    time.sleep(0.5)
                if PIPELINE_STATE["is_finished"]:
                    break

                with shared_resource_lock:
                    PIPELINE_STATE["steps_in_cycle"] = step + 1
                
                # Inferenz-Schritt ausführen und Ergebnis erhalten
                prediction_entry = inference_processor.process_step(payload)

                if prediction_entry:
                    # Ergebnisse für die UI und das Retraining sammeln
                    with shared_resource_lock:
                        all_predictions.append(prediction_entry)
                    if mode == "retraining":
                        retraining_data_list.append(payload)
                
                # 1-Hz-Takt stabilisieren
                elapsed = time.perf_counter() - start_cycle_time
                sleep_duration = target_interval_sec - elapsed
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

            if PIPELINE_STATE["is_finished"]:
                break

            # --- PHASE 4: RETRAINING-HANDLING ---
            if mode == "retraining" and cycle < max_cycles - 1:
                logging.info(f"--- Zyklus {cycle + 1}: Datensammlung abgeschlossen. Starte Nachtraining. ---")
                retraining_df = pd.DataFrame(retraining_data_list).set_index('datetime')
                
                retraining_thread = threading.Thread(
                    target=retraining_thread_task, args=(retraining_df.copy(), algorithm),
                    name=f"RetrainingThread-{cycle+1}"
                )
                retraining_thread.start()
                retraining_data_list.clear()

    except Exception as e:
        logging.error(f"Schwerwiegender Fehler im Inference Manager: {e}", exc_info=True)
        with shared_resource_lock:
            PIPELINE_STATE.update({"status": "error", "error_message": str(e)})

    finally:
        # --- PHASE 5: AUFRÄUMEN UND SPEICHERN ---
        logging.info("--- Pipeline beendet. Aufräumen und Speichern. ---")
        with shared_resource_lock:
            PIPELINE_STATE.update({"status": "finished", "is_finished": True})

        if hasattr(inference_processor, 'stop'):
            inference_processor.stop() # Stoppt z.B. den internen MQTT-Client

        if all_predictions:
            try:
                logging.info("Speichere finale Vorhersagen...")
                # Die komplexe Speicherlogik wird an den Prozessor oder ein Utility delegiert
                inference_processor.save_final_results(all_predictions)
            except Exception as e:
                logging.error(f"Fehler beim finalen Speichern der Ergebnisse: {e}", exc_info=True)

# =============================================================================
# HAUPTLOGIK
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vereinheitlichte ML-Pipeline mit Web-UI")

    # --- Argumente definieren ---
    parser.add_argument('--algorithm', type=str, required=True, choices=['random_forest', 'lstm'],
                        help="Zu verwendender Algorithmus.")
    # --config-name ist jetzt optional (required=True wurde entfernt)
    parser.add_argument('--config-name', type=str,
                        help="Optional: Name der Konfigurationsvariable. Default wird zu 'param_<algorithm>_test'.")

    parser.add_argument('--retraining', action=argparse.BooleanOptionalAction, default=False,
                        help="Aktiviert den Retraining-Modus.")
    parser.add_argument("--load_id", type=str,
                        help="Optionale Run ID zum Laden von Artefakten anstelle von Training.")
    parser.add_argument("--model_filename", type=str,
                        help="Optional: Name der zu ladenden Modelldatei (z.B. 'quantized_model.tflite').")

    args = parser.parse_args()

    # --- Default-Wert für config-name setzen, falls nicht angegeben ---
    if args.config_name is None:
        args.config_name = f"{args.algorithm}"
        logging.info(f"Kein --config-name angegeben. Verwende Default: '{args.config_name}'")

    # --- Dynamisches Laden der Konfiguration ---
    config = load_config_dynamically(args.algorithm, args.config_name)

    # --- Zuordnung der Klassen basierend auf dem Algorithmus ---
    if args.algorithm == 'random_forest':
        from ML_Algorithms.Random_Forest.rf_train import RandomForestTrainer
        from ML_Algorithms.Random_Forest.rf_inference import RFInference
        trainer_class = RandomForestTrainer
        inference_class = RFInference
        folder_flag = "RandomForest"
    else:  # lstm
        from ML_Algorithms.LSTM.LSTM_train import LSTMTrainer
        from ML_Algorithms.LSTM.LSTM_inference import LSTMInference
        trainer_class = LSTMTrainer
        inference_class = LSTMInference
        folder_flag = "LSTM"

    # --- Zusammenführen der restlichen Konfigurationen ---
    config.update(CONFIG_LOAD_ARTIFACTS)
    config.update(MQTT_CONFIG)
    config['paths'] = CONFIG_PATH['paths']

    # --- Restliche Logik ---
    mode = "retraining" if args.retraining else "no_retraining"
    port = 5002 if mode == "retraining" else 5001

    log_msg = f"--- MODUS: {mode.replace('_', ' ')} | ALGORITHMUS: {args.algorithm} | CONFIG: {args.config_name} ---"

    if args.model_filename:
        config['model_filename'] = args.model_filename

    if args.load_id:
        config['load_id'] = args.load_id
        config['run_id'] = args.load_id
        log_msg += f" | Lade Modell von Run ID: {args.load_id}"

        logging.info(f"Konfiguriere Pfade für das Laden von Run ID: {args.load_id}...")

        base_output_path = config['paths'].get('output')
        run_dir = os.path.join(base_output_path, folder_flag, args.load_id)

        config['paths'].update({
            "run_dir": run_dir,
            "Models": os.path.join(run_dir, "Models"),
            "Scalers": os.path.join(run_dir, "Scalers"),
            "Prediction_Data": os.path.join(run_dir, "Prediction_Data"),
            "Error_Metrics": os.path.join(run_dir, "Error_Metrics")
        })
        logging.info(f"Speicherpfad für Vorhersagen gesetzt auf: {config['paths']['Prediction_Data']}")

    else:
        exp_name = folder_flag
        _, paths = PipelineUtils.setup_experiment(config, exp_name, run_type='train')
        config['paths'] = paths

    logging.info(log_msg)

    # Initialisierungsthread starten (nur wenn NICHT geladen wird)
    if not args.load_id:
        threading.Thread(
            target=initial_training, args=(config, trainer_class, folder_flag),
            name="InitialTrainingThread", daemon=True
        ).start()
    else:
        PIPELINE_STATE["status"] = "ready_for_inference"

    # Inferenz-Manager-Thread starten
    threading.Thread(
        target=inference_manager, args=(config, inference_class, folder_flag, args.algorithm, mode),
        name="InferenceManagerThread", daemon=True
    ).start()

    # Flask-App starten
    from web_app import create_app
    app = create_app(config, PIPELINE_STATE, all_predictions, shared_resource_lock)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    local_ip = PipelineUtils.get_local_ip()
    logging.info(f"\n🚀 Webserver startet. Öffnen Sie http://{local_ip}:{port} in Ihrem Browser.")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
