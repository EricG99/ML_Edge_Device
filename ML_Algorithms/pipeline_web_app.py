# pipeline_web_app.py

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

import webbrowser # NEU
from threading import Timer # NEU
import importlib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler

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
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pipeline")

PIPELINE_STATE = {
    "status": "initializing", "error_message": None, "retraining_status": "idle",
    "cycle_count": 0, "steps_in_cycle": 0, "total_steps": 0, "total_cycles": 0,
    "is_paused": False, "is_finished": False, "mode": "unknown"
}
shared_resource_lock = threading.Lock()
shared_model = {"model": None, "scaler": None, "features": None, "config": None, "initial_training_data": None}
all_predictions = []

def load_config_dynamically(algorithm: str, config_name: str) -> dict:
    try:
        module_path = f"config.config_ml_{algorithm}"
        config_module = importlib.import_module(module_path)
        config_dict = getattr(config_module, config_name)
        logging.info(f"Konfiguration '{config_name}' erfolgreich aus '{module_path}' geladen.")
        return deepcopy(config_dict)
    except (ImportError, AttributeError) as e:
        logging.error(f"Fehler beim dynamischen Laden der Konfiguration '{config_name}' aus '{module_path}': {e}", exc_info=True)
        sys.exit(1)

def initial_training(config: dict, trainer_class, folder_flag: str):
    global shared_model
    logging.info(f"--- PHASE 1: Initiales Training für {folder_flag} startet ---")
    try:
        trainer = trainer_class(config=config, folder_flag=folder_flag)
        pipeline = trainer._setup_pipeline()
        initial_data_df, _ = pipeline._load_data(mode='train')
        model, scaler, y_scaler, features = trainer.run(save_artifacts=True)

        with shared_resource_lock:
            shared_model.update({
                "model": model, "scaler": scaler, "y_scaler": y_scaler, "features": features,
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
    Führt das Nachtraining im Hintergrund durch und bereitet Artefakte für den Hot-Swap vor.
    WICHTIG: Setzt den Pipeline-Status am Ende auf 'ready_to_swap', damit der Inferenz-Loop
    selbstständig die Artefakte übernimmt (kein Blockieren!).
    """
    global shared_model
    logging.info(f"--- RETRAINING THREAD ({algorithm}): Startet Nachtraining. ---")
    with shared_resource_lock:
        PIPELINE_STATE["retraining_status"] = "training"

    try:
        # Bestehende Artefakte & Konfiguration auslesen (thread-safe Kopie)
        with shared_resource_lock:
            config = deepcopy(shared_model['config'])
            initial_data = shared_model['initial_training_data']
            current_model_ref = shared_model['model']
            scaler_ref = shared_model['scaler']
            y_scaler_ref = shared_model.get('y_scaler')
            features_ref = shared_model['features']

        if algorithm == 'lstm':
            # LSTM: inkrementelles Fitten auf neuen Daten
            if hasattr(current_model_ref, 'clone_model'):
                current_model = tf.keras.models.clone_model(current_model_ref)
                current_model.set_weights(current_model_ref.get_weights())
            else:
                current_model = deepcopy(current_model_ref)

            logging.info("LSTM-Nachtraining (inkrementell) wird durchgeführt...")
            retraining_data_featured, _ = fe.add_all_features(retraining_data_df, config)
            retraining_data_featured = retraining_data_featured.loc[:, features_ref].dropna().copy()

            if not retraining_data_featured.empty:
                scaled_data = scaler_ref.transform(retraining_data_featured)
                X_retrain, y_retrain = LoadPrepareData.convert_data_to_sliding_window(
                    scaled_data,
                    lag_horizon=config["lags"],
                    forecast_horizon=config["horizon"]
                )
                if len(X_retrain) > 0:
                    current_model.compile(
                        optimizer=config.get("optimizer", "adam"),
                        loss=config.get("loss", "mse")
                    )
                    current_model.fit(
                        X_retrain, y_retrain,
                        epochs=config.get("retraining_epochs", 5),
                        batch_size=config.get("batch_size", 32),
                        verbose=0
                    )
                    with shared_resource_lock:
                        # Modell ersetzen (Scaler/Features bleiben)
                        shared_model["model"] = current_model
                        # Signal an Inferenz-Loop: Hot-Swap durchführen
                        PIPELINE_STATE["retraining_status"] = "ready_to_swap"
                        logging.info("--- RETRAINING THREAD (LSTM): Artefakte bereit zum Hot-Swap ---")

        elif algorithm == 'random_forest':
            # RF: Neu-Training auf kombinierten Daten + neuen Features/Scaler
            logging.info("Kombiniere alte und neue Daten für RF-Nachtraining...")
            combined_data = pd.concat([initial_data, retraining_data_df]).drop_duplicates().sort_index()

            logging.info("Feature Engineering für kombinierten Datensatz...")
            combined_df_featured, features_dict = fe.add_all_features(combined_data, config)
            new_features = features_dict["all"]
            combined_df_featured.dropna(inplace=True)

            target_col_name = config["base_features"][0].lower()
            X_retrain = combined_df_featured[new_features]
            y_retrain = combined_df_featured[target_col_name]

            logging.info("Passe Scaler neu an...")
            scaler_class = RobustScaler if config.get("scaler_type", "minmax") == "robust" else MinMaxScaler
            new_scaler = scaler_class()
            X_retrain_scaled = new_scaler.fit_transform(X_retrain)

            logging.info("Trainiere neues Random Forest Modell...")
            new_model = RandomForestRegressor(**config.get("model_params", {}))
            new_model.fit(X_retrain_scaled, y_retrain.values)

            with shared_resource_lock:
                shared_model.update({
                    "model": new_model,
                    "scaler": new_scaler,
                    "features": new_features,
                    "initial_training_data": combined_data
                })
                # Signal an Inferenz-Loop: Hot-Swap durchführen
                PIPELINE_STATE["retraining_status"] = "ready_to_swap"
                logging.info("--- RETRAINING THREAD (RF): Artefakte bereit zum Hot-Swap ---")

        else:
            raise ValueError(f"Unbekannter Algorithmus für Retraining: {algorithm}")

    except Exception as e:
        logging.error(f"RETRAINING THREAD: Fehler: {e}", exc_info=True)
        # Fehlerfall: Status zurücksetzen, damit ein späterer Versuch möglich bleibt
        with shared_resource_lock:
            PIPELINE_STATE["retraining_status"] = "idle"


def inference_manager(config: dict, inference_class, folder_flag: str, algorithm: str, mode: str):
    """
    Führt die Inferenz in Zyklen/Schritten aus.
    - Speichert pro Schritt: True, Forecast(H), inference_time_s (aus prediction_entry), total_time_s, CPU%, RAM.
    - Sammelt pro Schritt Retraining-Material und triggert am Zyklusende ein non-blocking Retraining.
    - Hot-Swap, sobald neues Modell/Abhängigkeiten bereitstehen.
    """
    import os
    import time
    import threading
    import logging
    import pandas as pd

    # psutil optional verwenden (CPU/RAM)
    try:
        import psutil
    except Exception:
        psutil = None

    # Globale Zustände verwenden (werden an anderer Stelle definiert)
    global all_predictions, shared_model, shared_resource_lock, PIPELINE_STATE, retraining_thread_task

    retraining_data_list = []
    inference_processor = None

    # Prozess & CPU-Kerne für Prozentberechnung
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            n_cpus = max(psutil.cpu_count(logical=True) or 1, 1)
        except Exception:
            proc, n_cpus = None, 1
    else:
        proc, n_cpus = None, 1

    try:
        # Auf Initialisierung warten
        while PIPELINE_STATE.get("status") == "initializing":
            time.sleep(0.2)
        if PIPELINE_STATE.get("status") == "error":
            logging.error("Inferenz-Manager startet nicht, da ein Fehler bei der Initialisierung aufgetreten ist.")
            return

        # Status setzen
        with shared_resource_lock:
            PIPELINE_STATE["status"] = "inference_running"
        logging.info(f"--- INFERENCE MANAGER: Startet im Modus '{mode}' ---")

        # Inferenz-Objekt aufsetzen + Artefakte laden/übernehmen
        inference_processor = inference_class(config, folder_flag=folder_flag)
        if shared_model.get("model") is not None:
            inference_processor.set_artifacts_from_memory(shared_model)
        elif config.get('load_id'):
            inference_processor.load_artifacts()
        else:
            raise RuntimeError("Kein trainiertes Modell und keine load_id zum Laden gefunden.")

        # Datenquelle (zustandsbehafteter Iterator-Fabrik)
        data_source_iterator = inference_processor.get_data_source_iterator()

        # Zyklen & Schritte pro Zyklus
        if mode == "retraining":
            max_cycles = int(config.get("retraining_cycles", 5))
            steps_per_cycle = int(config.get("retraining_interval_steps", 20))
        else:
            max_cycles = 1
            steps_per_cycle = config.get("inference_steps", "infinite")
            if steps_per_cycle == "infinite":
                if hasattr(inference_processor, "_batch_data_df") and inference_processor._batch_data_df is not None:
                    steps_per_cycle = len(inference_processor._batch_data_df)
                else:
                    # Fallback: sichere Obergrenze
                    steps_per_cycle = int(config.get("fallback_steps", 1000))

        target_interval_sec = float(config.get("inference_interval_sec", 1.0))

        with shared_resource_lock:
            PIPELINE_STATE.update({
                "total_cycles": max_cycles,
                "total_steps": steps_per_cycle,
                "mode": mode
            })

        # --- Hauptschleife über Zyklen ---
        for cycle in range(max_cycles):
            with shared_resource_lock:
                PIPELINE_STATE.update({
                    "cycle_count": cycle + 1,
                    "retraining_status": "collecting" if mode == "retraining" else "idle"
                })
            logging.info(f"--- Zyklus {cycle + 1}/{max_cycles}: Starte Inferenz. ---")

            current_cycle_iterator = data_source_iterator(steps_per_cycle)

            # --- Schritt-Schleife ---
            for step, payload in enumerate(current_cycle_iterator):
                step_wall_t0 = time.perf_counter()

                # Ressourcen am Anfang messen
                if proc is not None:
                    try:
                        cpu_t0 = proc.cpu_times()
                        ram_mb = proc.memory_info().rss / (1024 * 1024)
                    except Exception:
                        cpu_t0, ram_mb = None, None
                else:
                    cpu_t0, ram_mb = None, None

                # Pause/Stop behandeln
                while PIPELINE_STATE.get("is_paused"):
                    time.sleep(0.2)
                if PIPELINE_STATE.get("is_finished"):
                    break
                with shared_resource_lock:
                    PIPELINE_STATE["steps_in_cycle"] = step + 1

                # Einen Schritt verarbeiten
                prediction_entry = inference_processor.process_step(payload)

                if prediction_entry:
                    # Gesamtzeit (Wall)
                    step_total_time_s = time.perf_counter() - step_wall_t0

                    # CPU% aus Prozess-CPU-Zeit relativ zu n_cpus
                    cpu_percent = None
                    if proc is not None and cpu_t0 is not None:
                        try:
                            cpu_t1 = proc.cpu_times()
                            cpu_used = (cpu_t1.user + cpu_t1.system) - (cpu_t0.user + cpu_t0.system)
                            cpu_percent = (cpu_used / max(step_total_time_s, 1e-12)) / n_cpus * 100.0
                        except Exception:
                            cpu_percent = None

                    # Persistieren dieses Schritts
                    try:
                        inference_processor.save_step_result(
                            prediction_entry=prediction_entry,
                            total_time_s=step_total_time_s,
                            cpu_percent=cpu_percent,
                            ram_mb=ram_mb
                        )
                    except Exception as persist_err:
                        logging.error(f"Fehler beim Speichern des Inferenz-Schritts: {persist_err}", exc_info=True)

                    # In-Memory sammeln (für finale Metriken)
                    with shared_resource_lock:
                        all_predictions.append(prediction_entry)

                    # --- NEU: Retraining-Material sammeln (nur im Modus 'retraining') ---
                    if mode == "retraining":
                        try:
                            if hasattr(payload, "to_dict"):
                                pl = payload.to_dict()
                            else:
                                pl = dict(payload)
                        except Exception:
                            pl = {}

                        # datetime sicherstellen
                        if "datetime" not in pl and "datetime" in prediction_entry:
                            pl["datetime"] = prediction_entry["datetime"]

                        retraining_data_list.append(pl)

                # Hot-Swap, wenn Retraining fertig
                with shared_resource_lock:
                    ready = PIPELINE_STATE.get("retraining_status") == "ready_to_swap"
                if ready:
                    inference_processor.set_artifacts_from_memory(shared_model)
                    with shared_resource_lock:
                        PIPELINE_STATE["retraining_status"] = "idle"
                    logging.info("--- INFERENCE MANAGER: Neues Modell & Abhängigkeiten aktiv (Hot-Swap) ---")

                # Zyklus-Takt einhalten
                elapsed = time.perf_counter() - step_wall_t0
                sleep_dur = target_interval_sec - elapsed
                if sleep_dur > 0:
                    time.sleep(sleep_dur)

            # vorzeitig beendet?
            if PIPELINE_STATE.get("is_finished"):
                break

            # --- Am Zyklusende: Retraining non-blocking starten (nur wenn etwas gesammelt wurde) ---
            if mode == "retraining" and cycle < max_cycles - 1:
                if len(retraining_data_list) == 0:
                    logging.info("--- Kein neues Retraining-Material gesammelt; starte kein Retraining. ---")
                else:
                    logging.info(f"--- Zyklus {cycle + 1}: Datensammlung abgeschlossen. Starte Retraining (non-blocking). ---")
                    retraining_df = pd.DataFrame(retraining_data_list)
                    # Index setzen, falls vorhanden
                    if "datetime" in retraining_df.columns:
                        retraining_df = retraining_df.set_index("datetime")
                    # Liste für den nächsten Zyklus leeren
                    retraining_data_list.clear()

                    # nicht doppelt starten
                    with shared_resource_lock:
                        already_training = (PIPELINE_STATE.get("retraining_status") == "training")

                    if not already_training:
                        retraining_thread = threading.Thread(
                            target=retraining_thread_task,
                            args=(retraining_df.copy(), algorithm),
                            name=f"RetrainingThread-{cycle+1}",
                            daemon=True
                        )
                        retraining_thread.start()
                        logging.info("--- Retraining läuft im Hintergrund; Inferenz läuft weiter. ---")
                    else:
                        logging.info("--- Überspringe Retraining-Start: Ein Retraining läuft bereits. ---")

    except Exception as e:
        logging.error(f"Schwerwiegender Fehler im Inference Manager: {e}", exc_info=True)
        with shared_resource_lock:
            PIPELINE_STATE.update({"status": "error", "error_message": str(e)})
    finally:
        logging.info("--- Pipeline beendet. Aufräumen und Speichern. ---")
        with shared_resource_lock:
            PIPELINE_STATE.update({"status": "finished", "is_finished": True})

        # Sauber stoppen
        if inference_processor is not None and hasattr(inference_processor, 'stop'):
            try:
                inference_processor.stop()
            except Exception:
                pass

        # Letzten evtl. gepufferten Eintrag flushen
        if inference_processor is not None and hasattr(inference_processor, 'flush_pending_entry'):
            try:
                last_entry = inference_processor.flush_pending_entry()
                if last_entry:
                    with shared_resource_lock:
                        all_predictions.append(last_entry)
            except Exception:
                pass

        # Finale Ergebnisse speichern
        try:
            if inference_processor is not None and all_predictions:
                logging.info("Speichere finale Vorhersagen...")
                inference_processor.save_final_results(all_predictions)
        except Exception as e:
            logging.error(f"Fehler beim finalen Speichern der Ergebnisse: {e}", exc_info=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vereinheitlichte ML-Pipeline mit Web-UI")
    parser.add_argument('--algorithm', type=str, required=True, choices=['random_forest', 'lstm'], help="Zu verwendender Algorithmus.")
    parser.add_argument('--config-name', type=str, help="Optional: Name der Konfigurationsvariable.")
    parser.add_argument('--retraining', action=argparse.BooleanOptionalAction, default=False, help="Aktiviert den Retraining-Modus.")
    parser.add_argument("--load_id", type=str, help="Optionale Run ID zum Laden von Artefakten anstelle von Training.")
    parser.add_argument("--model_filename", type=str, help="Optional: Name der zu ladenden Modelldatei.")
    args = parser.parse_args()

    if args.config_name is None:
        args.config_name = f"{args.algorithm}"
        logging.info(f"Kein --config-name angegeben. Verwende Default: '{args.config_name}'")

    config = load_config_dynamically(args.algorithm, args.config_name)

    if args.algorithm == 'random_forest':
        from ML_Algorithms.Random_Forest.rf_train import RandomForestTrainer
        from ML_Algorithms.Random_Forest.rf_inference import RFInference
        trainer_class = RandomForestTrainer
        inference_class = RFInference
        folder_flag = "Random_Forest"
    else:
        from ML_Algorithms.LSTM.LSTM_train import LSTMTrainer
        from ML_Algorithms.LSTM.LSTM_inference import LSTMInference
        trainer_class = LSTMTrainer
        inference_class = LSTMInference
        folder_flag = "LSTM"

    config.update(CONFIG_LOAD_ARTIFACTS)
    config.update(MQTT_CONFIG)
    config['paths'] = CONFIG_PATH['paths']
    
    mode = "retraining" if args.retraining else "no_retraining"
    config['mode'] = mode
    port = 5002 if mode == "retraining" else 5001
    log_msg = f"--- MODUS: {mode.replace('_', ' ')} | ALGORITHMUS: {args.algorithm} | CONFIG: {args.config_name} ---"

    if args.model_filename: config['model_filename'] = args.model_filename
    if args.load_id:
        config['load_id'] = args.load_id
        config['run_id'] = args.load_id
        log_msg += f" | Lade Modell von Run ID: {args.load_id}"
        base_output_path = config['paths'].get('output')
        run_dir = os.path.join(base_output_path, folder_flag, args.load_id)
        config['paths'].update({
            "run_dir": run_dir, "Models": os.path.join(run_dir, "Models"),
            "Scalers": os.path.join(run_dir, "Scalers"),
            "Prediction_Data": os.path.join(run_dir, "Prediction_Data"),
            "Error_Metrics": os.path.join(run_dir, "Error_Metrics")
        })
    else:
        _, paths = PipelineUtils.setup_experiment(config, folder_flag, run_type='train')
        config['paths'] = paths

    logging.info(log_msg)

    if not args.load_id:
        threading.Thread(target=initial_training, args=(config, trainer_class, folder_flag), name="InitialTrainingThread", daemon=True).start()
    else:
        PIPELINE_STATE["status"] = "ready_for_inference"

    threading.Thread(target=inference_manager, args=(config, inference_class, folder_flag, args.algorithm, mode), name="InferenceManagerThread", daemon=True).start()

    from web_app import create_app
    app = create_app(config, PIPELINE_STATE, all_predictions, shared_resource_lock)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    # ===== NEUER TEIL: BROWSER AUTOMATISCH ÖFFNEN =====
    # Verwende localhost (127.0.0.1), da dies immer auf den lokalen Rechner verweist.
    url = f"http://127.0.0.1:{port}"
    
    def open_browser():
        webbrowser.open_new(url)

    # Starte einen Timer, der nach 1,5 Sekunden die Funktion zum Öffnen des Browsers aufruft.
    Timer(1.5, open_browser).start()
    
    app = create_app(config, PIPELINE_STATE, all_predictions, shared_resource_lock)
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    local_ip = PipelineUtils.get_local_ip()
    logging.info(f"\n🚀 Webserver startet. Öffnen Sie http://{local_ip}:{port} in Ihrem Browser.")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)