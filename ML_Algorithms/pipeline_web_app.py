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


def inference_manager(config: dict, inference_class, folder_flag: str, algorithm: str, mode: str):
    """
    Einheitlicher Inferenz-Manager.
    Die Kernlogik für Zyklen, Retraining und Pausen ist jetzt für beide Ladestrategien ('split' und 'live_mqtt') identisch.
    """
    global all_predictions
    retraining_data_list = []
    mqtt_client = None
    inference_processor = None

    try:
        # 1. INITIALISIERUNG (Warten und Prozessor/Modell laden)
        while PIPELINE_STATE["status"] == "initializing":
            time.sleep(1)
        if PIPELINE_STATE["status"] == "error":
            logging.error("Inferenz-Manager startet nicht, da ein Fehler bei der Initialisierung aufgetreten ist.")
            return

        with shared_resource_lock:
            PIPELINE_STATE["status"] = "inference_running"
        logging.info(f"--- INFERENCE MANAGER: Startet im Modus '{mode}' ---")

        inference_processor = inference_class(config, "", 0, "", folder_flag)

        model_is_trained_in_memory = shared_model.get("model") is not None
        if model_is_trained_in_memory:
            logging.info("Verwende das neu trainierte Modell aus dem initialen Lauf.")
            with shared_resource_lock:
                inference_processor.model = shared_model["model"]
                inference_processor.scaler = shared_model["scaler"]
                inference_processor.feature_list = shared_model["features"]
                inference_processor.target_feature = shared_model["config"]["base_features"][0]
        elif config.get('load_id'):
            logging.info(f"Lade Artefakte von Run ID: {config['load_id']}")
            inference_processor.load_artifacts()
        else:
            logging.error("Kein trainiertes Modell und keine load_id zum Laden gefunden.")
            with shared_resource_lock:
                PIPELINE_STATE.update({"status": "error", "error_message": "Kein Modell verfügbar."})
            return

        # 🔁 Einheitliches Backend für LSTM: TFLite bevorzugen (wie im MQTT-Fall bewährt)
        if algorithm == "lstm":
            try:
                models_dir = config["paths"].get("Models") or config["paths"].get("models")
                tfl_name = config.get("model_filename", "model_quant_float16.tflite")
                tfl_path = os.path.join(models_dir, tfl_name) if models_dir else None
                if tfl_path and os.path.exists(tfl_path):
                    interpreter = tf.lite.Interpreter(model_path=tfl_path)
                    # Direkt auf erwartete Form bringen und einmal allocaten
                    n_feats = len(inference_processor.feature_list or [])
                    lags = int(config.get("lags", 1))
                    try:
                        interpreter.resize_tensor_input(interpreter.get_input_details()[0]["index"],
                                                        [1, lags, n_feats], strict=False)
                    except Exception:
                        pass
                    interpreter.allocate_tensors()
                    inference_processor.model = interpreter
                    logging.info(f"ℹ️ LSTM-Inferenz nutzt TFLite-Interpreter: {tfl_path}")
                else:
                    logging.warning("⚠️ Kein TFLite-Modell gefunden – nutze Keras-Modell für LSTM.")
            except Exception as e:
                logging.warning(f"⚠️ TFLite-Interpreter konnte nicht geladen werden ({e}) – nutze Keras-Modell.")

        # 2. DATENQUELLE VORBEREITEN & SCHLEIFENPARAMETER SETZEN
        loading_strategy = config.get("loading_strategy", "live_mqtt")
        logging.info(f"Gewählte Ladestrategie: '{loading_strategy}'")
        data_source = []

        if mode == "retraining":
            max_cycles = config.get("retraining_cycles", 2)
            steps_per_cycle = config.get("retraining_interval_steps", 200)
        else:  # no_retraining
            max_cycles = 1
            steps_per_cycle = config.get("inference_steps", 500)

        if loading_strategy == "split":
            logging.info("Starte Inferenz im BATCH-MODUS aus CSV-Datei (1 Hz Simulation).")
            test_df = LoadPrepareData.load_test_data_by_fraction(
                config=config,
                train_fraction=config.get("train_fraction", 0.7),
                make_date_as_index=False
            )

            if test_df.empty:
                logging.warning("Test-DataFrame ist leer. Es gibt nichts zu verarbeiten.")
                return

            # Payloads minimal halten: nur datetime + base_features (Realtime-Prozessor berechnet Rest)
            data_source = []
            test_df = test_df.sort_values("datetime")
            for _, row in test_df.iterrows():
                row_dict = {str(k).lower(): v for k, v in row.to_dict().items()}
                rec = {"datetime": pd.to_datetime(row_dict.get("datetime"))}
                for feat in config.get("base_features", []):
                    f_lower = str(feat).lower()
                    if f_lower in row_dict:
                        rec[feat] = row_dict[f_lower]  # Key im Originalnamen belassen
                    else:
                        logging.warning(f"Base-Feature '{feat}' nicht in CSV gefunden – setze NaN.")
                        rec[feat] = np.nan
                data_source.append(rec)

            # Schritte auf die Datenlänge begrenzen (optional)
            steps_per_cycle = min(steps_per_cycle, len(data_source))

        else:  # live_mqtt
            mqtt_client = MqttInferenceClient(
                broker_ip=config['MQTT_BROKER_IP'],
                port=config['MQTT_PORT'],
                topic=config['MQTT_TOPIC'],
                on_message_callback=inference_processor.update_latest_data
            )
            # Nicht blockierend starten, damit die vereinheitlichte Schleife laufen kann
            mqtt_client.run()

        target_interval_sec = config.get("inference_interval_sec", 1.0)
        with shared_resource_lock:
            PIPELINE_STATE.update({"total_cycles": max_cycles, "total_steps": steps_per_cycle, "mode": mode})

        # 3. VEREINHEITLICHTE HAUPTSCHLEIFE
        current_data_index = 0
        for cycle in range(max_cycles):
            with shared_resource_lock:
                PIPELINE_STATE.update({"cycle_count": cycle + 1,
                                       "retraining_status": "collecting" if mode == "retraining" else "idle"})
            logging.info(f"--- Zyklus {cycle + 1}/{max_cycles}: Starte Inferenz. ---")

            for step in range(steps_per_cycle):
                start_cycle_time = time.perf_counter()
                while PIPELINE_STATE["is_paused"]:
                    time.sleep(0.5)
                if PIPELINE_STATE["is_finished"]:
                    break

                # --- Einheitliche Datenbeschaffung ---
                if loading_strategy == "split":
                    if current_data_index < len(data_source):
                        inference_processor.latest_payload = data_source[current_data_index]
                        current_data_index += 1
                    else:
                        logging.info("Alle Daten aus der CSV-Datei verarbeitet.")
                        break  # Innere Schleife beenden, wenn alle Daten verarbeitet sind

                with shared_resource_lock:
                    PIPELINE_STATE["steps_in_cycle"] = step + 1

                # Wenn (noch) keine Daten vorhanden, 1-Hz-Takt trotzdem halten
                if inference_processor.latest_payload is None:
                    elapsed = time.perf_counter() - start_cycle_time
                    sleep_duration = target_interval_sec - elapsed
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)
                    continue

                # --- Einheitliche Verarbeitungslogik ---
                with shared_resource_lock:
                    if model_is_trained_in_memory:
                        inference_processor.model = shared_model.get("model")

                input_data, timestamp, true_value = inference_processor._prepare_input_data()

                # NEU: Guard – wenn Fenster noch nicht bereit, 1 Hz halten und nächste Iteration
                if input_data is None:
                    elapsed = time.perf_counter() - start_cycle_time
                    sleep_duration = target_interval_sec - elapsed
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)
                    continue

                # Debug: Inputfenster
                try:
                    logging.debug(f"[{algorithm}] input_window shape={input_data.shape}, "
                                  f"min/max={float(input_data.min()):.6f}/{float(input_data.max()):.6f}")
                except Exception:
                    pass

                # Vereinheitlichte Inferenz (Keras oder TFLite)
                try:
                    prediction_scaled, model_inference_time_ms = _run_inference_unified(
                        inference_processor.model, input_data
                    )
                except Exception as e:
                    logging.error(f"Inferenz fehlgeschlagen: {e}", exc_info=True)
                    # Takt halten und weiter
                    elapsed = time.perf_counter() - start_cycle_time
                    sleep_duration = target_interval_sec - elapsed
                    if sleep_duration > 0:
                        time.sleep(sleep_duration)
                    continue

                try:
                    logging.debug(f"[{algorithm}] pred_scaled[:5]={np.asarray(prediction_scaled).reshape(-1)[:5]}")
                except Exception:
                    pass

                if algorithm == "random_forest":
                    # RF gibt y bereits im Originalraum aus -> KEIN inverse_transform!
                    predictions_unscaled_all = np.asarray(prediction_scaled).reshape(-1)
                else:
                    # (LSTM, CNN, …): y ist skaliert.
                    # Wir verwenden den y_scaler, der speziell für die Zielvariable trainiert wurde.
                    if inference_processor.y_scaler:
                        # Das Modell gibt (1, H) aus. Der Scaler erwartet (N, 1).
                        # -> Reshape von (1, H) zu (H, 1)
                        pred_reshaped = np.asarray(prediction_scaled).reshape(-1, 1)
                        # Anwenden der korrekten Invers-Transformation
                        pred_unscaled_reshaped = inference_processor.y_scaler.inverse_transform(pred_reshaped)
                        # Zurück in eine flache Form (H,)
                        predictions_unscaled_all = pred_unscaled_reshaped.flatten()
                    else:
                        # Fallback, falls y nicht skaliert wurde (scale_target=False)
                        predictions_unscaled_all = np.asarray(prediction_scaled).flatten()

                try:
                    logging.debug(f"[{algorithm}] pred_unscaled[:5]={predictions_unscaled_all[:5]}")
                except Exception:
                    pass

                total_processing_time_ms = (time.perf_counter() - start_cycle_time) * 1000.0
                prediction_entry = {
                    "datetime": timestamp,
                    "prediction": float(predictions_unscaled_all[0]) if predictions_unscaled_all.size > 0 else None,
                    "true_value": None if true_value is None else float(true_value),
                    "rolling_forecast": predictions_unscaled_all.tolist(),
                    "cpu_load": PipelineUtils.get_cpu_usage(),
                    "ram_usage": PipelineUtils.get_memory_usage(),
                    "model_inference_time_ms": float(model_inference_time_ms),
                    "total_processing_time_ms": float(total_processing_time_ms)
                }
                with shared_resource_lock:
                    all_predictions.append(prediction_entry)

                if mode == "retraining":
                    retraining_data_list.append(inference_processor.latest_payload)

                # Payload zurücksetzen und 1-Hz-Takt stabilisieren
                inference_processor.latest_payload = None
                elapsed = time.perf_counter() - start_cycle_time
                sleep_duration = target_interval_sec - elapsed
                if sleep_duration > 0:
                    time.sleep(sleep_duration)

            if PIPELINE_STATE["is_finished"]:
                break

            # --- Einheitliches Retraining-Handling ---
            if mode == "retraining" and cycle < max_cycles - 1:
                logging.info(f"--- Zyklus {cycle + 1}: Datensammlung abgeschlossen. Starte Nachtraining. ---")
                retraining_data_buffer = pd.DataFrame(retraining_data_list)
                retraining_data_buffer.columns = retraining_data_buffer.columns.str.lower()
                retraining_data_buffer['datetime'] = pd.to_datetime(retraining_data_buffer['datetime'])
                retraining_data_buffer = retraining_data_buffer.set_index('datetime')

                retraining_thread = threading.Thread(
                    target=retraining_thread_task,
                    args=(retraining_data_buffer.copy(), algorithm),
                    name=f"RetrainingThread-{cycle+1}"
                )
                retraining_thread.start()
                retraining_data_list.clear()

    finally:
        # 4. AUFRÄUMEN UND SPEICHERN
        logging.info("--- Pipeline beendet. Aufräumen und Speichern. ---")
        with shared_resource_lock:
            PIPELINE_STATE.update({"status": "finished", "is_finished": True})

        if mqtt_client:
            logging.info("Trenne MQTT-Client...")
            try:
                mqtt_client.client.loop_stop()
                mqtt_client.client.disconnect()
                logging.info("MQTT-Client erfolgreich getrennt.")
            except Exception as e:
                logging.error(f"Fehler beim Trennen des MQTT-Clients: {e}", exc_info=True)

        if all_predictions:
            logging.info("🔄 Speichere finale Vorhersagen mit CPU/RAM und Zeiten...")
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
                    y_pred = np.stack(df["rolling_forecast"].dropna().to_numpy())
                    valid_indices = df["rolling_forecast"].notna()
                    df_valid = df[valid_indices].copy()
                    if not df_valid.empty:
                        horizon = y_pred.shape[1]
                        df_valid["true_value_filled"] = df_valid["true_value"].ffill().bfill()
                        y_true = np.tile(df_valid["true_value_filled"].to_numpy().reshape(-1, 1), reps=(1, horizon))
                        dates = pd.to_datetime(df_valid["datetime"])
                        true_cols = [f"true_t+{i+1}" for i in range(horizon)]
                        pred_cols = [f"pred_t+{i+1}" for i in range(horizon)]
                        true_df = pd.DataFrame(y_true, columns=true_cols, index=df_valid.index)
                        pred_df = pd.DataFrame(y_pred, columns=pred_cols, index=df_valid.index)
                        system_cols = [c for c in ["cpu_load", "model_inference_time_ms", "total_processing_time_ms",
                                                   "ram_total_gb", "ram_used_gb", "ram_percent"] if c in df.columns]
                        sysinfo_df = df_valid[system_cols]
                        full_df = pd.concat([
                            pd.DataFrame({"date": dates.values}),
                            true_df.reset_index(drop=True),
                            pred_df.reset_index(drop=True),
                            sysinfo_df.reset_index(drop=True)
                        ], axis=1)
                        pred_path = os.path.join(config["paths"]["Prediction_Data"],
                                                 f"prediction_final_{config.get('run_id')}.csv")
                        os.makedirs(os.path.dirname(pred_path), exist_ok=True)
                        full_df.to_csv(pred_path, index=False)
                        metrics = PipelineUtils.evaluate_all_metrics(y_true, y_pred, horizon=config.get("horizon", 1))
                        PipelineUtils.save_metrics_summary(
                            metrics, config,
                            (getattr(inference_processor, "training_config", {}) if inference_processor else {}),
                            config["paths"]
                        )
                        logging.info(f"✅ Vorhersagen mit Systemdaten gespeichert: {pred_path}")
                except Exception as e:
                    logging.error(f"Fehler beim Speichern der erweiterten Vorhersagen: {e}", exc_info=True)


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
