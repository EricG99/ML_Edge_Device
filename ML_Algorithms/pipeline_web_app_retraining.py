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

# --- Systempfad-Setup ---
# Stellt sicher, dass das Projekt-Hauptverzeichnis im Python-Pfad ist
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
# PIPELINE_STATE wird von der Web-App verwendet, um den aktuellen Status anzuzeigen
PIPELINE_STATE = {
    "status": "initializing", 
    "error_message": None,
    "retraining_status": "idle", # idle, collecting, training
    "cycle_count": 0,
    "steps_in_cycle": 0
}
# Lock, um den sicheren Zugriff auf geteilte Ressourcen (Modell, Datenpuffer) zu gewährleisten
shared_resource_lock = threading.Lock()

# --- Globale Objekte für das Modell und die Daten ---
# Diese werden von mehreren Threads gemeinsam genutzt
shared_model = {
    "model": None,
    "scaler": None,
    "features": None,
    "config": None
}
# Puffer für die Inferenzdaten und die Daten für das Nachtraining
inference_processor = None
retraining_data_buffer = pd.DataFrame()
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
    """
    Führt das initiale Training des Modells durch, ohne die Artefakte zu speichern.
    Das trainierte Modell, der Scaler und die Features werden in der globalen Variable 'shared_model' gespeichert.
    """
    global shared_model
    logging.info("--- PHASE 1: Initiales Training startet ---")
    
    # LSTMTrainer wird mit der Konfiguration initialisiert
    trainer = LSTMTrainer(config=config, folder_flag="LSTM_retrain")
    
    # Die run-Methode wird mit save_artifacts=False aufgerufen.
    # Sie gibt das trainierte Modell, den Scaler und die Feature-Liste zurück.
    model, scaler, features = trainer.run(save_artifacts=False)
    
    # Sicherer Zugriff auf die globalen geteilten Ressourcen
    with shared_resource_lock:
        shared_model["model"] = model
        shared_model["scaler"] = scaler
        shared_model["features"] = features
        shared_model["config"] = config # Speichern der Konfiguration für das Nachtraining
        PIPELINE_STATE["status"] = "initial_training_complete"
    logging.info("--- PHASE 1: Initiales Training abgeschlossen. Modell ist im Speicher. ---")


def retraining_thread_task(current_model, scaler, features, retraining_data, config):
    """
    Diese Funktion wird in einem separaten Thread ausgeführt, um das Modell neu zu trainieren.
    Sie läuft parallel zur Inferenz.
    """
    global shared_model
    logging.info("--- RETRAINING THREAD: Startet Nachtraining des Modells. ---")
    PIPELINE_STATE["retraining_status"] = "training"

    try:
        # 1. Datenvorbereitung für das inkrementelle Training
        # Die gesammelten Live-Daten werden für das LSTM-Format vorbereitet.
        # Feature Engineering anwenden
        retraining_data_featured, _ = fe.add_all_features(retraining_data, config)
        retraining_data_featured.dropna(inplace=True)

        # Daten mit dem *vorhandenen* Scaler transformieren
        scaled_data = scaler.transform(retraining_data_featured[features])
        
        # In Sliding Windows umwandeln
        X_retrain, y_retrain = LoadPrepareData.convert_data_to_sliding_window(
            scaled_data,
            lag_horizon=config["lags"],
            forecast_horizon=config["horizon"]
        )

        if len(X_retrain) == 0:
            logging.warning("RETRAINING THREAD: Nicht genügend Daten für das Nachtraining nach der Vorbereitung.")
            PIPELINE_STATE["retraining_status"] = "idle"
            return

        # 2. Inkrementelles Training (model.fit)
        # Erstelle eine tiefe Kopie des Modells, um das Original nicht zu blockieren
        model_to_retrain = deepcopy(current_model)

        logging.info(f"RETRAINING THREAD: Führe model.fit mit {len(X_retrain)} neuen Datenpunkten aus.")
        model_to_retrain.fit(
            X_retrain, y_retrain,
            epochs=config.get("retraining_epochs", 5), # Weniger Epochen für schnelles Nachtraining
            batch_size=config.get("batch_size", 32),
            verbose=1
        )

        # 3. Modellaustausch (Atomarer Vorgang)
        # Das alte Modell wird durch das neu trainierte ersetzt.
        with shared_resource_lock:
            shared_model["model"] = model_to_retrain
            logging.info("--- RETRAINING THREAD: Modellaustausch erfolgreich! Das neue Modell ist jetzt aktiv. ---")

    except Exception as e:
        logging.error(f"RETRAINING THREAD: Fehler im Nachtrainings-Thread: {e}", exc_info=True)
    finally:
        PIPELINE_STATE["retraining_status"] = "idle"


def inference_and_retraining_manager(config: dict):
    """
    Haupt-Controller, der die Inferenzschleife verwaltet, Daten sammelt 
    und periodisch das Nachtraining anstößt.
    """
    global inference_processor, retraining_data_buffer, all_predictions
    
    # Warten, bis das initiale Training abgeschlossen ist
    while PIPELINE_STATE["status"] != "initial_training_complete":
        time.sleep(1)
        
    logging.info("--- PHASE 2: Starte Inferenz- und Retraining-Manager ---")
    PIPELINE_STATE["status"] = "inference_running"

    # Inferenz-Prozessor initialisieren (ohne Artefakte zu laden, da sie im Speicher sind)
    inference_processor = LSTMInference(config, "", 0, "", "LSTM_retrain")
    with shared_resource_lock:
        inference_processor.model = shared_model["model"]
        inference_processor.scaler = shared_model["scaler"]
        inference_processor.feature_list = shared_model["features"]
    
    # MQTT-Client für Live-Daten einrichten
    mqtt_client = MqttInferenceClient(
        broker_ip=config['MQTT_BROKER_IP'], port=config['MQTT_PORT'], topic=config['MQTT_TOPIC'],
        on_message_callback=inference_processor.update_latest_data
    )
    mqtt_client.run() # Startet den non-blocking MQTT Loop

    # --- Hauptschleife ---
    max_cycles = config.get("retraining_cycles", 5)
    steps_per_cycle = config.get("retraining_interval_steps", 100)
    
    for cycle in range(max_cycles):
        PIPELINE_STATE["cycle_count"] = cycle + 1
        PIPELINE_STATE["retraining_status"] = "collecting"
        logging.info(f"--- Zyklus {cycle + 1}/{max_cycles}: Starte Datensammlung für das nächste Training. ---")

        for step in range(steps_per_cycle):
            PIPELINE_STATE["steps_in_cycle"] = step + 1
            
            # Warten auf neue Nachricht
            while inference_processor.latest_payload is None:
                time.sleep(0.1)

            # 1. Inferenz ausführen
            with shared_resource_lock:
                # Stelle sicher, dass wir das aktuellste Modell für die Inferenz verwenden
                inference_processor.model = shared_model["model"]
            
            # Führe eine einzelne Vorhersage aus
            input_data, timestamp, true_value = inference_processor._prepare_input_data()
            
            if input_data is not None:
                prediction_scaled, _ = PipelineUtils.run_timed_inference(model=inference_processor.model, input_data=input_data)
                
                target_index = inference_processor.feature_list.index(inference_processor.target_feature)
                prediction_unscaled = PipelineUtils.safe_inverse_transform(
                    scaler=inference_processor.scaler, array=prediction_scaled.reshape(1, -1), target_index=target_index
                )
                
                # Speichere Vorhersage und Rohdaten
                all_predictions.append({"datetime": timestamp, "prediction": prediction_unscaled.flatten()[0], "true_value": true_value})
                
                # Füge Rohdaten zum Retraining-Puffer hinzu
                new_row = pd.DataFrame([inference_processor.latest_payload])
                new_row['datetime'] = pd.to_datetime(new_row['datetime'])
                new_row = new_row.set_index('datetime')
                retraining_data_buffer = pd.concat([retraining_data_buffer, new_row])

            # Reset für nächsten Schritt
            inference_processor.latest_payload = None
            time.sleep(config.get("inference_interval_sec", 1.0))

        # 2. Nachtraining anstoßen (in einem neuen Thread)
        logging.info(f"--- Zyklus {cycle + 1}: Datensammlung abgeschlossen. Starte Nachtraining. ---")
        with shared_resource_lock:
            # Erstelle Kopien der Artefakte für den Thread
            model_copy = deepcopy(shared_model["model"])
            scaler_copy = deepcopy(shared_model["scaler"])
            features_copy = shared_model["features"][:]
            config_copy = deepcopy(shared_model["config"])

        retraining_thread = threading.Thread(
            target=retraining_thread_task,
            args=(model_copy, scaler_copy, features_copy, retraining_data_buffer.copy(), config_copy),
            name=f"RetrainingThread-{cycle+1}"
        )
        retraining_thread.start()

        # Retraining-Puffer für den nächsten Zyklus leeren
        retraining_data_buffer = pd.DataFrame()

    # Warten, bis der letzte Retraining-Thread fertig ist
    if 'retraining_thread' in locals() and retraining_thread.is_alive():
        retraining_thread.join()

    # --- Aufräumen und Ergebnisse speichern ---
    logging.info("--- PHASE 3: Maximale Zyklen erreicht. Beende Pipeline und speichere Ergebnisse. ---")
    mqtt_client.client.loop_stop()
    mqtt_client.client.disconnect()

    # Speichern der gesammelten Vorhersagen
    if all_predictions:
        config, paths = PipelineUtils.setup_experiment(config, "LSTM_retrain_final", run_type='inference')
        results_df = pd.DataFrame(all_predictions)
        pred_filename = f"retraining_inference_results_{config['run_id']}.csv"
        pred_output_path = os.path.join(paths.get("Prediction_Data"), pred_filename)
        results_df.to_csv(pred_output_path, index=False)
        logging.info(f"Finale Vorhersagen gespeichert unter: {pred_output_path}")

        # Metriken berechnen und speichern
        y_true = results_df["true_value"].to_numpy()
        y_pred = results_df["prediction"].to_numpy()
        metrics = PipelineUtils.evaluate_all_metrics(y_true, y_pred)
        PipelineUtils.save_metrics_summary(metrics, config, config, paths)

    PIPELINE_STATE["status"] = "finished"
    logging.info("--- Pipeline erfolgreich beendet. ---")


def main():
    """Hauptfunktion zum Starten der Pipeline und der Web-App."""
    parser = argparse.ArgumentParser(description="LSTM Pipeline with Incremental Retraining and Web UI")
    args = parser.parse_args()

    # --- Konfiguration laden ---
    from config.config_ml_lstm import param_lstm_test
    config = param_lstm_test.copy()
    config.update(CONFIG_LOAD_ARTIFACTS)
    config.update(MQTT_CONFIG)
    config['paths'] = CONFIG_PATH['paths']
    
    # --- Start der Pipeline-Logik in einem Hintergrundthread ---
    pipeline_thread = threading.Thread(target=initial_training, args=(config,), name="InitialTrainingThread")
    pipeline_thread.start()
    
    manager_thread = threading.Thread(target=inference_and_retraining_manager, args=(config,), name="ManagerThread")
    manager_thread.start()

    # --- Flask Web App Setup ---
    app = Flask(__name__, template_folder=os.path.join(project_root, 'ML_Algorithms', 'templates'))

    @app.route('/')
    def index():
        return render_template('dashboard_retrain.html')

    @app.route('/api/status')
    def get_status():
        # Gibt den aktuellen Zustand der Pipeline an das Frontend
        return jsonify(PIPELINE_STATE)

    @app.route('/api/data')
    def get_data():
        # Gibt die letzte Vorhersage an das Frontend
        if not all_predictions:
            return jsonify({"status": "waiting"})
        return jsonify({"status": "success", "data": all_predictions[-1]})

    # --- Starte die Web-App ---
    local_ip = get_local_ip()
    logging.info(f"🚀 Web server starting. Open http://{local_ip}:5002 in your browser.")
    app.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
