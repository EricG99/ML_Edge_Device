# ML_Algorithms/pipeline_web_app.py
import time
import logging
import argparse
import sys
import threading
import os

# --- Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Generelle Imports ---
from config.config_general import CONFIG_PATH, generate_run_id, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient
from ML_Algorithms.web_app import create_app

# from LSTM.LSTM_train import run_training as run_lstm_training
from config.config_ml_lstm import param_lstm_test, param_lstm_server


# --- Globale Zustandsvariablen (unverändert) ---
PIPELINE_STATE = {"status": "initializing", "error_message": None}
PREDICTION_DATA = {"step_data": [], "inference_step": -1}

# --- Globale, dynamisch geladene Objekte ---
inference_processor = None
mqtt_client = None

# Die Funktionen handle_new_prediction, run_inference_background_task und start_inference_flow
# bleiben exakt so, wie sie in Ihrer rf_web_app_pipeline.py waren. Sie sind generisch genug.

def handle_new_prediction(output_data):
    PREDICTION_DATA["step_data"].append(output_data)
    PREDICTION_DATA["inference_step"] += 1

def run_inference_background_task(broker_ip, port, topic):
    global inference_processor, mqtt_client
    PIPELINE_STATE["status"] = "inference_running"
    inference_loop_thread = threading.Thread(target=inference_processor.run_inference_loop, daemon=True)
    inference_loop_thread.start()
    mqtt_client = MqttInferenceClient(broker_ip, port, topic, inference_processor.update_latest_data)
    mqtt_client.run()
    logging.info(f"Inference process started. Collecting {PIPELINE_STATE.get('total_steps', 100)} data points...")
    while PREDICTION_DATA["inference_step"] < PIPELINE_STATE.get('total_steps', 100) - 1:
        if PIPELINE_STATE["status"] != "inference_running": break
        time.sleep(0.1)
    inference_processor.stop()
    if mqtt_client and mqtt_client.client._state == 'RUNNING':
        mqtt_client.client.loop_stop()
        mqtt_client.client.disconnect()
    if PIPELINE_STATE["status"] == "inference_running":
        PIPELINE_STATE["status"] = "finished"
    logging.info("Inference process finished.")

def start_inference_flow(config):
    global inference_processor
    PREDICTION_DATA["step_data"].clear()
    PREDICTION_DATA["inference_step"] = -1
    broker_ip, port, topic = config['MQTT_BROKER_IP'], config['MQTT_PORT'], config['MQTT_TOPIC']
    run_inference_background_task(broker_ip, port, topic)

def prepare_inference(config: dict, algorithm: str):
    """Lädt die Artefakte und initialisiert den spezifischen Inference Processor."""
    global inference_processor
    
    PIPELINE_STATE["status"] = "loading_artifacts"
    logging.info(f"Phase 1: Loading artifacts for algorithm '{algorithm}'...")
    
    try:
        # Laden der Artefakte (Modell, Scaler, Features)
        model, scaler, features, _ = PipelineUtils.load_model_artifacts_for_inference(config)
        
        # --- DYNAMISCHE AUSWAHL DES INFERENCE PROCESSORS ---
        if algorithm == 'random_forest':
            from ML_Algorithms.Random_Forest.rf_inference import LiveInferenceProcessor
        elif algorithm == 'lstm':
            # from ML_Algorithms.LSTM.lstm_inference import LiveInferenceProcessor # Zukünftig
            raise NotImplementedError("LSTM LiveInferenceProcessor not yet available.")
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Erstellen der Prozessor-Instanz
        inference_processor = LiveInferenceProcessor(
            model=model, scaler=scaler, feature_list=features, config=config,
            on_prediction_callback=handle_new_prediction
        )
        
        PIPELINE_STATE.update({
            "status": "ready_for_inference",
            "horizon": config.get("horizon", 1),
            "inference_interval_sec": config.get("inference_interval_sec", 1.0)
        })
        logging.info("Phase 1: Model & artifacts loaded. Ready for inference.")
        
    except Exception as e:
        logging.error(f"Error during preparation: {e}", exc_info=True)
        PIPELINE_STATE["status"] = "error"
        PIPELINE_STATE["error_message"] = str(e)


def main():
    parser = argparse.ArgumentParser(description="Allgemeine ML Pipeline mit Web Visualisierung")
    parser.add_argument('--algorithm', type=str, required=True, choices=['random_forest', 'lstm'])
    args = parser.parse_args()

    logging.info(f"--- MODE: infer (Web App) | ALGORITHM: {args.algorithm} ---")
    
    # --- Konfiguration zusammenbauen ---
    if args.algorithm == 'random_forest':
        from config.config_ml_rf import param_rf_test
        base_config = param_rf_test.copy()

    elif args.algorithm == 'lstm':
        from config.config_ml_lstm import param_lstm_test
        base_config = param_lstm_test.copy()

        
    base_config.update(CONFIG_LOAD_ARTIFACTS)
    base_config.update(MQTT_CONFIG)
    base_config['paths'] = CONFIG_PATH['paths']
    
    # Vorbereitung im Hintergrund-Thread starten
    preparation_thread = threading.Thread(target=prepare_inference, args=(base_config, args.algorithm), daemon=True)
    preparation_thread.start()
    
    # Flask App erstellen und starten
    app = create_app(
        pipeline_state=PIPELINE_STATE,
        prediction_data=PREDICTION_DATA,
        start_inference_callback=lambda: start_inference_flow(base_config)
    )
    
    logging.info("\n🚀 Web server starting. Open http://127.0.0.1:5001 in your browser.")
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()