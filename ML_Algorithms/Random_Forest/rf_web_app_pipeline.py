# rf_pipeline.py
import time
import logging
import argparse
import sys
import threading
import os

# --- Suppress warnings ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
#from config.config_general import CONFIG_PATH, MQTT_CONFIG, generate_run_id
from config.config_ml_rf import param_rf_test
from config.config_general import CONFIG_PATH, generate_run_id, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS

from ML_Algorithms.Random_Forest.rf_train import run_training
from ML_Algorithms.Random_Forest.rf_inference import LiveInferenceProcessor
from ML_Algorithms.web_app import create_app
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Global State and Data Containers (shared between threads) ---
PIPELINE_STATE = {
    "status": "initializing",
    "error_message": None,
    "total_steps": 100,
    "horizon": 1,
    "inference_interval_sec": 1.0
}
PREDICTION_DATA = {
    "step_data": [],
    "inference_step": -1
}


# --- Background Task Management ---
inference_processor = None
mqtt_client = None

def handle_new_prediction(output_data):
    """Callback function for the LiveInferenceProcessor."""
    PREDICTION_DATA["step_data"].append(output_data)
    PREDICTION_DATA["inference_step"] += 1

def run_inference_background_task(broker_ip, port, topic):
    """The full background task: runs MQTT client and the inference loop."""
    global inference_processor, mqtt_client
    
    PIPELINE_STATE["status"] = "inference_running"
    
    # Start the inference loop. It will call handle_new_prediction on each result.
    inference_loop_thread = threading.Thread(target=inference_processor.run_inference_loop, daemon=True)
    inference_loop_thread.start()

    # Setup and run MQTT client
    mqtt_client = MqttInferenceClient(
        broker_ip=broker_ip, port=port, topic=topic,
        on_message_callback=inference_processor.update_latest_data
    )
    mqtt_client.run() # Blocks until client disconnects

    # Wait until the required number of steps is collected or status changes
    logging.info(f"Inference process started. Collecting {PIPELINE_STATE['total_steps']} data points...")
    while PREDICTION_DATA["inference_step"] < PIPELINE_STATE['total_steps'] - 1:
        if PIPELINE_STATE["status"] != "inference_running":
            break
        time.sleep(0.1)
    
    # Clean shutdown
    inference_processor.stop()
    if mqtt_client and mqtt_client.client._state == 'RUNNING':
        mqtt_client.client.loop_stop()
        mqtt_client.client.disconnect()
    
    if PIPELINE_STATE["status"] == "inference_running":
        PIPELINE_STATE["status"] = "finished"
    logging.info("Inference process finished.")

def start_inference_flow():
    """Function to be called by the web app to start the inference."""
    global inference_processor
    
    # Reset state for a new run
    PREDICTION_DATA["step_data"].clear()
    PREDICTION_DATA["inference_step"] = -1
    
    # Retrieve MQTT config from base_config used to launch the app
    broker_ip = base_config['MQTT_BROKER_IP']
    port = base_config['MQTT_PORT']
    topic = base_config['MQTT_TOPIC']
    
    # The 'inference_processor' is already loaded and configured
    run_inference_background_task(broker_ip, port, topic)

def simulate_training_and_prepare_inference(config):
    """Simulates training and loads artifacts to prepare for inference."""
    global inference_processor
    
    PIPELINE_STATE["status"] = "training"
    logging.info("Simulating Phase 1: Model training...")
    time.sleep(2) # Simulate work
    
    try:
        scaler, features, model = PipelineUtils.load_model_artifacts_for_inference(config)
        
        # Update global state from config
        PIPELINE_STATE["horizon"] = config.get("horizon", 1)
        PIPELINE_STATE["inference_interval_sec"] = config.get("inference_interval_sec", 1.0)
        
        # Create the processor instance that will be used by the background task
        inference_processor = LiveInferenceProcessor(
            model=model, scaler=scaler, feature_list=features, config=config,
            on_prediction_callback=handle_new_prediction
        )
        PIPELINE_STATE["status"] = "ready_for_inference"
        logging.info("Phase 1: Model & artifacts loaded. Ready for inference.")
        
    except FileNotFoundError as e:
        logging.error(f"Error loading artifacts: {e}", exc_info=True)
        PIPELINE_STATE["status"] = "error"
        PIPELINE_STATE["error_message"] = str(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        PIPELINE_STATE["status"] = "error"
        PIPELINE_STATE["error_message"] = str(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RF Live Inference Pipeline with Web Visualization")
    parser.add_argument('--mode', type=str, default='infer', choices=['train', 'infer'], help="Select 'train' or 'infer'.")
    args = parser.parse_args()

    # --- KORREKTE, ROBUSTE METHODE ZUM ZUSAMMENFÜHREN DER KONFIGURATION ---

    # 1. Starte mit den modellspezifischen Parametern als Basis
    base_config = param_rf_test.copy()
    
    # 2. Füge die Lade- und MQTT-Konfigurationen hinzu
    base_config.update(CONFIG_LOAD_ARTIFACTS)
    base_config.update(MQTT_CONFIG)
    
    # 3. Füge das 'paths'-Wörterbuch als verschachteltes Element hinzu.
    #    Dies ist die entscheidende Korrektur für den KeyError.
    base_config['paths'] = CONFIG_PATH['paths']
    
    # 4. Füge laufzeitspezifische Werte hinzu
    run_id = generate_run_id()
    base_config['run_id'] = run_id
    base_config['time_stamp'] = run_id.split('_')[1]


    # --- Modus auswählen ---
    if args.mode == 'train':
        logging.info("--- MODE: train ---")
        run_training(config=base_config, save_artifacts=True)
        logging.info("\n✅ Training complete. Run the script again in 'infer' mode to start the web app.")

    elif args.mode == 'infer':
        logging.info("--- MODE: infer (with Web Visualization) ---")
        
        # Start the preparation thread (simulated training + artifact loading)
        preparation_thread = threading.Thread(
            target=simulate_training_and_prepare_inference,
            args=(base_config,),
            daemon=True
        )
        preparation_thread.start()
        
        # Create and run the Flask App
        app = create_app(
            pipeline_state=PIPELINE_STATE,
            prediction_data=PREDICTION_DATA,
            start_inference_callback=start_inference_flow
        )
        
        logging.info(f"\n🚀 Web server starting. Open http://127.0.0.1:5001 in your browser.")
        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)