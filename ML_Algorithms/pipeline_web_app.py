import time
import logging
import argparse
import sys
import threading
import os
from flask import Flask, jsonify, render_template, request

# --- Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- General Imports ---
from config.config_general import CONFIG_PATH, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient
from ML_Algorithms.web_app import create_app

# --- Global State Variables ---
PIPELINE_STATE = {"status": "initializing", "error_message": None}
app_lock = threading.Lock()

# --- Global, dynamically loaded objects ---
inference_processor = None
mqtt_client = None


def run_inference_background_task(config):
    """
    Orchestrates the entire inference process in a background thread.
    It manages the MQTT client and the inference loop via the processor.
    """
    global inference_processor, mqtt_client
    
    with app_lock:
        if PIPELINE_STATE["status"] == "inference_running":
            logging.warning("Inference is already running.")
            return
        
        PIPELINE_STATE["status"] = "inference_running"
        logging.info("Starting inference background task...")

        # Setup MQTT Client
        broker_ip = config['MQTT_BROKER_IP']
        port = config['MQTT_PORT']
        topic = config['MQTT_TOPIC']
        mqtt_client = MqttInferenceClient(
            broker_ip=broker_ip, port=port, topic=topic,
            on_message_callback=inference_processor.update_latest_data
        )

    # Start MQTT (non-blocking) and the inference loop thread
    inference_thread = threading.Thread(target=inference_processor._run_inference_loop, daemon=True)
    
    mqtt_client.run()  # Starts the non-blocking MQTT loop
    inference_thread.start()
    logging.info("🚀 Inference process and MQTT client started.")

    # Wait for the inference thread to finish its job
    # The loop inside the processor will stop based on 'inference_steps'
    inference_thread.join()

    # --- Cleanup Phase ---
    logging.info("Inference loop finished. Cleaning up resources...")
    if mqtt_client:
        mqtt_client.client.loop_stop()
        mqtt_client.client.disconnect()
        logging.info("MQTT client stopped.")
    
    # Save results
    inference_processor._save_results()
    
    PIPELINE_STATE["status"] = "finished"
    logging.info("✅ Inference background task complete.")


def start_inference_flow(config):
    """Wrapper to start the background task in a new thread."""
    # Reset state from previous runs if necessary
    if inference_processor:
        inference_processor.step_counter = 0
        inference_processor.results_buffer.clear()
        
    # Start the main task in a daemon thread so it doesn't block the web server
    thread = threading.Thread(target=run_inference_background_task, args=(config,), daemon=True)
    thread.start()


def prepare_inference(config: dict, algorithm: str):
    """Loads artifacts and initializes the specific Inference Processor."""
    global inference_processor
    
    with app_lock:
        PIPELINE_STATE["status"] = "loading_artifacts"
        logging.info(f"Phase 1: Loading artifacts for algorithm '{algorithm}'...")
        
        try:
            # --- DYNAMIC SELECTION OF THE INFERENCE PROCESSOR ---
            if algorithm == 'random_forest':
                from ML_Algorithms.Random_Forest.rf_inference import RFInference
                processor_class = RFInference
            elif algorithm == 'lstm':
                from ML_Algorithms.LSTM.lstm_inference import LSTMInference
                processor_class = LSTMInference
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
            
            # Instantiate the processor
            # The MQTT details are passed here but will be used later
            proc = processor_class(
                config=config,
                broker_ip=config['MQTT_BROKER_IP'],
                port=config['MQTT_PORT'],
                topic=config['MQTT_TOPIC']
            )
            
            # Load artifacts (model, scaler, features)
            proc.load_artifacts()
            
            # Store the ready-to-use processor instance globally
            inference_processor = proc
            
            PIPELINE_STATE.update({
                "status": "ready_for_inference",
                "horizon": config.get("horizon", 1),
                "inference_interval_sec": config.get("inference_interval_sec", 1.0),
                "total_steps": inference_processor.inference_steps
            })
            logging.info("✅ Phase 1: Model & artifacts loaded. Ready for inference.")
            
        except Exception as e:
            logging.error(f"Error during preparation: {e}", exc_info=True)
            PIPELINE_STATE["status"] = "error"
            PIPELINE_STATE["error_message"] = str(e)


def main():
    parser = argparse.ArgumentParser(description="General ML Pipeline with Web Visualization")
    parser.add_argument('--algorithm', type=str, required=True, choices=['random_forest', 'lstm'])
    args = parser.parse_args()

    logging.info(f"--- MODE: infer (Web App) | ALGORITHM: {args.algorithm} ---")
    
    # --- Build Configuration ---
    if args.algorithm == 'random_forest':
        from config.config_ml_rf import param_rf_test
        base_config = param_rf_test.copy()
    elif args.algorithm == 'lstm':
        from config.config_ml_lstm import param_lstm_test
        base_config = param_lstm_test.copy()
    else:
        sys.exit(f"Algorithm {args.algorithm} not supported.")
        
    base_config.update(CONFIG_LOAD_ARTIFACTS)
    base_config.update(MQTT_CONFIG)
    base_config['paths'] = CONFIG_PATH['paths']
    
    # Set a finite number of steps for the web app demo
    base_config['inference_steps'] = base_config.get('inference_steps', 100)
    
    # Start preparation in a background thread
    preparation_thread = threading.Thread(target=prepare_inference, args=(base_config, args.algorithm), daemon=True)
    preparation_thread.start()
    
    # --- Create and run Flask App ---
    app = Flask(__name__, template_folder=os.path.join(project_root, 'ML_Algorithms', 'templates'))
    
    @app.route('/')
    def index():
        return render_template('dashboard.html')

    @app.route('/api/status')
    def get_status():
        status_data = PIPELINE_STATE.copy()
        if inference_processor:
            status_data['current_step'] = inference_processor.step_counter
        return jsonify(status_data)

    @app.route('/api/data')
    def get_data():
        step = request.args.get('step', type=int, default=0)
        
        if not inference_processor or step >= len(inference_processor.results_buffer):
            return jsonify({"status": "waiting"}), 200
        
        # Data is available, send it
        # The data format from results_buffer needs to be adapted for the frontend
        raw_data = inference_processor.results_buffer[step]
        
        # Assuming horizon=1 for simplicity, adapt if multi-step is needed for frontend
        pred_val = raw_data.get("prediction_step_1", None)

        output_data = {
            "date": raw_data["datetime"].isoformat(),
            "true_value": raw_data["true_value"],
            "predicted_value_step_1": pred_val,
            "cpu_load": raw_data["cpu_load_percent"],
            "inference_time_ms": raw_data["inference_time_ms"]
        }

        return jsonify({
            "status": "success",
            "data": output_data
        }), 200

    @app.route('/api/run_inference', methods=['POST'])
    def run_inference_endpoint():
        if PIPELINE_STATE.get("status") not in ["ready_for_inference", "finished"]:
            return jsonify({"error": "Not ready for inference."}), 400
        
        start_inference_flow(base_config)
        
        return jsonify({"status": "Inference process initiated."})

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)
    
    logging.info("\n🚀 Web server starting. Open http://127.0.0.1:5001 in your browser.")
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()