# lstm_train_3d.py
import logging
import argparse
import sys
import os
import tensorflow as tf
import numpy as np
import joblib

# --- Project Path Setup ---
# Stellt sicher, dass das Hauptverzeichnis des Projekts im Python-Pfad ist
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
# Import der neuen 3D-Datenpipeline und der Hilfsfunktionen
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline3D, load_test_data_by_fraction
from ML_Helpfunctions import Pipeline_Utils
from ML_Helpfunctions import LSTM_Utils

# Import der Konfigurationen
from config.config_ml_lstm import param_lstm_test
# from config.config_general import CONFIG_PATH # Auskommentiert, falls nicht vorhanden

# Konfiguriere das Logging-Format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_training_pipeline_3d(config: dict):
    """
    Orchestriert die gesamte Trainingspipeline für ein 3D-LSTM-Modell.
    """
    logging.info("--- 🚀 Starting LSTM 3D Training Pipeline ---")

    # --- SCHRITT 1: EXPERIMENT-SETUP ---
    logging.info("\nStep 1: Setting up experiment directories...")
    # Erstellt Ordner für den Lauf und fügt run_id/timestamp zur Konfig hinzu
    config, paths = Pipeline_Utils.setup_experiment(config)

    # --- SCHRITT 2: DATENVORBEREITUNG (3D) ---
    logging.info("\nStep 2: Preparing 3D data using DataPipeline3D...")
    # Initialisiere die 3D-Datenpipeline
    pipeline_3d = DataPipeline3D(config)
    
    # Bereite Trainingsdaten vor (Laden, FE, Skalieren, Fenster bilden)
    X_train_3D, y_train_3D = pipeline_3d.prepare_training_data()
    
    # Bereite Testdaten mit den trainierten Scalern vor
    X_test_3D, y_test_3D = pipeline_3d.prepare_testing_data()
    
    # Hole die Artefakte (Scaler, Feature-Liste) aus der Pipeline-Instanz
    scaler_3D = pipeline_3d.scaler_3D
    y_scaler = pipeline_3d.y_scaler
    full_feature_list = pipeline_3d.full_feature_list
    
    # Lade den Test-DataFrame für die Evaluierungs-Zeitstempel
    test_df = load_test_data_by_fraction(config=config, train_fraction=config["train_fraction"], make_date_as_index=True)

    logging.info(f"Data preparation complete. Train shape: {X_train_3D.shape}, Test shape: {X_test_3D.shape}")
    logging.info(f"Number of features used for training: {len(full_feature_list)}")

    # --- SCHRITT 3: MODELLTRAINING ---
    logging.info("\nStep 3: Building and training the LSTM model...")
    model, history, train_time = LSTM_Utils.train_model_LSTM(
        config=config, 
        X_train=X_train_3D, 
        y_train=y_train_3D,
        features=full_feature_list
    )
    logging.info(f"✅ Model training completed in {train_time:.2f} seconds.")

    # --- SCHRITT 4: MODELL-EVALUIERUNG ---
    logging.info("\nStep 4: Evaluating the model on test data...")
    # Inferenz auf den skalierten Testdaten
    predictions_scaled = LSTM_Utils.run_inference_lstm(model, X_test_3D)

    # Rücktransformation der Vorhersagen und wahren Werte
    # Wichtig: y_scaler wird hier verwendet, da das Ziel skaliert wurde
    pred_orig = Pipeline_Utils.safe_inverse_transform(y_scaler, predictions_scaled)
    true_orig = Pipeline_Utils.safe_inverse_transform(y_scaler, y_test_3D)
    
    # Zeitstempel für die Vorhersagen extrahieren
    # Der Index muss um die Lags und den Horizont angepasst werden, um zu den Vorhersagen zu passen
    num_predictions = len(pred_orig)
    start_index = config["lags"] + config["horizon"] -1
    dates = test_df.index[start_index : start_index + num_predictions]

    # Berechne Metriken
    metrics = Pipeline_Utils.evaluate_all_metrics(
        y_true=true_orig,
        y_pred=pred_orig,
        horizon=config["horizon"]
    )
    logging.info(f"Evaluation Metrics: {metrics}")

    # --- SCHRITT 5: ARTEFAKTE SPEICHERN ---
    logging.info("\nStep 5: Saving all model artifacts...")
    # Initialisiere den Saver für Modelle und Scaler
    saver = Pipeline_Utils.ModelScalerSaver(config, paths)
    
    # Erstelle ein repräsentatives Dataset für die TFLite-Quantisierung
    representative_dataset_gen = list(tf.data.Dataset.from_tensor_slices(X_train_3D).batch(1).take(100))
    
    # Speichere Modell (.keras, .tflite), Scaler, Plots etc.
    deployment_artifacts = saver.save_artifacts(
        model=model,
        scaler=scaler_3D, # Der Haupt-Feature-Scaler
        history=history,
        representative_dataset=representative_dataset_gen
    )
    logging.info(f"Deployment artifacts saved: {deployment_artifacts}")

    # Speichere Metriken und Vorhersagedatei
    metrics_results = LSTM_Utils.save_lstm_metrics_results(
        config=config,
        pred_orig=pred_orig,
        true_orig=true_orig,
        dates=dates,
        metrics=metrics,
        paths=paths,
        power_time=train_time
    )

    try:
        features_path = os.path.join(paths.get("Models"), "features.joblib")
        joblib.dump(full_feature_list, features_path)
        logging.info(f"✅ Feature-Liste gespeichert unter: {features_path}")
        deployment_artifacts["features_path"] = features_path
    except Exception as e:
        logging.error(f"Fehler beim Speichern der Feature-Liste: {e}")

    logging.info(f"Deployment artifacts saved: {deployment_artifacts}")
    logging.info(f"Metrics and prediction data saved: {metrics_results}")
    
    logging.info("\n--- ✅ LSTM 3D Training Pipeline Finished Successfully ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 3D LSTM Training Pipeline.")
    args = parser.parse_args()

    # --- Lade die Konfigurationen ---
    # 1. Importiere die allgemeine Basiskonfiguration (enthält die korrekten Pfade)
    from config.config_general import CONFIG_PATH
    # 2. Importiere die modellspezifischen LSTM-Parameter
    from config.config_ml_lstm import param_lstm_test

    logging.info("Merging general and model-specific configurations...")

    # --- Intelligente Verschmelzung der Konfigurationen ---
    # Starte mit einer Kopie der modellspezifischen Parameter
    training_config = param_lstm_test.copy()

    # Hole die allgemeinen Pfade und die spezifischen Pfade
    general_paths = CONFIG_PATH.get("paths", {})
    model_specific_paths = training_config.get("paths", {})

    # Verschmelze die Pfad-Wörterbücher.
    # Allgemeine Pfade dienen als Basis. Spezifische Pfade können sie überschreiben.
    # Wichtig: Der Schlüssel 'input' aus CONFIG_PATH bleibt so erhalten.
    merged_paths = {**general_paths, **model_specific_paths}

    # Setze das vollständig verschmolzene Pfad-Wörterbuch in der finalen Konfiguration
    training_config["paths"] = merged_paths

    # Debug-Ausgabe zur Überprüfung der finalen Pfade
    logging.info(f"Final paths for experiment: {training_config['paths']}")
    
    # Führe die gesamte Pipeline mit der korrekten Konfiguration aus
    run_training_pipeline_3d(config=training_config)