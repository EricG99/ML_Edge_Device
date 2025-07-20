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
from ML_Helpfunctions.base_trainer import BaseTrainer


# Import der Konfigurationen
from config.config_ml_lstm import param_lstm_test
# from config.config_general import CONFIG_PATH # Auskommentiert, falls nicht vorhanden

# Konfiguriere das Logging-Format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LSTMTrainer(BaseTrainer):
    """Spezialisierte Trainingsklasse für LSTM Modelle."""
    
    def _setup_pipeline(self):
        """Initialisiert die 3D-Datenpipeline."""
        return DataPipeline3D(self.config)

    def _train_model(self, X_train, y_train):
        """Trainiert das LSTM Modell."""
        # Hinweis: Das Speichern der Artefakte für Keras-Modelle (history, etc.)
        # könnte hier noch verfeinert werden, indem die `history` zurückgegeben wird.
        self.model, history, self.train_time = LSTM_Utils.train_model_LSTM(
            config=self.config,
            X_train=X_train,
            y_train=y_train,
            features=self.features
        )

# def run_training_pipeline_3d(config: dict, save_artifacts: bool = True):
#     """
#     Orchestriert die gesamte Trainingspipeline für ein 3D-LSTM-Modell.
#     """
#     logging.info("--- 🚀 Starting LSTM 3D Training Pipeline ---")

#     # --- SCHRITT 1: EXPERIMENT-SETUP ---
#     logging.info("\nStep 1: Setting up experiment directories...")
#     # Erstellt Ordner für den Lauf und fügt run_id/timestamp zur Konfig hinzu
#     config, paths = Pipeline_Utils.setup_experiment(config)

#     # --- SCHRITT 2: DATENVORBEREITUNG (3D) ---
#     logging.info("\nStep 2: Preparing 3D data using DataPipeline3D...")
#     # Initialisiere die 3D-Datenpipeline
#     pipeline_3d = DataPipeline3D(config)
    
#     # Bereite Trainingsdaten vor (Laden, FE, Skalieren, Fenster bilden)
#     X_train_3D, y_train_3D = pipeline_3d.prepare_training_data()
    
#     # Bereite Testdaten mit den trainierten Scalern vor
    
#     # Hole die Artefakte (Scaler, Feature-Liste) aus der Pipeline-Instanz
#     scaler_3D = pipeline_3d.scaler
#     y_scaler = pipeline_3d.y_scaler
#     full_feature_list = pipeline_3d.full_feature_list
    
#        # Überprüfe, ob die Pipeline die Artefakte erfolgreich erstellt hat
#     if scaler_3D is None or not full_feature_list:
#         logging.critical("CRITICAL: Data pipeline failed to generate scaler or feature list.")
#         sys.exit(1)
        
#     logging.info("Data preparation complete.")


#     # --- SCHRITT 3: MODELLTRAINING ---
#     logging.info("\nStep 3: Building and training the LSTM model...")
#     model, history, train_time = LSTM_Utils.train_model_LSTM(
#         config=config, 
#         X_train=X_train_3D, 
#         y_train=y_train_3D,
#         features=full_feature_list
#     )
#     logging.info(f"✅ Model training completed in {train_time:.2f} seconds.")

#     # --- SCHRITT 3: ARTEFAKTE SPEICHERN (unverändert, nutzt die neuen Variablen) ---
#     if save_artifacts:
#         logging.info("\nStep 3: Saving artifacts for inference...")
#         mode = config.get("inference_mode", "load_artifacts_fast")
        
#         # MODUS 1: Schnelles Speichern mit statischen Namen
#         if mode == 'load_artifacts_fast':
#             logging.info("Saving in 'fast' mode with static paths...")
#             scaler_path = config.get("scaler_path_static", "trained_rf_scaler.joblib")
#             features_path = config.get("features_path_static", "trained_rf_features.joblib")
#             model_path = config.get("model_path_static", "trained_rf_model.joblib")
            
#             joblib.dump(scaler_3D, scaler_path)
#             joblib.dump(full_feature_list, features_path)
#             joblib.dump(model, model_path)
            
#             logging.info(f"Artifacts saved to static paths: {model_path}, {scaler_path}, {features_path}")

#         # MODUS 2: Detailliertes Speichern in versioniertem Ordner
#         elif mode == 'load_artifacts_path':
#             logging.info("Saving in 'path' mode with versioned directory...")
            
#             # Erstellt die Ordnerstruktur (z.B. Output/run_20250718_...)
#             config, paths = Pipeline_Utils.setup_experiment(config)
            
#             # Die Hilfsklasse aus Pipeline_Utils kümmert sich um das Speichern
#             saver = Pipeline_Utils.ModelScalerSaver(config, paths)
            
#             # Speichere Modell und Scaler über die Klasse
#             saved_artifacts = saver.save_artifacts(model=model, scaler=scaler_3D)
            
#             # Speichere die Feature-Liste manuell im selben Ordner
#             try:
#                 features_path = os.path.join(paths.get("Models"), "features.joblib")
#                 joblib.dump(full_feature_list, features_path)
#                 logging.info(f"Feature list saved to: {features_path}")
#                 saved_artifacts["features_path"] = features_path
#             except Exception as e:
#                 logging.error(f"Failed to save feature list: {e}")

#             logging.info(f"All artifacts for run '{config['run_id']}' saved successfully.")
        
#         else:
#             logging.error(f"Unknown inference_mode '{mode}'. Artifacts not saved.")
#             return False

#         return True
    
#     else: # Falls save_artifacts == False
#         logging.info("\nStep 3: Returning trained artifacts without saving.")
#         return model, scaler_3D, full_feature_list


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
    training_config["inference_mode"] = "load_artifacts_path" # "load_artifacts_fast" oder "load_artifacts_path"


    # Debug-Ausgabe zur Überprüfung der finalen Pfade
    logging.info(f"Final paths for experiment: {training_config['paths']}")
    
    # Führe die gesamte Pipeline mit der korrekten Konfiguration aus
    # run_training_pipeline_3d(config=training_config)
    trainer = LSTMTrainer(config=training_config)
    trainer.run(save_artifacts=True)