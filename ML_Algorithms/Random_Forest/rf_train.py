# rf_train.py
import logging
import argparse
import joblib
import sys
import os

# --- Project Path Setup ---
# Stellt sicher, dass die Projekt-Root im Python-Pfad ist, um die Module zu finden
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
# NEU: Direkter Import der DataPipeline2D-Klasse
from ML_Helpfunctions.base_trainer import BaseTrainer
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline2D
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
from ML_Helpfunctions import RF_Utils as RFUtils

from config.config_ml_rf import param_rf_test
from config.config_general import CONFIG_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
FOLDER_FLAG = "RandomForest"


class RandomForestTrainer(BaseTrainer):
    """Spezialisierte Trainingsklasse für Random Forest Modelle."""
    
    def _setup_pipeline(self):
        """Initialisiert die 2D-Datenpipeline."""
        return DataPipeline2D(self.config)

    def _train_model(self, X_train, y_train):
        """Trainiert das Random Forest Modell."""
        self.model, self.train_time = RFUtils.train_random_forest_model(
            config=self.config,
            X_train=X_train,
            y_train=y_train,
            features=self.features
        )

# def run_training(config: dict, save_artifacts: bool = True):
#     """
#     Führt die Random Forest-Trainingspipeline unter Verwendung der DataPipeline2D-Klasse aus.

#     - "load_artifacts_fast": Speichert die drei Haupt-Artefakte (Modell, Scaler, Features)
#       unter statischen, vordefinierten Namen für schnelles Testen.
#     - "load_artifacts_path": Erstellt einen versionierten Ordner für den Trainingslauf
#       und speichert dort alle detaillierten Ergebnisse und Artefakte.
#     """
#     logging.info("--- Starting Random Forest Training Pipeline ---")

#     # --- SCHRITT 1: DATENVORBEREITUNG MIT DataPipeline2D ---
#     logging.info("\nStep 1: Preparing training data using DataPipeline2D...")
    
#     # Instanziiere die Pipeline mit der Konfiguration
#     pipeline = DataPipeline2D(config)
    
#     # Führe die Trainingsdaten-Vorbereitung aus
#     # Diese Methode führt Laden, Feature Engineering und Skalierung intern durch
#     X_train, y_train = pipeline.prepare_training_data()

#     # Hole die trainierten Artefakte direkt aus der Pipeline-Instanz
#     scaler = pipeline.scaler
#     final_feature_list = pipeline.full_feature_list
    
#     # Überprüfe, ob die Pipeline die Artefakte erfolgreich erstellt hat
#     if scaler is None or not final_feature_list:
#         logging.critical("CRITICAL: Data pipeline failed to generate scaler or feature list.")
#         sys.exit(1)
        
#     logging.info("Data preparation complete.")
#     logging.info(f"Anzahl der für das Training verwendeten Features: {len(final_feature_list)}")
#     logging.info(f"Shape of X_train: {X_train.shape}, Shape of y_train: {y_train.shape}")


#     # --- SCHRITT 2: MODELLTRAINING (unverändert) ---
#     logging.info("\nStep 2: Training Random Forest model...")
#     rf_model, train_time = RFUtils.train_random_forest_model(
#         config=config, X_train=X_train, y_train=y_train,
#         features=final_feature_list 
#     )
#     logging.info(f"Model trained in {train_time:.2f} seconds.")

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
            
#             joblib.dump(scaler, scaler_path)
#             joblib.dump(final_feature_list, features_path)
#             joblib.dump(rf_model, model_path)
            
#             logging.info(f"Artifacts saved to static paths: {model_path}, {scaler_path}, {features_path}")

#         # MODUS 2: Detailliertes Speichern in versioniertem Ordner
#         elif mode == 'load_artifacts_path':
#             logging.info("Saving in 'path' mode with versioned directory...")
            
#             # Erstellt die Ordnerstruktur (z.B. Output/run_20250718_...)
#             config, paths = PipelineUtils.setup_experiment(config)
            
#             # Die Hilfsklasse aus Pipeline_Utils kümmert sich um das Speichern
#             saver = PipelineUtils.ModelScalerSaver(config, paths)
            
#             # Speichere Modell und Scaler über die Klasse
#             saved_artifacts = saver.save_artifacts(model=rf_model, scaler=scaler)
            
#             # Speichere die Feature-Liste manuell im selben Ordner
#             try:
#                 features_path = os.path.join(paths.get("Models"), "features.joblib")
#                 joblib.dump(final_feature_list, features_path)
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
#         return rf_model, scaler, final_feature_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Random Forest Model Trainer")
    # Hier könnten Argumente für Konfigurationsdateien etc. hinzugefügt werden
    args = parser.parse_args()

    # --- Basiskonfiguration (unverändert) ---
    training_config = {**CONFIG_PATH, **param_rf_test}
    training_config.update({
        "dataset": "train_data_sample.csv",
        "scaler_path": "trained_rf_scaler.joblib",
        "features_path": "trained_rf_features.joblib",
        "model_path": "trained_rf_model.joblib",
        # Wichtiger Schlüssel für die DataPipeline2D
        "loading_strategy": "live_mqtt" #"split", "separate_csv", "live_mqtt"
    })
    
    # Führe das Training aus und speichere die Artefakte standardmäßig
    trainer = RandomForestTrainer(config=training_config)
    trainer.run(save_artifacts=True)  
    # run_training(config=training_config, save_artifacts=True)
    logging.info("\n✅ Training complete. Artifacts have been saved.")