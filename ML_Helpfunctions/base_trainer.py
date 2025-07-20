# ML_Helpfunctions/base_trainer.py
import logging
import sys
import os
import joblib
from abc import ABC, abstractmethod

from ML_Helpfunctions import Pipeline_Utils

class BaseTrainer(ABC):
    """
    Abstrakte Basisklasse für Trainingspipelines.
    Kapselt die Logik für Experiment-Setup, Datenvorbereitung und Artefaktspeicherung.
    """
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.scaler = None
        self.features = None
        self.train_time = 0.0

    @abstractmethod
    def _setup_pipeline(self):
        """Muss von der Subklasse implementiert werden, um die spezifische Datenpipeline zu initialisieren."""
        pass

    @abstractmethod
    def _train_model(self, X_train, y_train):
        """Muss von der Subklasse implementiert werden, um das spezifische Modell zu trainieren."""
        pass

    def run(self, save_artifacts: bool = True):
        """Führt die gesamte Trainingspipeline aus."""
        logging.info(f"--- 🚀 Starting {self.config.get('model_name', 'Unknown')} Training Pipeline ---")

        # --- SCHRITT 1: DATENVORBEREITUNG ---
        logging.info("\nStep 1: Preparing training data...")
        pipeline = self._setup_pipeline()
        X_train, y_train = pipeline.prepare_training_data()
        
        self.scaler = pipeline.scaler
        self.features = pipeline.full_feature_list

        if self.scaler is None or not self.features:
            logging.critical("CRITICAL: Data pipeline failed to generate scaler or feature list.")
            sys.exit(1)
        
        logging.info(f"Data preparation complete. Features: {len(self.features)}, X_train shape: {X_train.shape}")

        # --- SCHRITT 2: MODELLTRAINING ---
        logging.info("\nStep 2: Training model...")
        self._train_model(X_train, y_train)
        logging.info(f"✅ Model training completed in {self.train_time:.2f} seconds.")

        # --- SCHRITT 3: ARTEFAKTE SPEICHERN ---
        if save_artifacts:
            self._save_artifacts()
        else:
            logging.info("\nStep 3: Returning trained artifacts without saving.")
        
        logging.info("\n✅ Training pipeline finished successfully.")
        return self.model, self.scaler, self.features

    def _save_artifacts(self):
        """Speichert die trainierten Artefakte (Modell, Scaler, Features)."""
        logging.info("\nStep 3: Saving artifacts for inference...")
        mode = self.config.get("inference_mode", "load_artifacts_fast")
        
        # Erstellt Ordnerstruktur für 'path' mode
        self.config, paths = Pipeline_Utils.setup_experiment(self.config)
        
        if mode == 'load_artifacts_fast':
            logging.info("Saving in 'fast' mode with static paths...")
            # Statische Pfade werden direkt aus der Config gelesen
            static_paths = {
                "scaler": self.config.get("scaler_path_static", "scaler.joblib"),
                "features": self.config.get("features_path_static", "features.joblib"),
                "model": self.config.get("model_path_static", "model.joblib")
            }
            joblib.dump(self.scaler, static_paths["scaler"])
            joblib.dump(self.features, static_paths["features"])
            joblib.dump(self.model, static_paths["model"])
            logging.info(f"Artifacts saved to static paths: {static_paths}")

        elif mode == 'load_artifacts_path':
            logging.info("Saving in 'path' mode with versioned directory...")
            saver = Pipeline_Utils.ModelScalerSaver(self.config, paths)
            saved_artifacts = saver.save_artifacts(model=self.model, scaler=self.scaler)
            
            # Speichere die Feature-Liste manuell
            try:
                features_path = os.path.join(paths.get("Models"), "features.joblib")
                joblib.dump(self.features, features_path)
                logging.info(f"Feature list saved to: {features_path}")
                saved_artifacts["features_path"] = features_path
            except Exception as e:
                logging.error(f"Failed to save feature list: {e}")
            logging.info(f"All artifacts for run '{self.config['run_id']}' saved successfully.")
        
        else:
            logging.error(f"Unknown inference_mode '{mode}'. Artifacts not saved.")