# ML_Helpfunctions/base_trainer.py

import logging
import sys
import os
import joblib
from abc import ABC, abstractmethod

from ML_Helpfunctions import pipeline_utils
# NEU: Import für die Typprüfung der Pipeline
from ML_Helpfunctions.Load_Prepare_Data import DataPipeline3D, DataPipeline2D


class BaseTrainer(ABC):
    """
    Abstrakte Basisklasse für Trainingspipelines.
    Kapselt die Logik für Experiment-Setup, Datenvorbereitung und Artefaktspeicherung.
    """
    def __init__(self, config: dict , folder_flag:str):
        self.config = config
        self.model = None
        self.scaler = None
        self.features = None
        self.train_time = 0.0
        self.folder_flag = folder_flag

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
        
        # Zustand aus der Pipeline übernehmen
        self.scaler = pipeline.scaler
        self.y_scaler = getattr(pipeline, 'y_scaler', None)

        # KORREKTUR: Die Logik zur Bestimmung der Feature-Liste wird vereinfacht.
        # Die Pipeline ist nun dafür verantwortlich, die korrekte und vollständig geordnete
        # Feature-Liste bereitzustellen (`pipeline.full_feature_list`).
        # Der BaseTrainer speichert diese Liste einfach ab.
        # Die Unterscheidung zwischen DataPipeline3D und 2D ist hier nicht mehr nötig.
        self.features = pipeline.full_feature_list
        logging.info(f"Die zu speichernde Feature-Liste für die Inferenz enthält {len(self.features)} Spalten.")
        logging.debug(f"Feature-Liste: {self.features}")


        if self.scaler is None or not self.features:
            logging.critical("CRITICAL: Data pipeline failed to generate scaler or feature list.")
            sys.exit(1)
        
        logging.info(f"Data preparation complete. Features: {len(self.features)}, X_train shape: {X_train.shape}")

        # Representative-Dataset-Quelle für INT8-Full quantization
        self._rep_source = X_train


        # --- SCHRITT 2: MODELLTRAINING ---
        logging.info("\nStep 2: Training model...")
        self._train_model(X_train, y_train)
        logging.info(f"✅ Model training completed in {self.train_time:.2f} seconds.")

        # --- SCHRITT 3: ARTEFAKTE SPEICHERN ---
        if save_artifacts:
            if not self.folder_flag:
                raise ValueError("Für das Speichern der Artefakte muss ein 'folder_flag' übergeben werden.")
            self._save_artifacts()
        else:
            logging.info("\nStep 3: Returning trained artifacts without saving.")
        
        logging.info("\n✅ Training pipeline finished successfully.")
        return self.model, self.scaler, self.y_scaler, self.features

    def _save_artifacts(self):
        """Speichert die trainierten Artefakte (Modell, Scaler, Features)."""
        logging.info("\nStep 3: Saving artifacts for inference...")
        mode = self.config.get("inference_mode", "load_artifacts_path")

        
        
        paths = self.config.get("paths")
        if not paths:
            logging.error("Pfade nicht in der Konfiguration gefunden. Artefakte können nicht gespeichert werden.")
            return

        if mode == 'load_artifacts_fast':
            # Diese Logik bleibt unverändert
            logging.info("Saving in 'fast' mode with static paths...")
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

            try:
                self.config["train_time_s"] = float(self.train_time)
            except Exception:
                pass
            # Hier wird jetzt der korrekte, bereits erstellte Pfad verwendet.
            logging.info("Saving in 'path' mode with versioned directory...")
            saver = pipeline_utils.ModelScalerSaver(self.config, paths)
            
            # --- ANGEPASSTE LOGIK ZUR STEUERUNG DER QUANTISIERUNG START ---
            rep_gen = None
            # Prüft, ob die Quantisierung in der Konfiguration aktiviert ist (Standard ist True)
            if self.config.get('quantization_enabled', True):
                logging.info("Quantization is enabled. Creating representative dataset for TFLite conversion.")
                saved_artifacts = None
                try:
                    rep_gen = pipeline_utils.create_representative_dataset_generator(getattr(self, '_rep_source', None), config=self.config)
                except TypeError:
                    # Fallback für ältere Signaturen (nur eine Positionals)
                    rep_gen = pipeline_utils.create_representative_dataset_generator(getattr(self, '_rep_source', None))
            else:
                logging.info("Quantization is disabled via config. Skipping representative dataset generation.")
            # --- ANGEPASSTE LOGIK ZUR STEUERUNG DER QUANTISIERUNG ENDE ---

            # Das (potenziell leere) rep_gen wird an die Speicherfunktion übergeben.
            # Diese Funktion muss intern damit umgehen können, um die Quantisierung zu überspringen.
            saved_artifacts = saver.save_artifacts(model=self.model, scaler=self.scaler, representative_dataset=rep_gen)
            
            # Speichere den dedizierten y_scaler, falls er existiert
            if getattr(self, 'y_scaler', None) is not None:
                y_path = os.path.join(paths["Scalers"], "y_scaler.joblib")
                joblib.dump(self.y_scaler, y_path)
                logging.info(f"Target y_scaler saved to: {y_path}")
            
            # Speichere die korrekte Feature-Liste
            try:
                features_path = os.path.join(paths.get("Models"), "features.joblib")
                joblib.dump(self.features, features_path)
                logging.info(f"Feature list saved to: {features_path}")
                if saved_artifacts is not None:
                    saved_artifacts["features_path"] = features_path
            except Exception as e:
                logging.error(f"Failed to save feature list: {e}")

            logging.info(f"All artifacts for run '{self.config['run_id']}' saved successfully.")
        
        else:
            logging.error(f"Unknown inference_mode '{mode}'. Artifacts not saved.")