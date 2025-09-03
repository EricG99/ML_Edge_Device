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

        self.features = pipeline.full_feature_list
        logging.info(f"Die zu speichernde Feature-Liste für die Inferenz enthält {len(self.features)} Spalten.")
        logging.debug(f"Feature-Liste: {self.features}")

        # --- KORRIGIERTE PRÜFUNG ---
        # Prüft nur auf einen Feature-Scaler, wenn die Skalierung aktiviert war.
        # Die Feature-Liste muss aber immer vorhanden sein.
        if (self.config.get('scale_other_features', False) and self.scaler is None) or not self.features:
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

        # Kleine Helfer zum Aktualisieren der training_config.json um modellgrößen
        def _update_model_size(models_dir: str, filename: str) -> None:
            import json, os
            cfg_path = os.path.join(models_dir, "training_config.json")
            data = {}
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh) or {}
                except Exception:
                    data = {}
            sizes = data.get("model_sizes_mb", {})
            try:
                size_mb = round(os.path.getsize(os.path.join(models_dir, filename)) / (1024 * 1024), 6)
                sizes[filename] = size_mb
                data["model_sizes_mb"] = sizes
                with open(cfg_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
                logging.info(f"Model size recorded in training_config.json: {filename} = {size_mb} MB")
            except Exception as e:
                logging.warning(f"Could not record model size for {filename}: {e}")

        if mode == 'load_artifacts_fast':
            # Statische (legacy) Speicherpfade
            logging.info("Saving in 'fast' mode with static paths...")
            static_paths = {
                "scaler": self.config.get("scaler_path_static", "scaler.joblib"),
                "features": self.config.get("features_path_static", "features.joblib"),
                "model": self.config.get("model_path_static", "model.joblib"),
                "y_scaler": self.config.get("y_scaler_path_static", "y_scaler.joblib"),
            }
            try:
                joblib.dump(self.scaler, static_paths["scaler"])
                joblib.dump(self.features, static_paths["features"])
                joblib.dump(self.model, static_paths["model"])
                if getattr(self, "y_scaler", None) is not None:
                    joblib.dump(self.y_scaler, static_paths["y_scaler"])
                logging.info(f"Artifacts saved to static paths: {static_paths}")
            except Exception as e:
                logging.error(f"Failed to save artifacts in fast mode: {e}")
            return

        elif mode == 'load_artifacts_path':
            # Train-Zeit optional mitgeben
            try:
                self.config["train_time_s"] = float(self.train_time)
            except Exception:
                pass

            logging.info("Saving in 'path' mode with versioned directory...")

            # 1) Zentrale Saver-Logik (schreibt DL/TFLite/etc.)
            saver = pipeline_utils.ModelScalerSaver(self.config, paths)

            # Repräsentatives Dataset für Quantisierung nur erstellen, wenn benötigt
            rep_gen = None
            try:
                # Bevorzugte Signatur mit config
                rep_gen = pipeline_utils.create_representative_dataset_generator(
                    getattr(self, '_rep_source', None),
                    config=self.config
                )
            except TypeError:
                # Fallback auf alte Signatur ohne config
                try:
                    rep_gen = pipeline_utils.create_representative_dataset_generator(
                        getattr(self, '_rep_source', None)
                    )
                except Exception:
                    rep_gen = None
            except Exception:
                rep_gen = None

            saved_artifacts = None
            try:
                saved_artifacts = saver.save_artifacts(
                    model=self.model,
                    scaler=self.scaler,
                    representative_dataset=rep_gen
                )
            except Exception as e:
                logging.warning(f"Primary saver.save_artifacts failed or produced no model file: {e}")

            # 2) y_scaler separat speichern (falls vorhanden)
            try:
                if getattr(self, "y_scaler", None) is not None:
                    y_scaler_path = os.path.join(paths.get("Scalers"), "y_scaler.joblib")
                    os.makedirs(paths.get("Scalers"), exist_ok=True)
                    joblib.dump(self.y_scaler, y_scaler_path)
                    logging.info(f"Target y_scaler saved to: {y_scaler_path}")
                    if saved_artifacts is not None:
                        saved_artifacts["y_scaler_path"] = y_scaler_path
            except Exception as e:
                logging.warning(f"Failed to save y_scaler: {e}")

            # 3) Fallback: Wenn KEINE Modell-Datei existiert, generisch als model.joblib speichern (Sklearn etc.)
            models_dir = paths.get("Models")
            os.makedirs(models_dir, exist_ok=True)

            # Kandidaten, die die Orchestrierung als "gültiges Modell" erkennt
            candidate_files = [
                "model.keras",
                "model.json",  # z.B. LightGBM JSON
                "model.joblib",
                "model_quant_float16.tflite",
                "model_quant_int8.tflite",
                "model_quant_int8_full.tflite",
            ]
            has_model_blob = any(os.path.exists(os.path.join(models_dir, f)) for f in candidate_files)

            if not has_model_blob:
                # Universeller Dump für sklearn & Co.
                try:
                    out_path = os.path.join(models_dir, "model.joblib")
                    joblib.dump(self.model, out_path)
                    logging.info(f"Fallback: sklearn-like model saved to {out_path}")
                    _update_model_size(models_dir, "model.joblib")
                except Exception as e:
                    logging.warning(f"Fallback-Save for sklearn-like model failed: {e}")
            else:
                # Wenn ein Modell bereits existiert, soweit möglich dessen Größe nachtragen
                for fname in candidate_files:
                    fpath = os.path.join(models_dir, fname)
                    if os.path.exists(fpath):
                        _update_model_size(models_dir, fname)

            # 4) Features immer speichern
            try:
                features_path = os.path.join(models_dir, "features.joblib")
                joblib.dump(self.features, features_path)
                logging.info(f"Feature list saved to: {features_path}")
                if saved_artifacts is not None:
                    saved_artifacts["features_path"] = features_path
            except Exception as e:
                logging.error(f"Failed to save feature list: {e}")

            logging.info(f"All artifacts for run '{self.config['run_id']}' saved successfully.")
            return

        else:
            logging.error(f"Unknown inference_mode '{mode}'. Artifacts not saved.")
            return