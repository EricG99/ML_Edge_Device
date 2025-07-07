# In Ihrer Datei: ML_Helpfunctions/Pipeline_Utils.py

import os
import json
import joblib
import traceback
import numpy as np
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt

class ModelScalerSaver:
    """
    Eine Klasse zur Kapselung der gesamten Logik zum Speichern von Modellen
    und zugehörigen Artefakten.
    """
    def __init__(self, config: dict, paths: dict):
        """
        Initialisiert den Saver mit der Konfiguration und den Pfaden für einen Lauf.
        """
        self.config = config
        self.paths = paths
        if not paths:
            raise ValueError("Das 'paths'-Dictionary darf nicht None sein.")
        
        # Stelle sicher, dass die Basis-Verzeichnisse existieren
        self._ensure_output_dirs_exist(["Models", "Scalers", "Model_Structures", "Loss_Plots"])

    def save_model_scaler(self, model, scaler, **kwargs) -> dict:
        """
        Hauptmethode zum Speichern. Erkennt den Modelltyp und delegiert.
        
        Args:
            model: Das trainierte Modellobjekt.
            scaler: Das zugehörige Skalierer-Objekt.
            **kwargs: Optionale Argumente wie 'history' und 'representative_dataset'.
            
        Returns:
            Ein Dictionary mit den Pfaden zu allen gespeicherten Artefakten.
        """
        print(f"--- 🚀 Starte Speichern der Deployment-Artefakte für Modell: {self.config.get('model_name')} ---")
        
        results = {}

        # 1. Skalierer speichern
        scaler_path = self._save_scaler(scaler)
        if scaler_path:
            results["scaler_path"] = scaler_path
            
        # 2. Modellspezifische Artefakte speichern (Smart Dispatch)
        model_results = {}
        if isinstance(model, tf.keras.Model):
            model_results = self._save_keras_model(model, **kwargs)
        elif isinstance(model, xgb.XGBRegressor):
            model_results = self._save_xgboost_model(model)
        elif isinstance(model, (RandomForestRegressor, MultiOutputRegressor)):
            model_results = self._save_sklearn_model(model)
        else:
            print(f"⚠️ Warnung: Kein spezifischer Speicherpfad für Modelltyp {type(model).__name__} implementiert.")
            
        results.update(model_results)

        # 3. Edge-Artefakte speichern (falls konfiguriert)
        edge_artifacts_path = self._save_edge_artifacts()
        if edge_artifacts_path:
            results["edge_artifacts"] = edge_artifacts_path
            
        return results

    def _ensure_output_dirs_exist(self, dir_keys: list):
        """Stellt sicher, dass alle benötigten Ausgabe-Verzeichnisse existieren."""
        try:
            for key in dir_keys:
                dir_path = self.paths.get(key)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(f"❌ Fehler beim Erstellen von Verzeichnissen: {e}")

    def _save_scaler(scaler, config: dict, paths: dict) -> str:
        """
        Speichert ein Skalierer-Objekt, aber nur, wenn laut Konfiguration
        eine Skalierung stattgefunden hat und ein Skalierer-Objekt existiert.
        """
        scaler_path = None # Wichtig: Initialisieren für den Fehlerfall
        # Prüft beide Skalierungs-Flags in der Konfiguration
        scale_target_status = config.get("scale_target", "Nicht in Config gefunden")
        print(f"[Info] Konfiguration 'scale_target': {scale_target_status}")

        should_scale = config.get("scale_target", False)
        
        if should_scale and scaler is not None:
            try:
                scaler_dir = paths.get("Scalers")
                scaler_filename = f"scaler_{config.get('run_id')}_{config.get('time_stamp')}.joblib"
                scaler_path = os.path.join(scaler_dir, scaler_filename)
                joblib.dump(scaler, scaler_path)
                print(f"✅ Skalierer gespeichert unter: {scaler_path}")
            except Exception as e:
                print(f"⚠️ Fehler beim Speichern des Skalierers: {e}")
                print(traceback.format_exc())
        elif should_scale and scaler is None:
            print("⚠️ Warnung: Skalierung war konfiguriert, aber es wurde kein Skalierer-Objekt übergeben.")
        return scaler_path
    
    def save_loss_plot(history: dict, config: dict, paths: dict, output_path: str):
        """
        Erstellt und speichert einen Plot des Trainings- und Validierungsverlusts
        sowie der Metriken aus dem Keras-History-Objekt.

        Args:
            history (dict): Das History-Objekt von model.fit().
            config (dict): Das Konfigurationsdictionary für Titelinformationen etc.
            paths (dict): Das Pfad-Dictionary.
            output_path (str): Der vollständige Pfad zum Speichern des Plots.
        """
        if not hasattr(history, 'history') or not history.history:
            print("⚠️ Kein gültiges History-Objekt zum Plotten vorhanden.")
            return

        history_dict = history.history
        
        # Schlüssel für Loss und die erste Metrik dynamisch finden
        loss_keys = sorted([k for k in history_dict if 'loss' in k])
        metric_keys = sorted([k for k in history_dict if k not in loss_keys])
        
        # Erstelle ein 2x1 Subplot-Grid, falls Metriken vorhanden sind, sonst nur 1x1
        num_subplots = 2 if metric_keys else 1
        fig, axes = plt.subplots(num_subplots, 1, figsize=(12, 6 * num_subplots), sharex=True)
        
        # Sicherstellen, dass 'axes' immer ein Array ist, auch bei nur einem Subplot
        if num_subplots == 1:
            axes = [axes]

        # --- Subplot 1: Loss ---
        for key in loss_keys:
            axes[0].plot(history_dict[key], label=key)
        axes[0].set_title(f'Trainings- & Validierungs-Loss für {config.get("model_name")}')
        # Extrahiert den Loss-Namen aus der Konfiguration
        loss_name = str(config.get("loss")).split('.')[-1].replace("()", "")
        axes[0].set_ylabel(f'Loss ({loss_name})')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_yscale('log') # Log-Skala ist oft hilfreich für Loss-Plots

        # --- Subplot 2: Metriken (falls vorhanden) ---
        if num_subplots > 1:
            primary_metric_name = metric_keys[0].replace('val_', '') # z.B. 'mae'
            for key in metric_keys:
                axes[1].plot(history_dict[key], label=key)
            axes[1].set_title('Trainings- & Validierungs-Metrik')
            axes[1].set_ylabel(primary_metric_name.upper())
            axes[1].set_xlabel('Epoche')
            axes[1].legend()
            axes[1].grid(True)
        else:
            axes[0].set_xlabel('Epoche')

        fig.suptitle(f'Trainingsverlauf für Run: {config.get("run_id")}', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Platz für den suptitle lassen
        
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            print(f"✅ Loss Plot gespeichert unter: {output_path}")
        except Exception as e:
            print(f"❌ FEHLER beim Speichern des Loss Plots unter '{output_path}': {e}")
        finally:
            plt.close(fig) # Schließt die Figur, um Speicher freizugeben

    def _save_keras_model(self, model: tf.keras.Model, **kwargs) -> dict:
        """Speichert ein Keras-Modell und zugehörige Artefakte (Plots, TFLite)."""
        results = {}
        history = kwargs.get("history")
        
        # Dateinamen-Komponenten
        model_name = self.config.get("model_name", "keras_model")
        dataset = self.config.get("dataset", "data")
        run_id = self.config.get("run_id", "run")
        timestamp = self.config.get("time_stamp", "ts")
        base_filename = f"{model_name}_{dataset}_{run_id}_{timestamp}"

        # Normales Keras-Modell speichern
        try:
            model_dir = self.paths.get("Models")
            model_path = os.path.join(model_dir, f"{base_filename}.keras")
            model.save(model_path)
            results["model_path"] = model_path
            print(f"✅ Normales Keras-Modell gespeichert unter: {model_path}")
        except Exception as e:
            print(f"❌ Fehler beim Speichern des Keras-Modells: {e}", exc_info=True)

        # Quantisiertes TFLite-Modell speichern (NUR wenn Flag gesetzt ist)
        if self.config.get("edge_device", False):
            try:
                tflite_path = os.path.join(self.paths.get("Models"), f"{base_filename}.tflite")
                representative_dataset = kwargs.get("representative_dataset")
                
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset
                
                # Fix für LSTM-Modelle
                if any(isinstance(layer, (tf.keras.layers.LSTM, tf.keras.layers.GRU)) for layer in model.layers):
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
                    converter._experimental_lower_tensor_list_ops = False

                tflite_model_quant = converter.convert()
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_model_quant)
                results["quantized_model_path"] = tflite_path
                print(f"✅ TFLite-Modell (quantisiert) gespeichert unter: {tflite_path}")
            except Exception as e:
                print(f"❌ Fehler bei der TFLite-Quantisierung: {e}", exc_info=True)

        # Plots speichern
        if history:
            try:
                plot_dir = self.paths.get("Loss_Plots")
                loss_plot_path = os.path.join(plot_dir, f"loss_plot_{run_id}_{timestamp}.png")

                save_loss_plot(history, self.config, self.paths, loss_plot_path)
                results["loss_plot_path"] = loss_plot_path
            except Exception as e:
                print(f"❌ Fehler beim Speichern des Loss Plots: {e}", exc_info=True)

        try:
            structure_dir = self.paths.get("Model_Structures")
            structure_path = os.path.join(structure_dir, f"structure_{run_id}_{timestamp}.png")
            tf.keras.utils.plot_model(model, to_file=structure_path, show_shapes=True, show_layer_activations=True)
            results["model_structure_path"] = structure_path
            print(f"📊 Modellstruktur gespeichert unter: {structure_path}")
        except Exception as e:
            print(f"❌ Fehler beim Speichern der Modellstruktur: {e}", exc_info=True)
            
        return results
    


    def _save_sklearn_model(self, model) -> dict:
        """Speichert ein Scikit-learn-Modell."""
        try:
            model_dir = self.paths.get("Models")
            model_name = self.config.get("model_name", "sklearn_model")
            dataset = self.config.get("dataset", "data")
            model_filename = f"{model_name}_{dataset}_{self.config['run_id']}_{self.config['time_stamp']}.joblib"
            model_path = os.path.join(model_dir, model_filename)
            joblib.dump(model, model_path, compress=3)
            print(f"📤 Scikit-learn-Modell gespeichert unter: {model_path}")
            return {"model_path": model_path}
        except Exception as e:
            print(f"❌ Fehler beim Speichern des Scikit-learn-Modells: {e}", exc_info=True)
            return {}

    def _save_xgboost_model(self, model: xgb.XGBRegressor) -> dict:
        """Speichert ein XGBoost-Modell."""
        try:
            model_dir = self.paths.get("Models")
            model_name = self.config.get("model_name", "xgb_model")
            dataset = self.config.get("dataset", "data")
            model_filename = f"{model_name}_{dataset}_{self.config['run_id']}_{self.config['time_stamp']}.json"
            model_path = os.path.join(model_dir, model_filename)
            model.save_model(model_path)
            print(f"📤 XGBoost-Modell gespeichert unter: {model_path}")
            return {"model_path": model_path}
        except Exception as e:
            print(f"❌ Fehler beim Speichern des XGBoost-Modells: {e}", exc_info=True)
            return {}
            
    def _save_edge_artifacts(self) -> str:
        """Speichert optionale Artefakte für die Edge-Bereitstellung."""
        edge_dir_path = None
        if not self.config.get("enable_edge", False):
            return edge_dir_path
        # ... Logik zum Speichern von scaler_mean.npy etc. ...
        return edge_dir_path
    
