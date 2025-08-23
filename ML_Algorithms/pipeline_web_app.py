# pipeline_web_app.py

from html import parser
import time
import logging
import argparse
import sys
import threading
import os
import pandas as pd
import numpy as np
from flask import Flask, jsonify, render_template, request
from copy import deepcopy
from datetime import datetime, timedelta
import tensorflow as tf

import webbrowser  # NEU
from threading import Timer  # NEU
import importlib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, RobustScaler

try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from config.config_general import CONFIG_PATH, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS
    from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
    from ML_Helpfunctions.MQTT_Client import MqttInferenceClient
    from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
    from ML_Helpfunctions import Feature_Engeneering as fe
except ImportError as e:
    print(f"Fehler beim Importieren der Basis-Module: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pipeline")

PIPELINE_STATE = {
    "status": "initializing", "error_message": None, "retraining_status": "idle",
    "cycle_count": 0, "steps_in_cycle": 0, "total_steps": 0, "total_cycles": 0,
    "is_paused": False, "is_finished": False, "mode": "unknown"
}
shared_resource_lock = threading.Lock()
shared_model = {"model": None, "scaler": None, "y_scaler": None, "features": None, "config": None, "initial_training_data": None}
all_predictions = []


def load_config_dynamically(algorithm: str, config_name: str) -> dict:
    try:
        module_path = f"config.config_ml_{algorithm}"
        config_module = importlib.import_module(module_path)
        config_dict = getattr(config_module, config_name)
        logging.info(f"Konfiguration '{config_name}' erfolgrePich aus '{module_path}' geladen.")
        return deepcopy(config_dict)
    except (ImportError, AttributeError) as e:
        logging.error(f"Fehler beim dynamischen Laden der Konfiguration '{config_name}' aus '{module_path}': {e}", exc_info=True)
        sys.exit(1)


def initial_training(config: dict, trainer_class, folder_flag: str):
    global shared_model
    logging.info(f"--- PHASE 1: Initiales Training für {folder_flag} startet ---")
    try:
        trainer = trainer_class(config=config, folder_flag=folder_flag)
        pipeline = trainer._setup_pipeline()
        initial_data_df, _ = pipeline._load_data(mode='train')
        model, scaler, y_scaler, features = trainer.run(save_artifacts=True)

        with shared_resource_lock:
            shared_model.update({
                "model": model, "scaler": scaler, "y_scaler": y_scaler, "features": features,
                "config": config, "initial_training_data": initial_data_df
            })
        logging.info("--- PHASE 1: Initiales Training abgeschlossen. ---")
        with shared_resource_lock:
            PIPELINE_STATE["status"] = "ready_for_inference"
    except Exception as e:
        logging.error(f"Fehler während des initialen Trainings: {e}", exc_info=True)
        with shared_resource_lock:
            PIPELINE_STATE["status"] = "error"
            PIPELINE_STATE["error_message"] = str(e)


def retraining_thread_task(retraining_data_df: pd.DataFrame, algorithm: str):
    """
    Führt das Nachtraining im Hintergrund durch und bereitet Artefakte für den Hot-Swap vor.
    - LSTM/CNN1D: inkrementelles Fitten (nur In-Memory)
    - Random Forest: Refit auf kombinierten Daten (nur In-Memory)
    Persistiert: last_retraining_time_s, retraining_count, retraining_history in training_config.json
    """
    import logging, json, os, time
    from copy import deepcopy

    global shared_model, PIPELINE_STATE, shared_resource_lock

    # --- Helper: Retraining-Metriken (zeit, count, history) in Config & JSON speichern ---
    def _persist_retrain_metrics(cfg_in: dict, algo_name: str, retrain_time_s: float) -> None:
        import pandas as pd  # lokaler Import
        # In-Memory (mit Lock) aktualisieren
        with shared_resource_lock:
            cfg = deepcopy(cfg_in or {})
            cfg["last_retraining_time_s"] = float(retrain_time_s)
            cfg["retraining_count"] = int(cfg.get("retraining_count", 0)) + 1
            hist = cfg.get("retraining_history", [])
            if not isinstance(hist, list):
                hist = []
            hist.append({
                "timestamp": pd.Timestamp.utcnow().isoformat(),
                "algorithm": str(algo_name),
                "duration_s": float(retrain_time_s),
            })
            cfg["retraining_history"] = hist
            shared_model["config"] = cfg

            paths = cfg.get("paths") or {}
            models_dir = paths.get("Models") or paths.get("models_dir") or cfg.get("models_dir")

        # Auf Platte (außerhalb Lock)
        try:
            if models_dir:
                cfg_path = os.path.join(models_dir, "training_config.json")
                on_disk = {}
                if os.path.exists(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        try:
                            on_disk = json.load(f) or {}
                        except Exception:
                            on_disk = {}
                on_disk.update({
                    "last_retraining_time_s": cfg["last_retraining_time_s"],
                    "retraining_count": cfg["retraining_count"],
                    "retraining_history": cfg["retraining_history"],
                })
                with open(cfg_path, "w", encoding="utf-8") as f:
                    json.dump(on_disk, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.warning(f"Retraining-Zeit konnte nicht in training_config.json geschrieben werden: {e}")

    # --- Setup ---
    algo = (algorithm or "").lower()
    is_seq_model = algo in ("lstm", "cnn1d", "1dcnn", "cnn")

    logging.info(f"--- RETRAINING THREAD ({algorithm}): Startet Nachtraining. ---")
    t0_retrain = time.perf_counter()

    with shared_resource_lock:
        PIPELINE_STATE["retraining_status"] = "training"
        # Thread-sichere Kopie holen
        config = deepcopy(shared_model.get('config') or {})
        initial_data = shared_model.get('initial_training_data')
        current_model_ref = shared_model.get('model')
        scaler_ref = shared_model.get('scaler')
        y_scaler_ref = shared_model.get('y_scaler')
        features_ref = shared_model.get('features')

    if isinstance(features_ref, dict):
        features_ref = features_ref.get("all")
    if features_ref is not None:
        features_ref = list(features_ref)

    try:
        # === Sequenz-Modelle (LSTM/CNN1D) ===
        if is_seq_model:
            if current_model_ref is None or scaler_ref is None or not features_ref:
                logging.warning(f"{algorithm.upper()}: Fehlende Artefakte (Modell/Scaler/Features). Retraining abgebrochen.")
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return

            can_fit = hasattr(current_model_ref, "fit") and hasattr(current_model_ref, "get_weights")
            if not can_fit:
                logging.warning(f"{algorithm.upper()}: Modell nicht trainierbar (z. B. TFLite). Überspringe Retraining.")
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return

            # Modell klonen
            try:
                import tensorflow as tf
                cloned = tf.keras.models.clone_model(current_model_ref)
                cloned.set_weights(current_model_ref.get_weights())
                current_model = cloned
            except Exception:
                current_model = deepcopy(current_model_ref)

            # Feature-Engineering
            try:
                from ML_Helpfunctions import Feature_Engeneering as fe
                retraining_df_feat, _ = fe.add_all_features(retraining_data_df.copy(), config)
            except Exception as fe_err:
                logging.error(f"{algorithm.upper()}: Feature-Engineering fehlgeschlagen: {fe_err}", exc_info=True)
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return

            # gleiche Spalten/Order
            try:
                X_feat = retraining_df_feat.loc[:, features_ref].dropna().copy()
            except KeyError as kerr:
                logging.error(f"{algorithm.upper()}: Erwartete Features fehlen: {kerr}")
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return
            if X_feat.empty:
                logging.info(f"{algorithm.upper()}: Keine verwertbaren Zeilen nach Drop-NA.")
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return

            # Skalieren
            try:
                X_scaled = scaler_ref.transform(X_feat.values)
            except Exception as sc_err:
                logging.error(f"{algorithm.upper()}: Scaler.transform fehlgeschlagen: {sc_err}", exc_info=True)
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return

            # Sliding-Window
            try:
                from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
                H = int(config.get("horizon", 1))
                L = int(config.get("lags", 1))
                X_retrain, y_retrain = LoadPrepareData.convert_data_to_sliding_window(
                    X_scaled, lag_horizon=L, forecast_horizon=H
                )
            except Exception as sw_err:
                logging.error(f"{algorithm.upper()}: Sliding-Window-Erstellung fehlgeschlagen: {sw_err}", exc_info=True)
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return
            if X_retrain is None or y_retrain is None or len(X_retrain) == 0:
                logging.info(f"{algorithm.upper()}: Zu wenige Daten für Fenster (lags={L}).")
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return

            # Inkrementelles Fitten
            opt = config.get("optimizer", "adam")
            loss = config.get("loss", "mse")
            epochs = int(config.get("retraining_epochs", 3))
            batch_size = int(config.get("batch_size", 32))
            try:
                current_model.compile(optimizer=opt, loss=loss)
                current_model.fit(X_retrain, y_retrain, epochs=epochs, batch_size=batch_size, verbose=0)
            except Exception as fit_err:
                logging.error(f"{algorithm.upper()}: Fehler beim inkrementellen Fit: {fit_err}", exc_info=True)
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return

            # Zeit messen & speichern
            retrain_time_s = time.perf_counter() - t0_retrain
            _persist_retrain_metrics(config, algorithm, retrain_time_s)

            # Hot-Swap vorbereiten
            with shared_resource_lock:
                cfg_updated = deepcopy(shared_model.get("config") or config)
                shared_model.update({
                    "model": current_model,
                    "scaler": scaler_ref,
                    "y_scaler": y_scaler_ref,
                    "features": features_ref,
                    "config": cfg_updated,
                })
                PIPELINE_STATE["retraining_status"] = "ready_to_swap"

            logging.info(f"{algorithm.upper()}: Retraining abgeschlossen (samples={len(X_retrain)}, epochs={epochs}).")
            logging.info(f"--- RETRAINING THREAD ({algorithm.upper()}): Artefakte bereit zum Hot-Swap ---")
            return

        elif algo == "random_forest":
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import MinMaxScaler, RobustScaler
            from sklearn.multioutput import MultiOutputRegressor
            from ML_Helpfunctions import Feature_Engeneering as fe

            if initial_data is None:
                logging.warning("RandomForest: initial_training_data fehlt; überspringe Retraining.")
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return

            H = int(config.get("horizon", 1))
            target_col_name = config["base_features"][0].lower()

            logging.info("RandomForest: Kombiniere alte und neue Daten für Nachtraining...")
            combined_data = pd.concat([initial_data, retraining_data_df]).drop_duplicates().sort_index()

            logging.info("RandomForest: Feature Engineering für kombinierten Datensatz...")
            combined_df_featured, features_dict = fe.add_all_features(combined_data, config)
            new_features = features_dict["all"] if isinstance(features_dict, dict) else features_ref
            if not new_features:
                raise RuntimeError("RandomForest: Features nicht bestimmbar.")

            # X (t) + Y (t+1...t+H) sauber ausrichten
            X_all = combined_df_featured[new_features].copy()
            # Multi-Horizon Zielmatrix über Shifts bauen
            y_shifts = [combined_df_featured[target_col_name].shift(-k) for k in range(1, H + 1)]
            Y_all = pd.concat(y_shifts, axis=1)
            # Nur Zeilen, wo ALLE Horizonte vorhanden sind
            mask = Y_all.notna().all(axis=1)
            X_retrain = X_all.loc[mask]
            Y_retrain = (Y_all.loc[mask].iloc[:, 0] if H == 1 else Y_all.loc[mask])

            if X_retrain.empty or len(X_retrain) < 5:
                logging.warning("RandomForest: Zu wenige gültige Zeilen für Retraining. Abbruch.")
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return

            logging.info("RandomForest: Scaler neu anpassen...")
            scaler_class = RobustScaler if config.get("scaler_type", "minmax") == "robust" else MinMaxScaler
            new_scaler = scaler_class()
            X_retrain_scaled = new_scaler.fit_transform(X_retrain.values)

            logging.info("RandomForest: Neues Modell trainieren (In-Memory, horizon=%d)...", H)
            base_rf = RandomForestRegressor(**config.get("model_params", {}))
            if H == 1:
                new_model = base_rf
                new_model.fit(X_retrain_scaled, Y_retrain.values)
            else:
                # Explizit MultiOutput, passend zum Training
                new_model = MultiOutputRegressor(base_rf)
                new_model.fit(X_retrain_scaled, Y_retrain.values)

            # --- Dauer messen & persistieren (einheitlich) ---
            retrain_time_s = time.perf_counter() - t0_retrain
            _persist_retrain_metrics(config, algorithm, retrain_time_s)

            # Hot-Swap vorbereiten (nur In-Memory)
            with shared_resource_lock:
                cfg_updated = deepcopy(shared_model.get("config") or config)
                shared_model.update({
                    "model": new_model,
                    "scaler": new_scaler,
                    "features": new_features,
                    "initial_training_data": combined_data,
                    "config": cfg_updated,
                })
                PIPELINE_STATE["retraining_status"] = "ready_to_swap"

            logging.info("--- RETRAINING THREAD (RF): Artefakte bereit zum Hot-Swap ---")
            return

        else:
            raise ValueError(f"Unbekannter Algorithmus für Retraining: {algorithm}")

    except Exception as e:
        logging.error(f"RETRAINING THREAD: Fehler: {e}", exc_info=True)
        with shared_resource_lock:
            PIPELINE_STATE["retraining_status"] = "idle"



def _merge_artifacts_for_swap(current_infer_obj, shared_dict):
    """
    Sicherheits-Merge für Hot-Swap ohne Truthiness:
    - Nimmt jeweils den ersten Wert, der NICHT None ist.
    - Wichtig: Keine 'or'-Verknüpfungen mit DataFrames/Listen/etc., da das
      bei pandas.DataFrame zu 'ambiguous truth value' führt.
    """
    def pick(shared_key, infer_attr_name):
        val = shared_dict.get(shared_key, None)
        if val is not None:
            return val
        return getattr(current_infer_obj, infer_attr_name, None)

    return {
        "model": pick("model", "model"),
        "scaler": pick("scaler", "scaler"),
        "y_scaler": pick("y_scaler", "y_scaler"),
        "features": pick("features", "feature_list"),
        "config": pick("config", "config"),
        "initial_training_data": pick("initial_training_data", "_batch_data_df"),
    }


def inference_manager(config: dict, inference_class, folder_flag: str, algorithm: str, mode: str):
    """
    Führt die Inferenz in Zyklen/Schritten aus.
    - Speichert pro Schritt: True, Forecast(H), inference_time_s (aus prediction_entry), total_time_s, CPU%, RAM.
    - Sammelt pro Schritt Retraining-Material und triggert am Zyklusende ein non-blocking Retraining.
    - Hot-Swap, sobald neues Modell/Abhängigkeiten bereitstehen.
    - NEU: Falls nur Inferenz gestartet wird (mode != "retraining"), lade training_config.json aus dem
           Modellordner und merge sie in die config (laufende Werte haben Vorrang).
    """
    import os
    import json
    import time
    import threading
    import logging
    import pandas as pd
    from copy import deepcopy

    # psutil optional verwenden (CPU/RAM)
    try:
        import psutil
    except Exception:
        psutil = None

    # Globale Zustände verwenden (werden an anderer Stelle definiert)
    global all_predictions, shared_model, shared_resource_lock, PIPELINE_STATE, retraining_thread_task

    retraining_data_list = []
    inference_processor = None

    # ------------------------------------------------------------
    # Hilfsfunktionen: Config-Merge & Modell-Ordner finden
    # ------------------------------------------------------------
    def _deep_merge(base: dict, override: dict) -> dict:
        """
        Tiefen-Merge: Werte aus 'override' überschreiben 'base';
        verschachtelte Dicts werden rekursiv zusammengeführt.
        """
        if not isinstance(base, dict) or not isinstance(override, dict):
            return deepcopy(override)
        out = deepcopy(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_merge(out[k], v)
            else:
                # Wichtig: None aus override überschreibt NICHT (wir behalten base)
                if v is not None:
                    out[k] = v
        return out

    def _guess_model_dir(cfg: dict) -> str | None:
        """
        Rate den Modellordner. Priorität:
          1) cfg['paths']['Models']
          2) Ordner der Datei aus cfg['model_path_static'] (falls absolut und existent)
          3) cfg['paths']['Base_Output_Path']/Models
          4) cfg['artifacts_base_path']/Models
        Fällt zurück auf None, wenn nichts existiert.
        """
        paths = (cfg or {}).get("paths", {}) or {}

        # 1) Direkter Models-Pfad
        p_models = paths.get("Models")
        if p_models and os.path.isdir(p_models):
            return p_models

        # 2) Ordner der statischen Modell-Datei (absolute Pfade)
        m_static = (cfg or {}).get("model_path_static")
        if m_static and os.path.isabs(m_static) and os.path.isfile(m_static):
            return os.path.dirname(m_static)

        # 3) Base_Output_Path/Models
        base_out = paths.get("Base_Output_Path")
        if base_out:
            cand = os.path.join(base_out, "Models")
            if os.path.isdir(cand):
                return cand

        # 4) artifacts_base_path/Models
        art_base = (cfg or {}).get("artifacts_base_path")
        if art_base:
            cand = os.path.join(art_base, "Models")
            if os.path.isdir(cand):
                return cand

        # 5) Fallback: evtl. existiert Base_Output_Path selbst
        if base_out and os.path.isdir(base_out):
            return base_out

        return None


    def _try_load_training_config_for_inference_only(cfg: dict, _mode: str) -> dict:
        """
        Wenn wir NICHT im Retraining sind, versuche training_config.json aus dem
        Modellordner zu laden und in cfg zu mergen.
        WICHTIG: Kommandozeilen-Argumente wie 'model_filename' haben Priorität.
        """
        if str(_mode).lower() == "retraining":
            return cfg

        model_dir = _guess_model_dir(cfg)
        if not model_dir:
            logging.info("Kein Modellordner gefunden. Überspringe training_config.json-Import.")
            return cfg
            
        candidates = [
            os.path.join(model_dir, "training_config.json"),
            os.path.join(os.path.dirname(model_dir), "training_config.json")
        ]
        found = next((c for c in candidates if c and os.path.isfile(c)), None)

        if not found:
            logging.info(f"Keine training_config.json in {model_dir} gefunden. (Optional)")
            return cfg

        try:
            # --- KORRIGIERTE LOGIK START ---
            # 1. Lade die Konfiguration aus dem Trainingslauf. Sie dient als Basis.
            with open(found, "r", encoding="utf-8") as f:
                loaded_from_training = json.load(f)
                
            # 2. Führe die Konfigurationen zusammen.
            #    Die ursprüngliche `cfg`, die die Kommandozeilen-Argumente enthält,
            #    überschreibt die Werte aus dem gespeicherten Trainingslauf.
            #    Dadurch haben CLI-Argumente wie `loading_strategy` die höchste Priorität.
            merged = _deep_merge(loaded_from_training, cfg)
            
            merged["mode"] = _mode
            # --- KORRIGIERTE LOGIK ENDE ---

            logging.info(f"training_config.json geladen und gemergt: {found}")
            return merged
        except Exception as e:
            logging.warning(f"training_config.json konnte nicht geladen/gewertet werden ({found}): {e}")
            return cfg

    # ------------------------------------------------------------
    # Prozess & CPU-Kerne für Prozentberechnung
    # ------------------------------------------------------------
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            n_cpus = max(psutil.cpu_count(logical=True) or 1, 1)
        except Exception:
            proc, n_cpus = None, 1
    else:
        proc, n_cpus = None, 1

    try:
        # Auf Initialisierung warten
        while PIPELINE_STATE.get("status") == "initializing":
            time.sleep(0.2)
        if PIPELINE_STATE.get("status") == "error":
            logging.error("Inferenz-Manager startet nicht, da ein Fehler bei der Initialisierung aufgetreten ist.")
            return

        # NEU: falls nur Inferenz läuft -> versuche training_config.json zu laden/mergen
        config = _try_load_training_config_for_inference_only(config, mode)

        # Status setzen
        with shared_resource_lock:
            PIPELINE_STATE["status"] = "inference_running"
        logging.info(f"--- INFERENCE MANAGER: Startet im Modus '{mode}' ---")

        # Inferenz-Objekt aufsetzen + Artefakte laden/übernehmen
        inference_processor = inference_class(config, folder_flag=folder_flag)

        # 1) Falls bereits Artefakte im Speicher (z. B. nach initialem Training)
        if shared_model.get("model") is not None:
            inference_processor.set_artifacts_from_memory(shared_model)
            dp = getattr(inference_processor, "data_processor", None)
            if dp is not None and hasattr(dp, "reconfigure"):
                # Bevorzugt die explizit geladene training_config, sonst die gemergte runtime-config
                loaded_training_cfg = getattr(inference_processor, "training_config", None) or config
                try:
                    dp.reconfigure({"training_config": loaded_training_cfg}, keep_buffer=True)
                    logging.info("RealTimeDataProcessor re-initialized with loaded training config.")
                except Exception as e:
                    logging.warning(f"Reconfigure des DataProcessors fehlgeschlagen: {e}")
            # Warm-Start des DataProcessors
            try:
                init_df = shared_model.get("initial_training_data")
                dp = getattr(inference_processor, "data_processor", None)
                if dp is not None and init_df is not None and hasattr(dp, "prime_buffer"):
                    want = getattr(dp, "_min_data_points", 1)
                    dp.prime_buffer(init_df.tail(want))
                    logging.info("🔧 DataProcessor warm-started mit initialem Fenster.")
            except Exception as e:
                logging.warning(f"Konnte DataProcessor nicht vorfüttern: {e}")

        # 2) Sonst von Platte laden
        else:
            # a) Harte Vorgabe durch load_id (bestehende Logik)
            if config.get('load_id'):
                inference_processor.load_artifacts()
                dp = getattr(inference_processor, "data_processor", None)
                if dp is not None and hasattr(dp, "reconfigure"):
                    # Bevorzugt die explizit geladene training_config, sonst die gemergte runtime-config
                    loaded_training_cfg = getattr(inference_processor, "training_config", None) or config
                    try:
                        dp.reconfigure({"training_config": loaded_training_cfg}, keep_buffer=True)
                        logging.info("RealTimeDataProcessor re-initialized with loaded training config.")
                    except Exception as e:
                        logging.warning(f"Reconfigure des DataProcessors fehlgeschlagen: {e}")
            else:
                # b) Inferenz-only: versuche ohne load_id über Pfade/Dateien zu laden
                #    (BaseInferenceProcessor.load_artifacts nutzt Pipeline_Utils und cfg.paths)
                try:
                    inference_processor.load_artifacts()
                except SystemExit:
                    # Pipeline_Utils kann sys.exit(1) werfen – fange ab, um saubere Fehlermeldung zu loggen
                    logging.error("Artefakte konnten nicht geladen werden (SystemExit in load_artifacts).")
                    raise
                except Exception as e:
                    raise RuntimeError(
                        "Kein trainiertes Modell im Speicher und Artefakte konnten nicht von Platte geladen werden. "
                        "Stelle sicher, dass 'paths.Models' korrekt gesetzt ist und dort 'model.keras' / "
                        "'trained_*.joblib' sowie 'training_config.json' liegen."
                    ) from e

            # Nach Laden: shared_model BEFÜLLEN (wichtig für späteren Hot-Swap)
            with shared_resource_lock:
                shared_model.update({
                    "model": getattr(inference_processor, "model", None),
                    "scaler": getattr(inference_processor, "scaler", None),
                    "y_scaler": getattr(inference_processor, "y_scaler", None),
                    "features": getattr(inference_processor, "feature_list", None),
                    "config": getattr(inference_processor, "config", None),
                    "initial_training_data": getattr(inference_processor, "_batch_data_df", None)
                })

        # Datenquelle (zustandsbehafteter Iterator-Fabrik)
        data_source_iterator = inference_processor.get_data_source_iterator()

        # Zyklen & Schritte pro Zyklus
        if mode == "retraining":
            max_cycles = int(config.get("retraining_cycles", 5))
            steps_per_cycle = int(config.get("retraining_interval_steps", 20))
        else:
            max_cycles = 1
            steps_per_cycle = config.get("inference_steps", "infinite")
            if steps_per_cycle == "infinite":
                if hasattr(inference_processor, "_batch_data_df") and inference_processor._batch_data_df is not None:
                    steps_per_cycle = len(inference_processor._batch_data_df)
                else:
                    # Fallback: sichere Obergrenze
                    steps_per_cycle = int(config.get("fallback_steps", 1000))

        target_interval_sec = float(config.get("inference_interval_sec", 1.0))

        with shared_resource_lock:
            PIPELINE_STATE.update({
                "total_cycles": max_cycles,
                "total_steps": steps_per_cycle,
                "mode": mode
            })

        # --- Hauptschleife über Zyklen ---
        for cycle in range(max_cycles):
            with shared_resource_lock:
                PIPELINE_STATE.update({
                    "cycle_count": cycle + 1,
                    "retraining_status": "collecting" if mode == "retraining" else "idle"
                })
            logging.info(f"--- Zyklus {cycle + 1}/{max_cycles}: Starte Inferenz. ---")

            current_cycle_iterator = data_source_iterator(steps_per_cycle)

            # --- Schritt-Schleife ---
            for step, payload in enumerate(current_cycle_iterator):
                step_wall_t0 = time.perf_counter()

                # System-Ressourcen Messung (Anfang)
                cpu_t0, ram_usage_dict = None, None
                if psutil is not None:
                    try:
                        proc.cpu_percent(interval=None)  # Priming für genauere Messung
                        cpu_t0 = proc.cpu_times()
                        ram_usage_dict = PipelineUtils.get_memory_usage()
                    except Exception:
                        cpu_t0, ram_usage_dict = None, None

                # Pause/Stop
                while PIPELINE_STATE.get("is_paused"):
                    time.sleep(0.2)
                if PIPELINE_STATE.get("is_finished"):
                    break
                with shared_resource_lock:
                    PIPELINE_STATE["steps_in_cycle"] = step + 1

                # Einen Schritt verarbeiten
                prediction_entry = inference_processor.process_step(payload)

                if prediction_entry:
                    # Gesamtzeit (Wall)
                    step_total_time_s = time.perf_counter() - step_wall_t0

                    # CPU%
                    cpu_percent = None
                    if psutil is not None and cpu_t0 is not None:
                        try:
                            cpu_t1 = proc.cpu_times()
                            cpu_used = (cpu_t1.user + cpu_t1.system) - (cpu_t0.user + cpu_t0.system)
                            cpu_percent = (cpu_used / max(step_total_time_s, 1e-12)) / n_cpus * 100.0
                        except Exception:
                            cpu_percent = None

                    # RAM
                    ram_mb_val, ram_percent_val = None, None
                    if ram_usage_dict and ram_usage_dict.get("used_gb") != "N/A":
                        try:
                            ram_mb_val = float(ram_usage_dict["used_gb"]) * 1024
                            ram_percent_val = float(ram_usage_dict["percent"])
                            prediction_entry['ram_usage'] = ram_usage_dict  # für Web-UI
                        except (ValueError, TypeError):
                            pass

                    # Persistieren dieses Schritts
                    try:
                        inference_processor.save_step_result(
                            prediction_entry=prediction_entry,
                            total_time_s=step_total_time_s,
                            cpu_percent=cpu_percent,
                            ram_mb=ram_mb_val,
                            ram_percent=ram_percent_val
                        )
                    except Exception as persist_err:
                        logging.error(f"Fehler beim Speichern des Inferenz-Schritts: {persist_err}", exc_info=True)

                    # In-Memory sammeln (für finale Metriken)
                    with shared_resource_lock:
                        all_predictions.append(prediction_entry)

                    # Retraining-Material sammeln (nur im Modus 'retraining')
                    if mode == "retraining":
                        try:
                            if hasattr(payload, "to_dict"):
                                pl = payload.to_dict()
                            else:
                                pl = dict(payload)
                        except Exception:
                            pl = {}
                        if "datetime" not in pl and "datetime" in prediction_entry:
                            pl["datetime"] = prediction_entry["datetime"]
                        retraining_data_list.append(pl)

                # Hot-Swap, wenn Retraining fertig
                with shared_resource_lock:
                    ready = PIPELINE_STATE.get("retraining_status") == "ready_to_swap"
                if ready:
                    # Defensiver Merge, damit config/features nie auf None fallen
                    to_swap = _merge_artifacts_for_swap(inference_processor, shared_model)
                    inference_processor.set_artifacts_from_memory(to_swap)
                    with shared_resource_lock:
                        PIPELINE_STATE["retraining_status"] = "idle"
                    logging.info("--- INFERENCE MANAGER: Neues Modell & Abhängigkeiten aktiv (Hot-Swap) ---")

                # Zyklus-Takt einhalten
                elapsed = time.perf_counter() - step_wall_t0
                sleep_dur = target_interval_sec - elapsed
                if sleep_dur > 0:
                    time.sleep(sleep_dur)

            # vorzeitig beendet?
            if PIPELINE_STATE.get("is_finished"):
                break

            # --- Am Zyklusende: Retraining non-blocking starten (nur wenn etwas gesammelt wurde) ---
            if mode == "retraining" and cycle < max_cycles - 1:
                if len(retraining_data_list) == 0:
                    logging.info("--- Kein neues Retraining-Material gesammelt; starte kein Retraining. ---")
                else:
                    logging.info(f"--- Zyklus {cycle + 1}: Datensammlung abgeschlossen. Starte Retraining (non-blocking). ---")
                    retraining_df = pd.DataFrame(retraining_data_list)
                    # Index setzen, falls vorhanden
                    if "datetime" in retraining_df.columns:
                        retraining_df = retraining_df.set_index("datetime")
                    # Liste für den nächsten Zyklus leeren
                    retraining_data_list.clear()

                    # nicht doppelt starten
                    with shared_resource_lock:
                        already_training = (PIPELINE_STATE.get("retraining_status") == "training")

                    if not already_training:
                        retraining_thread = threading.Thread(
                            target=retraining_thread_task,
                            args=(retraining_df.copy(), algorithm),
                            name=f"RetrainingThread-{cycle+1}",
                            daemon=True
                        )
                        retraining_thread.start()
                        logging.info("--- Retraining läuft im Hintergrund; Inferenz läuft weiter. ---")
                    else:
                        logging.info("--- Überspringe Retraining-Start: Ein Retraining läuft bereits. ---")

    except Exception as e:
        logging.error(f"Schwerwiegender Fehler im Inference Manager: {e}", exc_info=True)
        with shared_resource_lock:
            PIPELINE_STATE.update({"status": "error", "error_message": str(e)})
    finally:
        logging.info("--- Pipeline beendet. Aufräumen und Speichern. ---")
        with shared_resource_lock:
            PIPELINE_STATE.update({"status": "finished", "is_finished": True})

        # Sauber stoppen
        if inference_processor is not None and hasattr(inference_processor, 'stop'):
            try:
                inference_processor.stop()
            except Exception:
                pass

        # Letzten evtl. gepufferten Eintrag flushen
        if inference_processor is not None and hasattr(inference_processor, 'flush_pending_entry'):
            try:
                last_entry = inference_processor.flush_pending_entry()
                if last_entry:
                    with shared_resource_lock:
                        all_predictions.append(last_entry)
            except Exception:
                pass

        # Finale Ergebnisse speichern
        try:
            if inference_processor is not None and all_predictions:
                logging.info("Speichere finale Vorhersagen...")
                inference_processor.save_final_results(all_predictions)
        except Exception as e:
            logging.error(f"Fehler beim finalen Speichern der Ergebnisse: {e}", exc_info=True)




def main():
    parser = argparse.ArgumentParser(description="Vereinheitlichte ML-Pipeline mit Web-UI")
    parser.add_argument('--algorithm', type=str, required=True, choices=['random_forest', 'lstm', 'cnn1d', 'xgboost', 'light_xgboost'],
                        help="Zu verwendender Algorithmus.")
    parser.add_argument('--config-name', type=str, help="Optional: Name der Konfigurationsvariable.")
    parser.add_argument('--retraining', action=argparse.BooleanOptionalAction, default=False,
                        help="Aktiviert den Retraining-Modus.")
    parser.add_argument("--load_id", type=str,
                        help="Optionale Run ID zum Laden von Artefakten anstelle von Training.")
    parser.add_argument("--model_filename", type=str,
                        help="Optional: Name der zu ladenden Modelldatei.")
    parser.add_argument("--web-only", action="store_true", help="Nur die Flask-Weboberfläche starten.")
    parser.add_argument("--no-web", action="store_true", help="Weboberfläche deaktivieren (Headless/Batch).")
    parser.add_argument("--host", default="0.0.0.0", help="Bind-Adresse für die Web-UI.")
    parser.add_argument("--port", type=int, default=None, help="Port für die Web-UI.")
    parser.add_argument("--inference-steps", type=int, default=None, help="Anzahl Inferenzschritte für Headless-Modus.")
    parser.add_argument("--set", action="append", default=[], help="Konfigurations-Override als key=value.")
    
    # GEÄNDERT: Das alte --no-quantization Flag wird durch das neue, flexiblere --quant-mode ersetzt
    parser.add_argument("--quant-mode", nargs='+', default=["no-quant"], choices=["no-quant", "quant-16", "quant-8"],
                        help="Quantisierungsmodus für das initiale Training. Default: no-quant.")

    args = parser.parse_args()

    # Defaults für config-name
    if args.config_name is None:
        args.config_name = f"{args.algorithm}"
        logging.info(f"Kein --config-name angegeben. Verwende Default: '{args.config_name}'")

    config = load_config_dynamically(args.algorithm, args.config_name)

    # NEU: Die neuen Quantisierungsmodi in die Konfiguration eintragen
    config['quant_modes'] = args.quant_mode
    if 'no-quant' not in args.quant_mode:
        logging.info(f"Quantisierungsmodi aktiviert: {args.quant_mode}")
    else:
        logging.info("Quantisierung ist deaktiviert.")

    # Algorithmus-spezifische Klassen wählen
    if args.algorithm == 'random_forest':
        from ML_Algorithms.Random_Forest.rf_train import RandomForestTrainer
        from ML_Algorithms.Random_Forest.rf_inference import RFInference
        trainer_class = RandomForestTrainer
        inference_class = RFInference
        folder_flag = "Random_Forest"
    elif args.algorithm == 'xgboost':
        from ML_Algorithms.XGBOOST.XGBOOST_train import XGBoostTrainer
        from ML_Algorithms.XGBOOST.XGBOOST_inference import XGBoostInference
        trainer_class = XGBoostTrainer
        inference_class = XGBoostInference
        folder_flag = "XGBOOST"

    elif args.algorithm == 'light_xgboost':
            from ML_Algorithms.XGBOOST.XGBOOST_train import XGBoostTrainer
            from ML_Algorithms.XGBOOST.XGBOOST_inference import XGBoostInference
            trainer_class = XGBoostTrainer
            inference_class = XGBoostInference
            folder_flag = "Light_XGBOOST"

    else:
        from ML_Algorithms.LSTM.LSTM_train import LSTMTrainer
        from ML_Algorithms.LSTM.LSTM_inference import LSTMInference
        trainer_class = LSTMTrainer
        inference_class = LSTMInference
        folder_flag = "LSTM"

    # --- CNN1D Support ---
    if args.algorithm == 'cnn1d':
        from ML_Algorithms.CNN1D.cnn1d_train import CNN1DTrainer
        from ML_Algorithms.CNN1D.cnn1d_inference import CNN1DInference
        trainer_class = CNN1DTrainer
        inference_class = CNN1DInference
        folder_flag = "CNN1D"

    # Basis-Konfigs mergen
    config.update(CONFIG_LOAD_ARTIFACTS)
    config.update(MQTT_CONFIG)
    config['paths'] = CONFIG_PATH['paths']

    # Modus + Port
    mode = "retraining" if args.retraining else "no_retraining"
    config['mode'] = mode
    default_port = 5002 if args.retraining else 5001
    port = args.port if args.port is not None else default_port

    # Inline-Overrides anwenden
    for kv in (args.set or []):
        if "=" in kv:
            k, v = kv.split("=", 1)
            k = k.strip()
            vv = v.strip()
            # primitive Typ-Erkennung
            if vv.lower() in ("true", "false"):
                vv = (vv.lower() == "true")
            else:
                try:
                    vv = int(vv)
                except ValueError:
                    try:
                        vv = float(vv)
                    except ValueError:
                        pass
            config[k] = vv
            logging.info(f"Override: {k} = {vv}")

    # Optional: Inferenzschritte deterministisch setzen
    if args.inference_steps is not None:
        if args.retraining:
            # Retraining: ein Zyklus mit N Schritten
            config['retraining_cycles'] = 1
            config['retraining_interval_steps'] = int(args.inference_steps)
        else:
            # Nur Inferenz: N Schritte
            config['inference_steps'] = int(args.inference_steps)

    log_msg = f"--- MODUS: {mode.replace('_', ' ')} | ALGORITHMUS: {args.algorithm} | CONFIG: {args.config_name} ---"

    # Modelldatei-Name (optional)
    if args.model_filename:
        config['model_filename'] = args.model_filename

    # Pfade & Run-ID konfigurieren
    if args.load_id:
        config['load_id'] = args.load_id
        config['run_id'] = args.load_id
        log_msg += f" | Lade Modell von Run ID: {args.load_id}"
        base_output_path = config['paths'].get('output')
        run_dir = os.path.join(base_output_path, folder_flag, args.load_id)
        config['paths'].update({
            "run_dir": run_dir,
            "Models": os.path.join(run_dir, "Models"),
            "Scalers": os.path.join(run_dir, "Scalers"),
            "Prediction_Data": os.path.join(run_dir, "Prediction_Data"),
            "Error_Metrics": os.path.join(run_dir, "Error_Metrics")
        })
    else:
        # Neuer Run -> experimentellen Ordner erzeugen
        _, paths = PipelineUtils.setup_experiment(config, folder_flag, run_type='train')
        config['paths'] = paths

    logging.info(log_msg)

    # Threads starten je nach Modus
    training_thread = None
    if not args.load_id:
        # Initiales Training im Hintergrund
        training_thread = threading.Thread(
            target=initial_training,
            args=(config, trainer_class, folder_flag),
            name="InitialTrainingThread",
            daemon=True
        )
        training_thread.start()
    else:
        # Modell wird von Platte geladen – Status für Inferenz setzen
        with shared_resource_lock:
            PIPELINE_STATE["status"] = "ready_for_inference"

    # Inferenz-Manager starten (läuft in eigenem Thread)
    inference_thread = threading.Thread(
        target=inference_manager,
        args=(config, inference_class, folder_flag, args.algorithm, mode),
        name="InferenceManagerThread",
        daemon=True
    )
    inference_thread.start()

    # Flask-App erzeugen
    from web_app import create_app
    app = create_app(config, PIPELINE_STATE, all_predictions, shared_resource_lock)

    # Werkzeug-Log dämpfen
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    # --- Steuerung Webserver ---
    if args.web_only:
        local_ip = PipelineUtils.get_local_ip()
        logging.info(f"\n🚀 Webserver (web-only) startet. Öffnen Sie http://{local_ip}:{port} in Ihrem Browser.")
        app.run(host=args.host, port=port, debug=False, use_reloader=False)
        return

    if not args.no_web:
        local_ip = PipelineUtils.get_local_ip()
        url = f"http://127.0.0.1:{port}"

        def open_browser():
            try:
                webbrowser.open_new(url)
            except Exception:
                pass

        Timer(1.5, open_browser).start()
        logging.info(f"\n🚀 Webserver startet. Öffnen Sie http://{local_ip}:{port} in Ihrem Browser.")
        threading.Thread(
            target=lambda: app.run(host=args.host, port=port, debug=False, use_reloader=False),
            name="FlaskThread",
            daemon=True
        ).start()

    # Headless: warten, bis Inferenz fertig ist (wenn inference_steps gesetzt)
    if args.no_web:
        if training_thread is not None:
            training_thread.join()
        inference_thread.join()
        logging.info("--- Pipeline (headless) beendet. ---")
        return

    # Mit Web: Keep-Alive-Schleife
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("Beende auf Benutzerwunsch (Ctrl+C).")


if __name__ == "__main__":
    main()
