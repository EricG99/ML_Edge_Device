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
    from ML_Helpfunctions import pipeline_utils as PipelineUtils
    from ML_Helpfunctions.MQTT_Client import MqttInferenceClient
    from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
    from ML_Helpfunctions import feature_engineering as fe
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
        logging.info(f"Konfiguration '{config_name}' erfolgreich aus '{module_path}' geladen.")
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
    - Speichert pro Schritt: True, Forecast(H), inference_time_s, total_time_s, CPU%, RAM.
    - Sammelt pro Schritt Retraining-Material und triggert am Zyklusende ein non-blocking Retraining.
    - Führt einen Hot-Swap durch, sobald ein neues Modell/Abhängigkeiten bereitstehen.
    - Stellt einen robusten Warm-Start des Datenpuffers sicher, inkl. Vorsammeln bei Live-Daten.
    """
    import os
    import json
    import time
    import threading
    import logging
    import pandas as pd
    import math
    import numpy as np  # <— für robuste NaN/Inf-Erkennung
    from copy import deepcopy

    try:
        import psutil
    except ImportError:
        psutil = None

    global all_predictions, shared_model, shared_resource_lock, PIPELINE_STATE, retraining_thread_task

    retraining_data_list = []
    inference_processor = None

    def _deep_merge(base: dict, override: dict) -> dict:
        if not isinstance(base, dict) or not isinstance(override, dict):
            return deepcopy(override)
        out = deepcopy(base)
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _deep_merge(out[k], v)
            else:
                if v is not None:
                    out[k] = v
        return out

    def _guess_model_dir(cfg: dict) -> str | None:
        paths = (cfg or {}).get("paths", {}) or {}
        p_models = paths.get("Models")
        if p_models and os.path.isdir(p_models): return p_models
        base_out = paths.get("Base_Output_Path")
        if base_out:
            cand = os.path.join(base_out, "Models")
            if os.path.isdir(cand): return cand
        return None

    def _try_load_training_config_for_inference_only(cfg: dict, _mode: str) -> dict:
        if str(_mode).lower() == "retraining":
            return cfg
        model_dir = _guess_model_dir(cfg)
        if not model_dir:
            logging.info("Kein Modellordner gefunden. Überspringe training_config.json-Import.")
            return cfg
        found = os.path.join(model_dir, "training_config.json")
        if not os.path.isfile(found):
            logging.info(f"Keine training_config.json in {model_dir} gefunden. (Optional)")
            return cfg
        try:
            with open(found, "r", encoding="utf-8") as f:
                loaded_from_training = json.load(f)
            merged = _deep_merge(loaded_from_training, cfg)
            merged["mode"] = _mode
            logging.info(f"training_config.json geladen und gemergt: {found}")
            return merged
        except Exception as e:
            logging.warning(f"training_config.json konnte nicht geladen/gewertet werden ({found}): {e}")
            return cfg

    # ---------- NEU/ROBUST: Forecast-Validierung ----------
    def _has_real_forecast(fc) -> bool:
        """
        True, wenn mindestens ein Wert numerisch und endlich ist.
        Deckt Python-Floats, numpy-Skalare, Listen/Arrays/Series ab.
        """
        if fc is None:
            return False
        try:
            if isinstance(fc, (list, tuple)):
                arr = np.asarray(fc, dtype=float)
            elif hasattr(fc, "to_numpy"):  # pandas Series
                arr = np.asarray(fc.to_numpy(), dtype=float)
            else:
                arr = np.asarray([fc], dtype=float)
            return np.isfinite(arr).any()
        except Exception:
            return False
    # ------------------------------------------------------

    proc, n_cpus = None, 1
    if psutil:
        try:
            proc = psutil.Process(os.getpid())
            n_cpus = max(psutil.cpu_count(logical=True) or 1, 1)
        except Exception:
            pass

    # ======================================================================
    # PHASE 1: INITIALISIERUNG
    # ======================================================================
    try:
        while PIPELINE_STATE.get("status") == "initializing":
            time.sleep(0.2)
        if PIPELINE_STATE.get("status") == "error":
            logging.error("Inferenz-Manager startet nicht, da ein Fehler bei der Initialisierung aufgetreten ist.")
            return

        config = _try_load_training_config_for_inference_only(config, mode)
        with shared_resource_lock:
            PIPELINE_STATE["status"] = "inference_running"
        logging.info(f"--- INFERENCE MANAGER: Startet im Modus '{mode}' ---")

        inference_processor = inference_class(config, folder_flag=folder_flag)

        if shared_model.get("model") is not None:
            logging.info("Übernehme Artefakte aus dem initialen Training (In-Memory).")
            inference_processor.set_artifacts_from_memory(shared_model)
        else:
            logging.info("Lade Artefakte von der Festplatte.")
            inference_processor.load_artifacts()
            with shared_resource_lock:
                shared_model.update({
                    "model": getattr(inference_processor, "model", None),
                    "scaler": getattr(inference_processor, "scaler", None),
                    "y_scaler": getattr(inference_processor, "y_scaler", None),
                    "features": getattr(inference_processor, "feature_list", None),
                    "config": getattr(inference_processor, "config", None),
                    "initial_training_data": getattr(inference_processor, "_batch_data_df", None)
                })

        # ======================================================================
        # PHASE 2: WARM-START & VORSAMMELN
        # ======================================================================
        data_processor = getattr(inference_processor, "data_processor", None)
        if data_processor is None:
            raise RuntimeError("Inference Processor hat keinen DataProcessor initialisiert.")

        initial_data_source = shared_model.get("initial_training_data")
        if config.get("loading_strategy") == "split" and initial_data_source is not None and not getattr(initial_data_source, "empty", True):
            initial_history_df = shared_model.get("initial_history_df")
            if initial_history_df is not None and not initial_history_df.empty:
                min_points = getattr(data_processor, "_min_data_points", 1)
                data_processor.prime_buffer(initial_history_df.tail(min_points))
                logging.info(f"🔧 DataProcessor warm-gestartet mit {len(initial_history_df.tail(min_points))} Zeilen.")
        data_source_iterator = inference_processor.get_data_source_iterator()

        if str(config.get("loading_strategy", "split")).lower() == "live_mqtt":
            min_points_needed = getattr(data_processor, "_min_data_points", 1)
            priming_timeout = float(config.get("priming_timeout_sec", 60.0))
            logging.info(f"--- MQTT-Vorsammeln: Benötige {min_points_needed} Punkte (Timeout {priming_timeout}s). ---")
            primed = len(getattr(data_processor, "_buffer", [])) if hasattr(data_processor, "_buffer") else 0
            t0 = time.perf_counter()
            priming_iterator = data_source_iterator('infinite')
            while primed < min_points_needed:
                if PIPELINE_STATE.get("is_finished"):
                    logging.warning("Vorsammeln durch Benutzer beendet.")
                    return
                try:
                    payload = next(priming_iterator)
                except StopIteration:
                    if (time.perf_counter() - t0) > priming_timeout:
                        logging.warning(f"Priming-Timeout nach {priming_timeout}s mit {primed}/{min_points_needed}.")
                        break
                    time.sleep(0.05)
                    continue
                try:
                    data_processor.update_and_process(payload)
                    primed = len(getattr(data_processor, "_buffer", [])) if hasattr(data_processor, "_buffer") else primed
                    with shared_resource_lock:
                        PIPELINE_STATE["primed_count"] = primed
                except Exception as e:
                    logging.warning(f"Priming: Fehler bei Payload-Verarbeitung: {e}")
                if (time.perf_counter() - t0) > priming_timeout:
                    logging.warning(f"Priming-Timeout nach {priming_timeout}s mit {primed}/{min_points_needed}.")
                    break
            logging.info(f"--- MQTT-Vorsammeln abgeschlossen. Puffer hat {primed} Einträge. ---")

        # ======================================================================
        # PHASE 3: HAUPT-INFERENZ
        # ======================================================================
        if str(mode).lower() == "retraining":
            max_cycles = int(config.get("retraining_cycles", 5))
            steps_per_cycle = int(config.get("retraining_interval_steps", 20))
        else:
            max_cycles = 1
            steps_cfg = config.get("inference_steps", "infinite")
            steps_per_cycle = float('inf') if str(steps_cfg).lower() == "infinite" else int(steps_cfg)

        target_interval_sec = float(config.get("inference_interval_sec", 1.0))
        with shared_resource_lock:
            PIPELINE_STATE.update({
                "total_cycles": max_cycles,
                "total_steps": -1 if steps_per_cycle == float('inf') else steps_per_cycle,
                "mode": mode
            })

        for cycle in range(max_cycles):
            with shared_resource_lock:
                PIPELINE_STATE.update({
                    "cycle_count": cycle + 1,
                    "retraining_status": "collecting" if str(mode).lower() == "retraining" else "idle"
                })
            logging.info(f"--- Zyklus {cycle + 1}/{max_cycles}: Starte Inferenz. ---")

            iterator_steps = 'infinite' if steps_per_cycle == float('inf') else (10**12 if str(config.get("loading_strategy","split")).lower() == "split" else 'infinite')
            current_cycle_iterator = data_source_iterator(iterator_steps)

            predictions_done = 0

            while predictions_done < steps_per_cycle:
                while PIPELINE_STATE.get("is_paused"): time.sleep(0.2)
                if PIPELINE_STATE.get("is_finished"): break

                try:
                    payload = next(current_cycle_iterator)
                except StopIteration:
                    logging.info("Datenquelle erschöpft, beende Zyklus.")
                    break

                step_wall_t0 = time.perf_counter()

                cpu_t0, ram_mb_val, ram_percent_val = None, None, None
                if psutil:
                    try:
                        proc.cpu_percent(interval=None)
                        cpu_t0 = proc.cpu_times()
                        pinfo = proc.memory_info()
                        ram_mb_val = pinfo.rss / (1024 * 1024)
                        ram_percent_val = psutil.virtual_memory().percent
                    except Exception:
                        pass

                prediction_entry = inference_processor.process_step(payload)
                if prediction_entry:
                    # === WICHTIG: Nur „echte“ Forecasts zählen/speichern (keine NaN/Inf) ===
                    fc = prediction_entry.get("future_forecast")
                    if not _has_real_forecast(fc):
                        # warm-up/platzhalter → NICHT zählen & NICHT speichern
                        elapsed = time.perf_counter() - step_wall_t0
                        if (sleep_dur := target_interval_sec - elapsed) > 0:
                            time.sleep(sleep_dur)
                        continue
                    # =====================================================================

                    predictions_done += 1
                    with shared_resource_lock:
                        PIPELINE_STATE["steps_in_cycle"] = predictions_done

                    step_total_time_s = time.perf_counter() - step_wall_t0
                    cpu_percent = None
                    if psutil and cpu_t0:
                        try:
                            cpu_t1 = proc.cpu_times()
                            cpu_used = (cpu_t1.user + cpu_t1.system) - (cpu_t0.user + cpu_t0.system)
                            cpu_percent = (cpu_used / max(step_total_time_s, 1e-12)) / n_cpus * 100.0
                        except Exception:
                            cpu_percent = None

                    # Persistieren
                    try:
                        inference_processor.save_step_result(
                            prediction_entry=prediction_entry,
                            total_time_s=step_total_time_s,
                            cpu_percent=cpu_percent,
                            ram_mb=ram_mb_val,
                            ram_percent=ram_percent_val  # CSV behält ram_percent
                        )
                    except Exception as persist_err:
                        logging.error(f"Fehler beim Speichern des Inferenz-Schritts: {persist_err}", exc_info=True)

                    # Aliasse für die Web-UI
                    if prediction_entry.get("inference_time_s") is not None and "inference_time_ms" not in prediction_entry:
                        prediction_entry["inference_time_ms"] = float(prediction_entry["inference_time_s"]) * 1000.0
                    prediction_entry["total_time_s"] = float(step_total_time_s)
                    prediction_entry["total_time_ms"] = float(step_total_time_s) * 1000.0
                    if "memory_percent" not in prediction_entry and ram_percent_val is not None:
                        prediction_entry["memory_percent"] = float(ram_percent_val)
                    if "ram_mb" not in prediction_entry and ram_mb_val is not None:
                        prediction_entry["ram_mb"] = float(ram_mb_val)

                    with shared_resource_lock:
                        all_predictions.append(prediction_entry)

                    if str(mode).lower() == "retraining":
                        try:
                            if hasattr(payload, "to_dict"):
                                pl = payload.to_dict()
                            else:
                                pl = dict(payload)
                        except Exception:
                            pl = {}
                        retraining_data_list.append(pl)

                # Hot-Swap?
                with shared_resource_lock:
                    ready = PIPELINE_STATE.get("retraining_status") == "ready_to_swap"
                if ready:
                    try:
                        to_swap = _merge_artifacts_for_swap(inference_processor, shared_model)  # noqa: F821
                        inference_processor.set_artifacts_from_memory(to_swap)
                        with shared_resource_lock:
                            PIPELINE_STATE["retraining_status"] = "idle"
                        logging.info("--- INFERENCE MANAGER: Neues Modell & Abhängigkeiten aktiv (Hot-Swap) ---")
                    except Exception as e:
                        logging.error(f"Fehler beim Hot-Swap: {e}", exc_info=True)

                # Zyklus-Takt
                elapsed = time.perf_counter() - step_wall_t0
                sleep_dur = target_interval_sec - elapsed
                if sleep_dur > 0:
                    time.sleep(sleep_dur)

            if PIPELINE_STATE.get("is_finished"):
                break

            if str(mode).lower() == "retraining" and cycle < max_cycles - 1 and retraining_data_list:
                logging.info(f"--- Zyklus {cycle + 1}: Starte Retraining (non-blocking). ---")
                retraining_df = pd.DataFrame(retraining_data_list)
                if "datetime" in retraining_df.columns:
                    retraining_df = retraining_df.set_index("datetime")
                retraining_data_list.clear()
                with shared_resource_lock:
                    is_training = PIPELINE_STATE.get("retraining_status") == "training"
                if not is_training:
                    threading.Thread(
                        target=retraining_thread_task, args=(retraining_df.copy(), algorithm),
                        name=f"RetrainingThread-{cycle+1}", daemon=True
                    ).start()
                else:
                    logging.info("--- Überspringe Retraining-Start: Ein Training läuft bereits. ---")

    except Exception as e:
        logging.error(f"Schwerwiegender Fehler im Inference Manager: {e}", exc_info=True)
        with shared_resource_lock:
            PIPELINE_STATE.update({"status": "error", "error_message": str(e)})
    finally:
        logging.info("--- Pipeline beendet. Aufräumen und Speichern. ---")
        with shared_resource_lock:
            PIPELINE_STATE.update({"status": "finished", "is_finished": True})

        if inference_processor:
            if hasattr(inference_processor, 'stop'):
                try: inference_processor.stop()
                except Exception: pass
            if hasattr(inference_processor, 'flush_pending_entry'):
                try:
                    last_entry = inference_processor.flush_pending_entry()
                    if last_entry:
                        with shared_resource_lock:
                            all_predictions.append(last_entry)
                except Exception:
                    pass
            try:
                if all_predictions:
                    inference_processor.save_final_results(all_predictions)
            except Exception as e:
                logging.error(f"Fehler beim finalen Speichern der Ergebnisse: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Vereinheitlichte ML-Pipeline mit Web-UI")
    parser.add_argument(
        "--algorithm",
        required=True,
        # 'svm' zur Liste der erlaubten Auswahlmöglichkeiten hinzufügen
        choices=['random_forest', 'lstm', 'cnn1d', 'xgboost', 'light_xgboost', 'svm', "ridge"],
        help="The algorithm to run."
    )
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
    parser.add_argument("--quant-mode", nargs='+', default=["no-quant"], choices=["no-quant", "quant-16", "quant-8"],
                        help="Quantisierungsmodus für das initiale Training. Default: no-quant.")
    parser.add_argument(
                        "--auto-quant-fallback",
                        action="store_true",
                        help="Starte erst no-quant in einem Subprozess; bei Fehler automatisch quant-16/quant-8 probieren.")


    args = parser.parse_args()

    if args.config_name is None:
        args.config_name = f"{args.algorithm}"
        logging.info(f"Kein --config-name angegeben. Verwende Default: '{args.config_name}'")

    config = load_config_dynamically(args.algorithm, args.config_name)

    config['quant_modes'] = args.quant_mode
    if 'no-quant' not in args.quant_mode:
        logging.info(f"Quantisierungsmodi aktiviert: {args.quant_mode}")
    else:
        logging.info("Quantisierung ist deaktiviert.")

    if args.algorithm == 'random_forest':
        from ML_Algorithms.Random_Forest.rf_train import RandomForestTrainer
        from ML_Algorithms.Random_Forest.rf_inference import RFInference
        trainer_class = RandomForestTrainer
        inference_class = RFInference
        folder_flag = "Random_Forest"
    elif args.algorithm == 'xgboost':
        from ML_Algorithms.XGBOOST.xgboost_train import XGBoostTrainer
        from ML_Algorithms.XGBOOST.xgboost_inference import XGBoostInference
        trainer_class = XGBoostTrainer
        inference_class = XGBoostInference
        folder_flag = "XGBOOST"
    elif args.algorithm == 'light_xgboost':
        from ML_Algorithms.Light_XGBOOST.light_xgboost_train import LightXGBoostTrainer
        from ML_Algorithms.Light_XGBOOST.light_xgboost_inference import LightXGBoostInference 
        trainer_class = LightXGBoostTrainer
        inference_class = LightXGBoostInference
        folder_flag = "Light_XGBOOST"
    elif args.algorithm == 'lstm':
        from ML_Algorithms.LSTM.lstm_train import LSTMTrainer
        from ML_Algorithms.LSTM.lstm_inference import LSTMInference
        trainer_class = LSTMTrainer
        inference_class = LSTMInference
        folder_flag = "LSTM"
    elif args.algorithm == 'cnn1d':
        from ML_Algorithms.CNN1D.cnn1d_train import CNN1DTrainer
        from ML_Algorithms.CNN1D.cnn1d_inference import CNN1DInference
        trainer_class = CNN1DTrainer
        inference_class = CNN1DInference
        folder_flag = "CNN1D"

    elif args.algorithm in ('ridge', 'lasso'):
        from ML_Algorithms.RIDGE.ridge_lasso_train import RidgeLassoTrainer
        from ML_Algorithms.RIDGE.ridge_lasso_inference import RidgeLassoInference
        trainer_class = RidgeLassoTrainer
        inference_class = RidgeLassoInference
        folder_flag = "RIDGE_LASSO"
    elif args.algorithm in ('svm', 'linear_svr', 'svr'):
        from ML_Algorithms.SVM.svm_train import SVMTrainer
        from ML_Algorithms.SVM.svm_inference import SVMInference
        trainer_class = SVMTrainer
        inference_class = SVMInference
        folder_flag = "SVM"

    config.update(CONFIG_LOAD_ARTIFACTS)
    config.update(MQTT_CONFIG)
    config['paths'] = CONFIG_PATH['paths']

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
            config['retraining_cycles'] = 1
            config['retraining_interval_steps'] = int(args.inference_steps)
        else:
            config['inference_steps'] = int(args.inference_steps)

    # >>> FIX: Web aktiv & keine Steps gesetzt → endloser Stream
    if (args.web_only or not args.no_web) and args.inference_steps is None:
        config['inference_steps'] = "infinite"

    log_msg = f"--- MODUS: {mode.replace('_', ' ')} | ALGORITHMUS: {args.algorithm} | CONFIG: {args.config_name} ---"

    if args.model_filename:
        config['model_filename'] = args.model_filename

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
        _, paths = PipelineUtils.setup_experiment(config, folder_flag, run_type='train')
        config['paths'] = paths

    logging.info(log_msg)

    # === AUTO QUANT FALLBACK LAUNCHER ===
    if args.auto_quant_fallback and args.load_id:
        model_dir = config['paths'].get('Models')
        if not model_dir or not os.path.isdir(model_dir):
            logging.error(f"Models-Ordner nicht gefunden: {model_dir}")
            sys.exit(2)

        variants = list_model_variants(model_dir)  # [(pfad, label), label in {'no-quant','quant-16','quant-8'}]
        if not variants:
            logging.error(f"Keine Modellvarianten in {model_dir} gefunden.")
            sys.exit(2)

        # Bevorzugte Reihenfolge: erst FP32/FP16 testen, dann INT8
        try_order = ["no-quant", "quant-16", "quant-8"]

        # Wenn Nutzer explizit eine Datei angibt, versuche deren Label zuerst
        first = []
        if args.model_filename:
            fn_low = args.model_filename.lower()
            if fn_low.endswith((".keras", ".h5", ".json")):
                first = ["no-quant"]
            elif any(k in fn_low for k in ("fp16", "float16", "quant16", "_16", "-16")):
                first = ["quant-16"]
            elif any(k in fn_low for k in ("int8", "quant8", "q8", "_8", "-8")):
                first = ["quant-8"]

        # Map label -> Dateiname
        by_label = {}
        for p, lbl in variants:
            by_label[lbl] = os.path.basename(p)

        # Zusätzliche Args für das Kind bauen (Web-Flags etc.)
        add_args = []
        if not args.no_web:
            # gleiche Host/Port an Kind durchreichen
            add_args += ["--host", args.host]
            if port is not None:
                add_args += ["--port", str(port)]
        if args.web_only:
            add_args += ["--web-only"]
        if args.retraining:
            add_args += ["--retraining"]

        # Konfig-Overrides durchreichen (wichtige Keys)
        extra_sets = {}
        if "loading_strategy" in config:
            extra_sets["loading_strategy"] = config["loading_strategy"]
        if "inference_interval_sec" in config:
            extra_sets["inference_interval_sec"] = config["inference_interval_sec"]

        # Inference-Steps nur setzen, wenn explizit angegeben (sonst darf 'infinite' gelten)
        steps = args.inference_steps if args.inference_steps is not None else None

        tried = set()
        for lbl in first + [l for l in try_order if l not in first]:
            if lbl in tried:
                continue
            tried.add(lbl)
            fn = by_label.get(lbl)
            if not fn:
                continue

            logging.info(f"[Launcher] Versuche Modellvariante: {lbl} ({fn})")
            rc = run_inference_via_subprocess(
                load_id=args.load_id,
                model_filename=fn,
                algorithm=args.algorithm,
                no_web=args.no_web,
                inference_steps=steps,
                extra_sets=extra_sets,
                additional_args=add_args
            )

            # Erfolgreich?
            if rc == 0:
                logging.info(f"[Launcher] Variante {lbl} lief erfolgreich. Beende Launcher.")
                sys.exit(0)

            # Typische OOM-/Kill-Codes abfangen
            if rc in (137, 9, -9):
                logging.warning(f"[Launcher] Kindprozess durch OOM/SIGKILL beendet (rc={rc}). Probiere nächste Variante.")
            else:
                logging.warning(f"[Launcher] Kindprozess endete mit rc={rc}. Probiere nächste Variante.")

        logging.error("[Launcher] Keine Modellvariante lief erfolgreich.")
        sys.exit(1)
    # === ENDE Launcher ===


    training_thread = None
    if not args.load_id:
        training_thread = threading.Thread(
            target=initial_training,
            args=(config, trainer_class, folder_flag),
            name="InitialTrainingThread",
            daemon=True
        )
        training_thread.start()
    else:
        with shared_resource_lock:
            PIPELINE_STATE["status"] = "ready_for_inference"

    inference_thread = threading.Thread(
        target=inference_manager,
        args=(config, inference_class, folder_flag, args.algorithm, mode),
        name="InferenceManagerThread",
        daemon=True
    )
    inference_thread.start()

    from web_app import create_app
    app = create_app(config, PIPELINE_STATE, all_predictions, shared_resource_lock)

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.WARNING)

    if args.web_only:
        local_ip = PipelineUtils.get_local_ip()
        logging.info(f"\n🚀 Webserver (web-only) startet. Öffnen Sie http://{local_ip}:{port} in Ihrem Browser.")
        app.run(host=args.host, port=port, debug=False, use_reloader=False)
        return

    if not args.no_web:
        local_ip = PipelineUtils.get_local_ip()
        url = f"http://{ '127.0.0.1' }:{port}"

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

    if args.no_web:
        if training_thread is not None:
            training_thread.join()
        inference_thread.join()
        logging.info("--- Pipeline (headless) beendet. ---")
        return

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("Beende auf Benutzerwunsch (Ctrl+C).")


if __name__ == "__main__":
    main()


# === BEGIN: Shared experiment utilities (centralized per requests 6,7,8) ===
import os
import json
import subprocess
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd


def algorithm_to_folder(algorithm: str) -> str:
    mapping = {
        "lstm": "LSTM",
        "cnn1d": "CNN1D",
        "random_forest": "Random_Forest",
        "xgboost": "XGBOOST",
        "light_xgboost": "Light_XGBOOST",
    }
    algo_key = (algorithm or "").strip().lower()
    return mapping.get(algo_key, algo_key.upper())


def normalize_quant_label(label: Optional[str]) -> str:
    """
    Normalisiert verschiedene Schreibweisen auf: no-quant | quant-16 | quant-8
    """
    if not label:
        return "no-quant"
    l = str(label).strip().lower()
    if l in ("quant-8-full", "int8", "8", "q8", "full-int8", "quant8"):
        return "quant-8"
    if l in ("quant-16", "float16", "fp16", "q16", "16", "quant16"):
        return "quant-16"
    if l in ("none", "no", "no-quant", "fp32", "float32"):
        return "no-quant"
    return l


def list_model_variants(model_dir: str) -> List[Tuple[str, str]]:
    """
    Liefert Liste von (variant_path, variant_label).
    Erkennt Keras, TFLite, joblib/PKL und JSON-Modelle.
    Labels sind normalisiert: no-quant | quant-16 | quant-8
    """
    p = Path(model_dir)
    out: List[Tuple[str, str]] = []
    if not p.exists():
        return out

    for f in ["model.keras", "model.h5"]:
        if (p / f).exists():
            out.append((str(p / f), "no-quant"))
            break

    for f in p.glob("*.tflite"):
        name = f.name.lower()
        q = "no-quant"
        if any(k in name for k in ("int8", "quant8", "q8")):
            q = "quant-8"
        elif any(k in name for k in ("fp16", "float16", "quant16", "16")):
            q = "quant-16"
        out.append((str(f), normalize_quant_label(q)))

    for g in ("*.joblib", "*.pkl", "model.json"):
        for f in p.glob(g):
            out.append((str(f), "no-quant"))

    seen = set()
    unique: List[Tuple[str, str]] = []
    for path, label in out:
        key = (Path(path).name, label)
        if key not in seen:
            seen.add(key)
            unique.append((path, label))
    return unique


def summarize_step_csv(step_csv_path: str) -> Dict[str, float]:
    """
    Berechnet Mittelwerte aus einer Step-CSV:
    inference_time_ms, total_time_ms, cpu_percent, memory_percent.
    """
    step_csv_path = str(step_csv_path)
    if not os.path.exists(step_csv_path):
        return {}
    try:
        df = pd.read_csv(step_csv_path)
    except Exception:
        return {}

    metrics = [
        c
        for c in ["inference_time_ms", "total_time_ms", "cpu_percent", "memory_percent"]
        if c in df.columns
    ]
    res: Dict[str, float] = {}
    for m in metrics:
        try:
            res[f"avg_{m}"] = float(pd.to_numeric(df[m], errors="coerce").mean())
        except Exception:
            pass
    return res


def discover_predictions_file_from_json(output_dir: str) -> Optional[str]:
    """
    Versucht, die Step-CSV über eine JSON-Zusammenfassung im output_dir zu finden.
    """
    output_dir = str(output_dir)
    candidates = ["inference_summary.json", "predictions_meta.json"]
    for nm in candidates:
        fp = os.path.join(output_dir, nm)
        if os.path.exists(fp):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if isinstance(data, dict):
                    step_csv = data.get("step_csv") or data.get("predictions_step_csv")
                    if step_csv:
                        cand = step_csv if os.path.isabs(step_csv) else os.path.join(output_dir, step_csv)
                        if os.path.exists(cand):
                            return cand
            except Exception:
                pass
    return None


def fallback_find_step_csv(output_dir: str) -> Optional[str]:
    """
    Fallback: rekursiv nach *_predictions_step.csv im output_dir suchen.
    """
    output_dir = str(output_dir)
    for root, _, files in os.walk(output_dir):
        for fn in files:
            low = fn.lower()
            if low.endswith("_predictions_step.csv") or low.endswith("predictions_step.csv"):
                return os.path.join(root, fn)
    return None


def get_summary_output_path(output_root: str, filename: str = "Experiment_Summary.csv") -> str:
    """
    Erzeugt/erzwingt den einheitlichen Zielort .../output/Error_Metrics/<filename>
    und gibt den absoluten Pfad zurück.
    """
    output_root = str(output_root)
    if os.path.basename(output_root) != "output":
        output_root = os.path.join(output_root, "output")
    em_dir = os.path.join(output_root, "Error_Metrics")
    os.makedirs(em_dir, exist_ok=True)
    return os.path.join(em_dir, filename)


def _resolve_python_executable(python_executable: Optional[str]) -> str:
    """
    Wählt robust einen Python-Interpreter (fix für WindowsApps-Stub):
    - expliziter Pfad (falls existiert)
    - sys.executable (falls existent und nicht WindowsApps)
    - which('python') / which('python3') / which('py')
    - sonst 'python'
    """
    if python_executable and os.path.exists(python_executable):
        return python_executable

    cand = sys.executable
    if cand and os.path.exists(cand) and "WindowsApps" not in cand:
        return cand

    for name in ("python", "python3", "py"):
        path = shutil.which(name)
        if path:
            return path

    return "python"


def run_inference_via_subprocess(
    load_id: str,
    model_filename: str,
    horizon: int,
    *,
    algorithm: str,
    folder_flag: str,
    no_web: bool = True,
    extra_sets: Optional[Dict[str, str]] = None,
    python_executable: Optional[str] = None,
    pipeline_script: Optional[str] = None,
    additional_args: Optional[List[str]] = None,
) -> int:
    """
    Startet die pipeline_web_app in einem Subprozess für die Inferenz.
    Gibt den Returncode des Prozesses zurück.
    """
    py = _resolve_python_executable(python_executable)
    script = pipeline_script or os.path.realpath(__file__)
    args = [
        py,
        script,
        "--load_id",
        str(load_id),
        "--inference-steps",
        str(horizon),
        "--model_filename",
        str(model_filename),
    ]
    if no_web:
        args.append("--no-web")
    if algorithm:
        args += ["--algorithm", str(algorithm)]
    if folder_flag:
        args += ["--folder_flag", str(folder_flag)]
    if extra_sets:
        for k, v in extra_sets.items():
            args += ["--set", f"{k}={v}"]
    if additional_args:
        args += list(additional_args)

    print("[SPAWN]", " ".join(args))
    return subprocess.call(args)
# === END: Shared experiment utilities ===