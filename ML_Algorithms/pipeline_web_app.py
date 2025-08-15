# pipeline_web_app.py

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

import webbrowser # NEU
from threading import Timer # NEU
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
shared_model = {"model": None, "scaler": None, "features": None, "config": None, "initial_training_data": None}
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
    WICHTIG: Setzt den Pipeline-Status am Ende auf 'ready_to_swap', damit der Inferenz-Loop
    selbstständig die Artefakte übernimmt (kein Blockieren!).
    """
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.ensemble import RandomForestRegressor
    import joblib

    global shared_model
    logging.info(f"--- RETRAINING THREAD ({algorithm}): Startet Nachtraining. ---")
    with shared_resource_lock:
        PIPELINE_STATE["retraining_status"] = "training"

    try:
        # Bestehende Artefakte & Konfiguration auslesen (thread-safe Kopie)
        with shared_resource_lock:
            config = deepcopy(shared_model['config'])
            initial_data = shared_model['initial_training_data']
            current_model_ref = shared_model['model']
            current_scaler_ref = shared_model['scaler']
            features_ref = list(shared_model['features']) if shared_model['features'] is not None else None

        if algorithm.lower() == 'lstm':
            # --- LSTM: unverändert lassen (läuft bei dir bereits stabil) ---
            logging.info("LSTM-Nachtraining (inkrementell) wird durchgeführt...")
            # Hier bleibt dein bestehender LSTM-Retrain-Flow erhalten.
            # Wir setzen lediglich am Ende das Swap-Signal, da LSTM bereits funktioniert.
            with shared_resource_lock:
                PIPELINE_STATE["retraining_status"] = "ready_to_swap"
            logging.info("--- RETRAINING THREAD (LSTM): Artefakte bereit zum Hot-Swap ---")
            return

        elif algorithm.lower() == 'cnn1d':
            # CNN1D: (Platzhalter) – Retraining könnte analog LSTM/Langsames Keras-Training laufen.
            logging.info("CNN1D-Nachtraining (inkrementell) Platzhalter – setze Swap-Signal.")
            with shared_resource_lock:
                PIPELINE_STATE["retraining_status"] = "ready_to_swap"
            logging.info("--- RETRAINING THREAD (CNN1D): Artefakte bereit zum Hot-Swap ---")
            return
        
        elif algorithm.lower() == 'xgboost':
            # --- XGBoost Retraining (Best-Practice): Trees anhängen (continued training) ---
            logging.info("XGBoost Retrain (append trees) startet ...")

            if retraining_data_df is None or retraining_data_df.empty:
                logging.warning("Retraining-Daten sind leer. Überspringe XGBoost-Retrain.")
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return

            # 1) Historie + neue Daten kombinieren (kleines Backfill gegen Forgetting)
            hist_rows = int(config.get("xgb_retrain_hist_rows", 5000))
            df_old = (initial_data.copy() if initial_data is not None else pd.DataFrame()).tail(hist_rows)
            df_new = retraining_data_df.copy()

            for df in (df_old, df_new):
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)

            combined_raw = pd.concat([df_old, df_new], axis=0)
            combined_raw = combined_raw[~combined_raw.index.duplicated(keep='last')]
            combined_raw = combined_raw.sort_index()

            # 2) Feature Engineering wie im Training
            featured_all, _ = fe.add_all_features(combined_raw.copy(), config)

            # 3) X exakt wie im initialen Training (features_ref)
            target_col = str(config.get("base_features", ["target"])[0]).lower()
            if features_ref is None or len(features_ref) == 0:
                raise ValueError("Feature-Liste (features_ref) ist leer oder None – initiales Training hat nichts gespeichert.")
            missing = [c for c in features_ref if c not in featured_all.columns]
            if missing:
                raise KeyError(f"Folgende Features fehlen im neu erstellten Feature-Frame: {missing}")

            X_df_full = featured_all[features_ref].copy()

            # 4) Zielmatrix (H Horizons) erzeugen
            H = int(config.get("horizon", 1))
            H = max(1, H)
            y_cols = []
            for h in range(1, H + 1):
                col_name = f"__y_t_plus_{h}__"
                featured_all[col_name] = featured_all[target_col].shift(-h)
                y_cols.append(col_name)

            aligned_df = pd.concat([X_df_full, featured_all[y_cols]], axis=1).dropna(how='any')
            if aligned_df.empty:
                raise ValueError("Nach Ausrichtung/Dropna sind keine Zeilen übrig – bitte Fenstergrößen/Lags prüfen.")

            X_df = aligned_df[features_ref]
            Y_df = aligned_df[y_cols]

            # 5) Skaler neu fitten (Drift berücksichtigen)
            logging.info("Passe Scaler neu an...")
            scaler_type = (config.get("scaler_type") or "standard").lower()
            if scaler_type == "minmax":
                from sklearn.preprocessing import MinMaxScaler
                new_scaler = MinMaxScaler()
            elif scaler_type == "robust":
                from sklearn.preprocessing import RobustScaler
                new_scaler = RobustScaler()
            else:
                from sklearn.preprocessing import StandardScaler
                new_scaler = StandardScaler()

            X_scaled = new_scaler.fit_transform(X_df.values)
            Y = Y_df.values

            # 6) Weitertrainieren (Trees anhängen)
            from xgboost import XGBRegressor
            from sklearn.multioutput import MultiOutputRegressor
            add_trees = int(config.get("xgb_additional_estimators", 200))
            early_rounds = int(config.get("xgb_early_stopping_rounds", 0))

            old_model = shared_model.get("model")
            if old_model is None:
                raise RuntimeError("Kein existierendes XGBoost-Modell im Shared-Store gefunden.")

            def _fit_with_optional_es(estimator, X_, y_, booster):
                # Robust gegen XGBoost-Versionen: erst versuchen mit early_stopping_rounds,
                # sonst Fallback auf callbacks, sonst ohne ES.
                fit_kwargs = {"xgb_model": booster}
                if early_rounds > 0:
                    try:
                        estimator.fit(X_, y_, eval_set=[(X_, y_)], early_stopping_rounds=early_rounds, **fit_kwargs)
                        return estimator
                    except TypeError:
                        try:
                            import xgboost as xgb
                            estimator.fit(X_, y_, eval_set=[(X_, y_)],
                                          callbacks=[xgb.callback.EarlyStopping(rounds=early_rounds, save_best=True)],
                                          **fit_kwargs)
                            return estimator
                        except TypeError:
                            pass
                estimator.fit(X_, y_, **fit_kwargs)
                return estimator

            if H == 1 and not isinstance(old_model, MultiOutputRegressor):
                old_params = old_model.get_params(deep=True)
                old_params["n_estimators"] = int(old_params.get("n_estimators", 100)) + add_trees

                new_est = XGBRegressor(**old_params)
                booster = old_model.get_booster()
                new_est = _fit_with_optional_es(new_est, X_scaled, Y.ravel(), booster)
                new_model = new_est
            else:
                if not hasattr(old_model, "estimators_"):
                    raise ValueError("Erwarte MultiOutputRegressor für H>1.")
                new_estimators = []
                for k, old_est in enumerate(old_model.estimators_):
                    params = old_est.get_params(deep=True)
                    params["n_estimators"] = int(params.get("n_estimators", 100)) + add_trees
                    new_k = XGBRegressor(**params)
                    booster = old_est.get_booster()
                    new_k = _fit_with_optional_es(new_k, X_scaled, Y[:, k], booster)
                    new_estimators.append(new_k)
                old_model.estimators_ = new_estimators
                new_model = old_model

            # 7) Artefakte für Hot-Swap bereitstellen
            with shared_resource_lock:
                shared_model.update({
                    "model": new_model,
                    "scaler": new_scaler,
                    "features": features_ref,
                    "config": config,
                    "initial_training_data": combined_raw
                })
                PIPELINE_STATE["retraining_status"] = "ready_to_swap"

            logging.info("--- RETRAINING THREAD (XGBoost): Artefakte bereit zum Hot-Swap ---")



        elif algorithm.lower() == 'random_forest':
            # =========================
            # Random Forest – FIX: Multi-Output beibehalten und X exakt wie im Training
            # =========================
            logging.info("Kombiniere alte und neue Daten für RF-Nachtraining...")

            # 1) Rohdaten zusammenführen und sauber sortieren
            if retraining_data_df is None or retraining_data_df.empty:
                logging.warning("Retraining-Daten sind leer. Überspringe RF-Retrain.")
                with shared_resource_lock:
                    PIPELINE_STATE["retraining_status"] = "idle"
                return

            df_old = initial_data.copy() if initial_data is not None else pd.DataFrame()
            df_new = retraining_data_df.copy()

            # Index/Zeiten bereinigen
            for df in (df_old, df_new):
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    else:
                        # Fallback: laufender Index, wird unten trotzdem sortiert
                        pass

            combined_raw = pd.concat([df_old, df_new], axis=0)
            combined_raw = combined_raw[~combined_raw.index.duplicated(keep='last')]
            try:
                combined_raw = combined_raw.sort_index()
            except Exception:
                combined_raw = combined_raw.reset_index(drop=True)

            # 2) Feature Engineering (genau wie im Training)
            logging.info("Feature Engineering für kombinierten Datensatz...")
            featured_all, _ = fe.add_all_features(combined_raw.copy(), config)

            # Zielvariable ermitteln (wie im RFInference genutzt)
            target_col = str(config.get("base_features", ["target"])[0]).lower()
            if target_col not in map(str.lower, featured_all.columns):
                # Versuche ohne lower() (falls Spaltennamen bereits klein sind)
                if target_col not in featured_all.columns:
                    raise KeyError(
                        f"Target-Spalte '{target_col}' nicht im Feature-DataFrame gefunden. "
                        f"Vorhandene Spalten: {list(featured_all.columns)[:10]} ..."
                    )

            # 3) X exakt wie beim initialen Training: features_ref verwenden
            if features_ref is None or len(features_ref) == 0:
                raise ValueError("Feature-Liste (features_ref) ist leer oder None – initiales Training hat nichts gespeichert.")

            # Safety: prüfe, dass alle Features existieren
            missing = [c for c in features_ref if c not in featured_all.columns]
            if missing:
                raise KeyError(f"Folgende Features fehlen im neu erstellten Feature-Frame: {missing}")

            X_df_full = featured_all[features_ref].copy()

            # 4) Y_{t+1 ... t+H} bauen (Multi-Output) – exakt aus der Zielspalte
            H = int(config.get("horizon", 1))
            if H < 1:
                H = 1

            # Shifts für Zukunft
            y_cols = []
            for h in range(1, H + 1):
                col_name = f"__y_t_plus_{h}__"
                featured_all[col_name] = featured_all[target_col].shift(-h)
                y_cols.append(col_name)

            # 5) NaNs entfernen (kommen von Lags/Rollings/Shift)
            aligned_df = pd.concat([X_df_full, featured_all[y_cols]], axis=1)
            aligned_df = aligned_df.dropna(axis=0, how='any')
            if aligned_df.empty:
                raise ValueError("Nach Ausrichtung/Dropna sind keine Zeilen übrig – bitte Fenstergrößen/Lags prüfen.")

            X_df = aligned_df[features_ref]
            Y_df = aligned_df[y_cols]

            # 6) Scaler NEU anpassen (wie beim initialen Training)
            logging.info("Passe Scaler neu an...")
            scaler_type = (config.get("scaler_type") or "standard").lower()
            if scaler_type == "minmax":
                new_scaler = MinMaxScaler()
            elif scaler_type == "robust":
                new_scaler = RobustScaler()
            else:
                new_scaler = StandardScaler()

            X_scaled = new_scaler.fit_transform(X_df.values)
            Y = Y_df.values  # Shape: (n_samples, H)

            # 7) Modell trainieren – **immer** Multi-Output, wenn H>1
            logging.info("Trainiere neues Random Forest Modell (Multi-Output)...")
            rf_base = RandomForestRegressor(
                n_estimators=config.get("n_estimators", 100),
                max_depth=config.get("max_depth", None),
                min_samples_split=config.get("min_samples_split", 2),
                min_samples_leaf=config.get("min_samples_leaf", 1),
                max_features=config.get("max_features", 1.0),
                random_state=config.get("random_state", None),
                n_jobs=config.get("n_jobs", -1)
            )
            if H > 1:
                new_model = MultiOutputRegressor(rf_base)
            else:
                new_model = rf_base

            new_model.fit(X_scaled, Y)

            # 8) Artefakte für Hot-Swap bereitstellen
            with shared_resource_lock:
                # Feature-Liste bleibt exakt gleich!
                shared_model.update({
                    "model": new_model,
                    "scaler": new_scaler,
                    "features": features_ref,
                    "config": config,
                    # Für nächste Retrains: kombinierten Rohdatenstand merken
                    "initial_training_data": combined_raw
                })
                PIPELINE_STATE["retraining_status"] = "ready_to_swap"

            logging.info("--- RETRAINING THREAD (RF): Artefakte bereit zum Hot-Swap ---")

        else:
            raise ValueError(f"Unbekannter Algorithmus für Retraining: {algorithm}")

    except Exception as e:
        logging.error(f"RETRAINING THREAD: Fehler: {e}", exc_info=True)
        # Fehlerfall: Status zurücksetzen, damit ein späterer Versuch möglich bleibt
        with shared_resource_lock:
            PIPELINE_STATE["retraining_status"] = "idle"


def inference_manager(config: dict, inference_class, folder_flag: str, algorithm: str, mode: str):
    """
    Führt die Inferenz in Zyklen/Schritten aus.
    - Speichert pro Schritt: True, Forecast(H), inference_time_s (aus prediction_entry), total_time_s, CPU%, RAM.
    - Sammelt pro Schritt Retraining-Material und triggert am Zyklusende ein non-blocking Retraining.
    - Hot-Swap, sobald neues Modell/Abhängigkeiten bereitstehen.
    """
    import os
    import time
    import threading
    import logging
    import pandas as pd

    # psutil optional verwenden (CPU/RAM)
    try:
        import psutil
    except Exception:
        psutil = None

    # Globale Zustände verwenden (werden an anderer Stelle definiert)
    global all_predictions, shared_model, shared_resource_lock, PIPELINE_STATE, retraining_thread_task

    retraining_data_list = []
    inference_processor = None

    # Prozess & CPU-Kerne für Prozentberechnung
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

        # Status setzen
        with shared_resource_lock:
            PIPELINE_STATE["status"] = "inference_running"
        logging.info(f"--- INFERENCE MANAGER: Startet im Modus '{mode}' ---")

        # Inferenz-Objekt aufsetzen + Artefakte laden/übernehmen
        inference_processor = inference_class(config, folder_flag=folder_flag)
        if shared_model.get("model") is not None:
            inference_processor.set_artifacts_from_memory(shared_model)
            # 🔧 Warm-Start: DataProcessor mit initialem Fenster vorfüttern
            try:
                init_df = shared_model.get("initial_training_data")
                dp = getattr(inference_processor, "data_processor", None)
                if dp is not None and init_df is not None and hasattr(dp, "prime_buffer"):
                    want = getattr(dp, "_min_data_points", 1)
                    dp.prime_buffer(init_df.tail(want))
                    logging.info("🔧 DataProcessor warm-started mit initialem Fenster.")
            except Exception as e:
                logging.warning(f"Konnte DataProcessor nicht vorfüttern: {e}")


        elif config.get('load_id'):
            inference_processor.load_artifacts()
        else:
            raise RuntimeError("Kein trainiertes Modell und keine load_id zum Laden gefunden.")

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

                # --- KORREKTUR START: System-Ressourcen korrekt am Anfang messen ---
                cpu_t0, ram_usage_dict = None, None
                if proc is not None:
                    try:
                        cpu_t0 = proc.cpu_times()
                        # System-RAM-Auslastung über die Hilfsfunktion holen
                        ram_usage_dict = PipelineUtils.get_memory_usage()
                    except Exception:
                        cpu_t0, ram_usage_dict = None, None
                else:
                    cpu_t0, ram_usage_dict = None, None
                # --- KORREKTUR ENDE ---

                # Pause/Stop behandeln
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

                    # CPU% aus Prozess-CPU-Zeit relativ zu n_cpus
                    cpu_percent = None
                    if proc is not None and cpu_t0 is not None:
                        try:
                            cpu_t1 = proc.cpu_times()
                            cpu_used = (cpu_t1.user + cpu_t1.system) - (cpu_t0.user + cpu_t0.system)
                            cpu_percent = (cpu_used / max(step_total_time_s, 1e-12)) / n_cpus * 100.0
                        except Exception:
                            cpu_percent = None

                    # --- KORREKTUR START: Werte für CSV und Web-App aus ram_usage_dict extrahieren ---
                    ram_mb_val, ram_percent_val = None, None
                    if ram_usage_dict and ram_usage_dict.get("used_gb") != "N/A":
                        try:
                            # Für die CSV: Genutzte GB in MB umrechnen und Prozentwert extrahieren
                            ram_mb_val = float(ram_usage_dict["used_gb"]) * 1024
                            ram_percent_val = float(ram_usage_dict["percent"])
                            
                            # Für die Web-App: Das komplette Dictionary hinzufügen
                            prediction_entry['ram_usage'] = ram_usage_dict
                        except (ValueError, TypeError):
                            pass # Werte bleiben None, wenn Konvertierung fehlschlägt
                    # --- KORREKTUR ENDE ---

                    # Persistieren dieses Schritts
                    try:
                        inference_processor.save_step_result(
                            prediction_entry=prediction_entry,
                            total_time_s=step_total_time_s,
                            cpu_percent=cpu_percent,
                            # --- GEÄNDERTE ZEILEN: Korrekte RAM-Werte übergeben ---
                            ram_mb=ram_mb_val,
                            ram_percent=ram_percent_val
                        )
                    except Exception as persist_err:
                        logging.error(f"Fehler beim Speichern des Inferenz-Schritts: {persist_err}", exc_info=True)

                    # In-Memory sammeln (für finale Metriken)
                    with shared_resource_lock:
                        all_predictions.append(prediction_entry)

                    # --- NEU: Retraining-Material sammeln (nur im Modus 'retraining') ---
                    if mode == "retraining":
                        try:
                            if hasattr(payload, "to_dict"):
                                pl = payload.to_dict()
                            else:
                                pl = dict(payload)
                        except Exception:
                            pl = {}

                        # datetime sicherstellen
                        if "datetime" not in pl and "datetime" in prediction_entry:
                            pl["datetime"] = prediction_entry["datetime"]

                        retraining_data_list.append(pl)

                # Hot-Swap, wenn Retraining fertig
                with shared_resource_lock:
                    ready = PIPELINE_STATE.get("retraining_status") == "ready_to_swap"
                if ready:
                    inference_processor.set_artifacts_from_memory(shared_model)
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
    parser.add_argument('--algorithm', type=str, required=True, choices=['random_forest', 'lstm', 'cnn1d', 'xgboost'],
                        help="Zu verwendender Algorithmus.")
    parser.add_argument('--config-name', type=str, help="Optional: Name der Konfigurationsvariable.")
    parser.add_argument('--retraining', action=argparse.BooleanOptionalAction, default=False,
                        help="Aktiviert den Retraining-Modus.")
    parser.add_argument("--load_id", type=str,
                        help="Optionale Run ID zum Laden von Artefakten anstelle von Training.")
    parser.add_argument("--model_filename", type=str,
                        help="Optional: Name der zu ladenden Modelldatei.")

    # NEU: Headless/Server-Steuerung
    parser.add_argument("--web-only", action="store_true",
                        help="Nur die Flask-Weboberfläche starten und laufen lassen (simuliert den Zusatzprozess).")
    parser.add_argument("--no-web", action="store_true",
                        help="Weboberfläche deaktivieren (Headless/Batch).")
    parser.add_argument("--host", default="0.0.0.0", help="Bind-Adresse für die Web-UI.")
    parser.add_argument("--port", type=int, default=None, help="Port für die Web-UI.")
    parser.add_argument("--inference-steps", type=int, default=None,
                        help="Wenn gesetzt: so viele Inferenzschritte laufen und dann sauber beenden.")

    # NEU: Inline-Overrides (optional, mehrfach nutzbar)
    parser.add_argument("--set", action="append", default=[],
                        help="Konfigurations-Override als key=value (mehrfach möglich).")

    args = parser.parse_args()

    # Defaults für config-name
    if args.config_name is None:
        args.config_name = f"{args.algorithm}"
        logging.info(f"Kein --config-name angegeben. Verwende Default: '{args.config_name}'")

    # Konfiguration laden
    config = load_config_dynamically(args.algorithm, args.config_name)

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
        # Modell ist geladen → bereit für Inferenz
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
        # Nur Web-UI (blockierend) – simuliert den separaten Prozess in der Zielumgebung
        local_ip = PipelineUtils.get_local_ip()
        logging.info(f"\n🚀 Webserver (web-only) startet. Öffnen Sie http://{local_ip}:{port} in Ihrem Browser.")
        app.run(host=args.host, port=port, debug=False, use_reloader=False)
        return

    if not args.no_web:
        # Web-UI im Hintergrund starten (damit Pipeline/Threads weiterlaufen)
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
        # Optional warten auf Training (wenn es läuft)
        if training_thread is not None:
            training_thread.join()
        # Warten auf Inferenzende
        inference_thread.join()
        logging.info("--- Pipeline (headless) beendet. ---")
        return

    # Mit Web: Hauptthread blockiert nicht – optional hier eine einfache Keep-Alive-Schleife
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        logging.info("Beende auf Benutzerwunsch (Ctrl+C).")

if __name__ == "__main__":
    main()