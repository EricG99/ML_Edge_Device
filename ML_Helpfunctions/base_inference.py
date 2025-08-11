# ML_Helpfunctions/base_inference.py
import logging
import sys
import threading
import time
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from ML_Helpfunctions import Pipeline_Utils
from ML_Helpfunctions import Load_Prepare_Data
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient


class BaseInferenceProcessor(ABC):
    """
    Vereinheitlichte Inferenzbasis für 'split' (CSV) und 'live_mqtt'.
    Subklassen implementieren nur _prepare_input_data().
    """

    def __init__(self, config: dict, broker_ip: str, port: int, topic: str, folder_flag: str):
        self.config = config
        self.broker_ip = broker_ip
        self.port = port
        self.topic = topic
        self.folder_flag = folder_flag

        self.training_config = None
        self.model = None
        self.scaler = None
        self.y_scaler = None
        self.feature_list = None

        self.target_feature = self.config.get('base_features', [None])[0]
        self.inference_interval = float(self.config.get("inference_interval_sec", 1.0))
        self.inference_steps = self.config.get("inference_steps", "infinite")
        self.step_counter = 0

        self.latest_payload = None                   # wird von MQTT/Batch befüllt
        self.results_buffer: list[dict] = []         # sammelt Ergebnisse je Sekunde
        self._lock = threading.Lock()
        self._mqtt_client = None

    # -------------------- Artefakte laden --------------------

    def load_artifacts(self):
        try:
            logging.info("Lade Artefakte für Inferenz...")
            # NEU: y_scaler als Rückgabewert empfangen
            self.scaler, self.feature_list, self.model, self.training_config, self.y_scaler = \
                Pipeline_Utils.load_model_artifacts_for_inference(self.config, self.folder_flag)
            
            # Wichtiger Check: Wenn y_scaler für LSTM gebraucht wird, aber nicht da ist
            if "lstm" in self.folder_flag.lower() and self.y_scaler is None and self.config.get("scale_target"):
                 logging.error("FATAL: LSTM-Modell benötigt einen y_scaler, aber keiner wurde geladen. Beende.")
                 sys.exit(1)

            self.target_feature = self.config.get('base_features', [self.target_feature])[0]
            logging.info("✅ Artefakte geladen.")
        except Exception as e:
            logging.error(f"Artefakte konnten nicht geladen werden: {e}", exc_info=True)
            sys.exit(1)

    # -------------------- Muss die Subklasse liefern --------------------

    @abstractmethod
    def _prepare_input_data(self):
        """
        Muss ein Tupel (input_3d, timestamp, true_value) zurückgeben.
        - input_3d: np.ndarray der Form (1, lags, features)
        - timestamp: pd.Timestamp oder datetime
        - true_value: float | None
        """
        ...

    # -------------------- Hilfs-APIs --------------------

    def update_latest_data(self, data: dict):
        with self._lock:
            self.latest_payload = data

    def _should_stop(self) -> bool:
        return isinstance(self.inference_steps, int) and self.step_counter >= self.inference_steps

    # -------------------- Ein einzelner Schritt --------------------

    def _run_single_prediction(self) -> bool:
        """
        Führt genau eine Vorhersage aus. Gibt True zurück, wenn beendet werden soll.
        """
        with self._lock:
            payload_none = (self.latest_payload is None)
        if payload_none:
            return False

        input_window, ts, true_value = self._prepare_input_data()
        if input_window is None:
            return False

        # Zeitmessung + Vorhersage
        pred_scaled, inf_ms = Pipeline_Utils.run_timed_inference(self.model, input_window)
        cpu = Pipeline_Utils.get_cpu_usage()

        # Rückskalieren auf Zielgröße (multi-horizon möglich)
        try:
            target_idx = self.feature_list.index(self.target_feature) if self.feature_list else 0
        except ValueError:
            target_idx = 0

        # pred_scaled: (1, H) oder (1,) -> in (1, H) normalisieren
        pred_scaled = np.asarray(pred_scaled).reshape(1, -1)
        pred_unscaled = Pipeline_Utils.safe_inverse_transform(self.scaler, pred_scaled, target_index=target_idx).flatten()

        self.step_counter += 1
        logging.info(f"[{self.step_counter}] {ts}: {', '.join(f'{p:.4f}' for p in pred_unscaled)} | {inf_ms:.1f} ms | CPU {cpu:.1f}%")

        # Ergebnis puffern
        entry = {
            "datetime": pd.to_datetime(ts),
            "inference_time_ms": float(inf_ms),
            "cpu_load_percent": float(cpu),
            "true_value": None if true_value is None else float(true_value),
        }
        for i, v in enumerate(pred_unscaled, start=1):
            entry[f"prediction_step_{i}"] = float(v)
        self.results_buffer.append(entry)

        return self._should_stop()

    # -------------------- Strategien --------------------

    def _run_live_mode(self):
        """
        Startet MQTT und inferiert im 1-Sekunden-Takt, sobald Payloads eintreffen.
        """
        logging.info("🚀 Live-Modus (MQTT) gestartet.")
        self._mqtt_client = MqttInferenceClient(
            broker_ip=self.broker_ip,
            port=self.port,
            topic=self.topic,
            on_message=self.update_latest_data
        )
        self._mqtt_client.start()

        try:
            while True:
                start = time.perf_counter()
                should_stop = self._run_single_prediction()
                if should_stop:
                    break
                # stabile 1 Hz
                sleep_for = max(0.0, self.inference_interval - (time.perf_counter() - start))
                time.sleep(sleep_for)
        finally:
            try:
                self._mqtt_client.stop()
            except Exception:
                pass
            logging.info("MQTT-Modus beendet.")

    def _run_batch_inference(self):
        """
        CSV-Modus: „split“ – jede Sekunde wird die nächste Zeile verarbeitet (Simulation 1 Hz).
        """
        logging.info("🚀 Batch-Modus (split) gestartet.")
        df = Load_Prepare_Data.load_test_data_by_fraction(
            config=self.config,
            train_fraction=self.config.get("train_fraction", 0.7),
            make_date_as_index=True
        )
        if df.empty:
            logging.warning("Testdaten leer – nichts zu tun.")
            return

        # Zeilen in zeitlicher Reihenfolge, jede Sekunde eine Zeile einspeisen
        for _, row in df.sort_index().iterrows():
            start = time.perf_counter()
            # baue ein MQTT-ähnliches Payload-Dict
            payload = {"datetime": row.name}
            for col in df.columns:
                payload[col] = row[col]
            self.update_latest_data(payload)

            should_stop = self._run_single_prediction()
            if should_stop:
                break

            sleep_for = max(0.0, self.inference_interval - (time.perf_counter() - start))
            time.sleep(sleep_for)

    # -------------------- Ergebnisse speichern --------------------

    def _save_results(self):
        if not self.results_buffer:
            logging.info("Keine Ergebnisse zu speichern.")
            return

        res_df = pd.DataFrame(self.results_buffer).sort_values("datetime")
        horizon = self.config.get("horizon", 1)

        # Arrays für Speicher/CSV/Metriken bauen
        dates = res_df["datetime"].to_numpy()
        # Predictions zu (N, H)
        pred_cols = [f"prediction_step_{i}" for i in range(1, horizon + 1)]
        y_pred = res_df[pred_cols].to_numpy() if all(c in res_df for c in pred_cols) else res_df.filter(like="prediction_step_").to_numpy()
        # True-Werte ggf. auffüllen/tilen
        if "true_value" in res_df:
            y_true_1d = res_df["true_value"].ffill().bfill().to_numpy()
            y_true = np.tile(y_true_1d.reshape(-1, 1), reps=(1, y_pred.shape[1]))
        else:
            y_true = np.zeros_like(y_pred)

        # flach speichern (Pipeline_Utils erwartet flache Vektoren)
        y_true_flat = y_true.reshape(-1)
        y_pred_flat = y_pred.reshape(-1)

        # speichern
        try:
            out_csv = Pipeline_Utils.save_prediction_data(self.config, y_true_flat, y_pred_flat, dates)
            metrics = Pipeline_Utils.evaluate_all_metrics(y_true, y_pred, horizon=y_pred.shape[1])
            Pipeline_Utils.save_metrics_summary(metrics, self.config, getattr(self, "training_config", {}), self.config.get("paths", {}))
            logging.info(f"✅ Ergebnisse gespeichert: {out_csv}")
        except Exception as e:
            logging.error(f"Fehler beim Speichern der Ergebnisse: {e}", exc_info=True)

    # -------------------- Öffentliche API --------------------

    def run(self):
        """Startet die Inferenz je nach 'loading_strategy' und speichert danach die Ergebnisse."""
        if self.model is None or self.scaler is None or self.feature_list is None:
            self.load_artifacts()

        strategy = self.config.get("loading_strategy", "split")
        logging.info(f"Inferenz-Strategie: {strategy}")

        if strategy == "live_mqtt":
            self._run_live_mode()
        elif strategy == "split":
            self._run_batch_inference()
        else:
            logging.error(f"Unbekannte loading_strategy: {strategy}")
            return

        self._save_results()
        logging.info("🏁 Inferenz abgeschlossen.")
