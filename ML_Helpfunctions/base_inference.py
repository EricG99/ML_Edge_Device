import logging
import sys
import os
import threading
import time
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from ML_Helpfunctions import Pipeline_Utils
from ML_Helpfunctions import Load_Prepare_Data 
from ML_Helpfunctions import Feature_Engeneering as fe
from ML_Helpfunctions.MQTT_Client import MqttInferenceClient


class BaseInferenceProcessor(ABC):
    """
    Abstrakte Basisklasse für Inferenz-Pipelines.
    KORRIGIERT: Unterstützt Live- (MQTT) und Batch-Inferenz (CSV) mit korrekter,
    modell-spezifischer Logik.
    """
    def __init__(self, config: dict, broker_ip: str, port: int, topic: str, folder_flag:str):
        self.config = config
        self.broker_ip = broker_ip
        self.port = port
        self.topic = topic
        self.folder_flag = folder_flag
        
        # NEU: Unterscheidung zwischen Modelltypen für korrekte Verarbeitung
        self.is_lstm = "lstm" in self.config.get("algorithm", "").lower()
        
        self.training_config = None
        self.model, self.scaler, self.feature_list = None, None, None
        self.target_feature = None
        self.is_running = False
        self.lock = threading.Lock()
        
        self.latest_payload = None
        self.results_buffer = []
        self.step_counter = 0

    def load_artifacts(self):
        """Lädt die für die Inferenz benötigten Artefakte."""
        try:
            logging.info("Lade Inferenz-Artefakte...")
            self.scaler, self.feature_list, self.model, self.training_config = \
                Pipeline_Utils.load_model_artifacts_for_inference(self.config, self.folder_flag)
            
            # KORREKTUR: Verwende die Trainings-Konfiguration, um die Zielvariable zu bestimmen
            config_to_use = self.training_config if self.training_config else self.config
            self.target_feature = config_to_use.get('base_features', [None])[0]
            
            logging.info("✅ Artefakte erfolgreich geladen.")
            logging.info(f"Modell-Typ erkannt: {'LSTM' if self.is_lstm else 'Tree-based (z.B. RF)'}")
            logging.info(f"Zielvariable (target_feature): '{self.target_feature}'")

        except (FileNotFoundError, ValueError) as e:
            logging.error(f"FATAL: Fehler beim Laden der Artefakte: {e}", exc_info=True)
            sys.exit(1)

    @abstractmethod
    def _prepare_input_data(self):
        """Muss von der Subklasse implementiert werden (für Live-Inferenz)."""
        pass

    def update_latest_data(self, data: dict):
        """Callback, um die neueste MQTT-Nachricht zu erhalten."""
        with self.lock:
            self.latest_payload = data

    def _run_single_prediction(self):
        """
        Führt eine einzelne Vorhersage für den LIVE-Modus aus.
        Gibt True zurück, wenn die Inferenz gestoppt werden soll.
        """
        if self.latest_payload is None:
            return False

        # _prepare_input_data wird von der Subklasse (z.B. LSTMInference) implementiert
        # und bereitet das Input-Fenster für ein einzelnes Sample vor.
        input_data, timestamp, true_value = self._prepare_input_data()

        if input_data is not None:
            prediction_raw, inference_time_ms = Pipeline_Utils.run_timed_inference(
                model=self.model, input_data=input_data
            )
            
            # KORREKTUR: Wende die inverse Transformation NUR für LSTM an
            if self.is_lstm:
                target_index = self.feature_list.index(self.target_feature)
                prediction_unscaled = Pipeline_Utils.safe_inverse_transform(
                    scaler=self.scaler, array=prediction_raw.reshape(1, -1), target_index=target_index
                )
            else:
                # RF-Modelle sagen bereits den unskalierten Wert voraus
                prediction_unscaled = prediction_raw

            prediction_list = prediction_unscaled.flatten().tolist()
            self.step_counter += 1
            logging.info(f"Step {self.step_counter}: Prediction for {timestamp.isoformat()}: {[f'{p:.2f}' for p in prediction_list]}")

            result_entry = { "datetime": timestamp, "true_value": true_value }
            for i, pred_val in enumerate(prediction_list):
                result_entry[f"prediction_step_{i+1}"] = pred_val
            self.results_buffer.append(result_entry)
            
            inference_steps = self.config.get("inference_steps", "infinite")
            if isinstance(inference_steps, int) and self.step_counter >= inference_steps:
                logging.info(f"--- ✅ Konfiguriertes Inferenz-Limit von {inference_steps} Schritten erreicht. ---")
                return True # Signal zum Stoppen
        return False

    def _live_inference_loop(self):
        """Die Kernschleife, die periodisch die Live-Inferenz ausführt."""
        self.is_running = True
        logging.info("🚀 Starte Live-Inferenz-Schleife...")
        while self.is_running:
            start_time = time.time()
            with self.lock:
                should_stop = self._run_single_prediction()
            if should_stop:
                self.is_running = False
            elapsed = time.time() - start_time
            sleep_for = max(0, self.config.get("inference_interval_sec", 1.0) - elapsed)
            time.sleep(sleep_for)
        logging.info("Live-Inferenz-Schleife gestoppt.")

    def _run_batch_inference(self):
        """
        NEUE VERSION: Führt eine Batch-Inferenz auf dem gesamten Test-Split aus.
        Diese Methode ist vektorisiert, effizient und korrekt.
        """
        logging.info("--- 🚀 Starte NEUE BATCH-Inferenz auf Test-Split aus CSV ---")
        
        # 1. Lade und bereite den gesamten Testdatensatz vor
        test_df = Load_Prepare_Data.load_test_data_by_fraction(
            config=self.config,
            train_fraction=self.config.get("train_fraction", 0.75),
            make_date_as_index=True 
        )
        featured_test_df, _ = fe.add_all_features(test_df, self.config)
        featured_test_df.dropna(inplace=True)

        if featured_test_df.empty:
            logging.warning("Test-DataFrame ist nach Feature Engineering leer. Breche ab.")
            return

        # Wahre Werte und Zeitstempel extrahieren, bevor sie durch Windowing verloren gehen
        y_true_full = featured_test_df[self.target_feature].values
        dates_full = featured_test_df.index

        # 2. Vorhersage basierend auf dem Modelltyp durchführen
        if self.is_lstm:
            logging.info("Verarbeite Daten für LSTM (skalieren, fenstern, vorhersagen, invers-transformieren)")
            # Alle Features (inkl. Zielvariable) mit dem gelernten Skalierer transformieren
            scaled_test_data = self.scaler.transform(featured_test_df[self.feature_list])
            
            # Testdaten in 3D-Fenster umwandeln
            X_test, _ = Load_Prepare_Data.convert_data_to_sliding_window(
                scaled_test_data,
                lag_horizon=self.config["lags"],
                forecast_horizon=self.config["horizon"]
            )
            
            if X_test.shape[0] == 0:
                logging.warning("Nicht genügend Testdaten, um ein einziges Fenster zu erstellen.")
                return

            # Vorhersagen für den gesamten Batch durchführen
            predictions_scaled = self.model.predict(X_test, batch_size=self.config.get("batch_size", 32))
            
            # Inverse Transformation auf den Vorhersagen durchführen
            target_index = self.feature_list.index(self.target_feature)
            predictions_unscaled = Pipeline_Utils.safe_inverse_transform(self.scaler, predictions_scaled, target_index)

            # y_true und dates an die Fenster anpassen
            start_index = self.config["lags"]
            y_true_aligned = y_true_full[start_index : start_index + len(X_test)]
            dates_aligned = dates_full[start_index : start_index + len(X_test)]

        else: # Random Forest oder ähnliches Modell
            logging.info("Verarbeite Daten für RF (X skalieren, vorhersagen)")
            # self.feature_list enthält nur die X-Features (sichergestellt durch DataPipeline2D)
            X_test_df = featured_test_df[self.feature_list]
            X_test_scaled = self.scaler.transform(X_test_df)
            
            # Vorhersagen sind direkt die unskalierten Werte
            predictions_unscaled = self.model.predict(X_test_scaled)
            
            # Sicherstellen, dass die Vorhersage 2D ist
            if predictions_unscaled.ndim == 1:
                predictions_unscaled = predictions_unscaled.reshape(-1, 1)

            # y_true und dates an die Vorhersagelänge anpassen
            y_true_aligned = y_true_full[:len(predictions_unscaled)]
            dates_aligned = dates_full[:len(predictions_unscaled)]

        logging.info(f"Batch-Vorhersage abgeschlossen. {len(predictions_unscaled)} Vorhersagen generiert.")
        # 3. Ergebnisse im Puffer speichern
        for i in range(len(predictions_unscaled)):
            result_entry = {
                "datetime": dates_aligned[i],
                "true_value": y_true_aligned[i],
            }
            # Multi-Step-Vorhersagen verarbeiten
            for step, pred_val in enumerate(predictions_unscaled[i]):
                 result_entry[f"prediction_step_{step+1}"] = pred_val
            self.results_buffer.append(result_entry)

    def _run_live_mode(self):
        """Startet den gesamten Inferenzprozess für den Live-Modus."""
        inference_thread = threading.Thread(target=self._live_inference_loop, daemon=True)
        mqtt_client = MqttInferenceClient(
            broker_ip=self.broker_ip, port=self.port, topic=self.topic,
            on_message_callback=self.update_latest_data
        )
        try:
            logging.info("🚀 Starte MQTT-Client für LIVE-Inferenz...")
            mqtt_client.run()
            inference_thread.start()
            # Hauptthread wartet, bis der Inferenz-Thread endet (z.B. durch user input oder limit)
            inference_thread.join() 
        except KeyboardInterrupt:
            logging.info("\n🚨 Shutdown-Signal (KeyboardInterrupt) empfangen.")
        finally:
            self.is_running = False
            if inference_thread.is_alive():
                inference_thread.join()
            mqtt_client.client.loop_stop()
            mqtt_client.client.disconnect()
            logging.info("MQTT-Client gestoppt.")

    def _save_results(self):
        """Speichert die gesammelten Ergebnisse am Ende des Laufs."""
        if not self.results_buffer:
            logging.warning("Keine Ergebnisse gesammelt, nichts zu speichern.")
            return

        logging.info(f"\n💾 Speichere {len(self.results_buffer)} gesammelte Inferenz-Ergebnisse...")
        self.config, paths = Pipeline_Utils.setup_experiment(self.config, self.folder_flag, run_type='inference')
        results_df = pd.DataFrame(self.results_buffer)

        pred_filename = f"inference_results_{self.config.get('loading_strategy')}_{self.config['run_id']}.csv"
        pred_output_path = os.path.join(paths.get("Prediction_Data"), pred_filename)
        results_df.to_csv(pred_output_path, index=False, date_format='%Y-%m-%dT%H:%M:%S.%f')
        logging.info(f"✅ Detaillierte Vorhersage-Ergebnisse gespeichert in: {pred_output_path}")

        # Metriken berechnen und speichern
        if "true_value" in results_df.columns and "prediction_step_1" in results_df.columns:
            results_df.dropna(subset=["true_value", "prediction_step_1"], inplace=True)
            y_true = results_df["true_value"].to_numpy()
            y_pred = results_df["prediction_step_1"].to_numpy()
            if len(y_true) > 0:
                metrics = Pipeline_Utils.evaluate_all_metrics(y_true=y_true, y_pred=y_pred)
                Pipeline_Utils.save_metrics_summary(metrics=metrics, infer_config=self.config, train_config=self.training_config, paths=paths)
        else:
            logging.warning("Metriken konnten nicht berechnet werden: 'true_value' oder 'prediction_step_1' fehlen.")

    def run(self):
        """Startet den gesamten Inferenzprozess basierend auf der Lade-Strategie."""
        self.load_artifacts()

        strategy = self.config.get("loading_strategy")
        logging.info(f"Führe Inferenz mit Strategie aus: '{strategy}'")

        if strategy == "split":
            self._run_batch_inference()
        elif strategy == "live_mqtt":
            self._run_live_mode()
        else:
            logging.error(f"Unbekannte loading_strategy: '{strategy}'. Breche ab.")
            return
        
        self._save_results()
        logging.info("✅ Inferenz-Aufgabe abgeschlossen.")