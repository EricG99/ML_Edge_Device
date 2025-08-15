# rf_inference.py

import pandas as pd
import numpy as np
import logging
import sys
import os

# --- Suppress scikit-learn feature name warnings ---
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',  '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- Application Imports ---
from ML_Helpfunctions.base_inference import BaseInferenceProcessor
from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor

FOLDER_FLAG = "RandomForest"

class RFInference(BaseInferenceProcessor):
    """
    Spezialisierte Inferenzklasse für Random Forest, die den optimierten RealTimeDataProcessor verwendet.
    """

    # --- KORREKTUR 1: __init__ an die neue, schlanke Form angepasst ---
    def __init__(self, config: dict, folder_flag: str = FOLDER_FLAG):
        super().__init__(config, folder_flag)
        self.target_feature = config['base_features'][0]
        self.data_processor = RealTimeDataProcessor(config)

    def _on_artifacts_swapped(self):
        import logging, numpy as np, pandas as pd
        from ML_Helpfunctions.base_data_processing import RealTimeDataProcessor

        old_buf = None
        try:
            if getattr(self, "data_processor", None) is not None and hasattr(self.data_processor, "_buffer"):
                old_buf = self.data_processor._buffer.copy()
        except Exception:
            old_buf = None

        self.data_processor = RealTimeDataProcessor(self.config)

        try:
            if old_buf is not None and not old_buf.empty:
                max_len = getattr(self.data_processor, "_max_buffer_size", len(old_buf))
                self.data_processor._buffer = old_buf.tail(max_len)
                logging.info("RFInference: DataProcessor warm-started mit %d Zeilen aus altem Puffer.",
                            len(self.data_processor._buffer))
            else:
                logging.info("RFInference: DataProcessor neu initialisiert (kein alter Puffer verfügbar).")
        except Exception as e:
            logging.warning("RFInference: Konnte alten Puffer nicht übernehmen: %s", e)

        try:
            if getattr(self, "_last_input_data", None) is None:
                raise RuntimeError("kein _last_input_data verfügbar")

            pred_scaled, t_inf_ms = self._run_inference_unified(self._last_input_data)
            pred_unscaled = self._inverse_transform_prediction(pred_scaled)
            pred_unscaled = np.asarray(pred_unscaled, dtype=float).reshape(-1)

            H = int(self.config.get("horizon", 1))
            if pred_unscaled.size < H:
                pad = np.full(H - pred_unscaled.size, np.nan, dtype=float)
                pred_unscaled = np.concatenate([pred_unscaled, pad], axis=0)
            elif pred_unscaled.size > H:
                pred_unscaled = pred_unscaled[:H]

            # Timestamp t+1 ableiten
            if self._pending_entry and self._pending_entry.get("datetime") is not None:
                dt_next = self._pending_entry["datetime"]
            else:
                dfb = getattr(self.data_processor, "_buffer", None)
                last_ts = None
                try:
                    if dfb is not None and len(dfb) > 0:
                        if isinstance(dfb.index, pd.DatetimeIndex):
                            last_ts = dfb.index[-1]
                        elif "datetime" in dfb.columns:
                            last_ts = pd.to_datetime(dfb["datetime"].iloc[-1])
                except Exception:
                    last_ts = None
                if last_ts is None:
                    last_ts = pd.Timestamp.utcnow()
                dt_next = last_ts + pd.Timedelta(seconds=float(self.config.get("inference_interval_sec", 1.0)))

            # System-Metriken (in erwarteten Keys)
            try:
                from ML_Helpfunctions.Pipeline_Utils import PipelineUtils
                cpu = float(PipelineUtils.get_cpu_usage())
                ram = float(PipelineUtils.get_memory_usage())
            except Exception:
                cpu, ram = None, None

            self._pending_entry = {
                "datetime": dt_next,
                "prediction": float(pred_unscaled[0]) if pred_unscaled.size > 0 and np.isfinite(pred_unscaled[0]) else None,
                "true_value": None,
                "rolling_forecast": pred_unscaled.tolist(),
                "cpu_percent": cpu,
                "ram_mb": ram,
                "model_inference_time_ms": float(t_inf_ms),
                "total_processing_time_ms": 0.0,
            }
            logging.info("--- INFERENCE MANAGER: Neues Modell & Pending reseeded (Hot-Swap) ---")

        except Exception as e:
            logging.warning("RFInference: Reseed Pending nach Hot-Swap fehlgeschlagen (%s). Fallback: Pending=None", e)
            self._pending_entry = None

    
    def _run_inference_unified(self, input_data):
        """
        Führt einen einzelnen Inferenzschritt aus und gibt (y_hat_scaled, t_infer_ms) zurück.
        y_hat_scaled wird als 2D-Array (1, k) zurückgegeben.
        """
        import time
        import numpy as np
        import logging

        t0 = time.perf_counter()
        y_hat = self.model.predict(input_data)
        t_ms = (time.perf_counter() - t0) * 1000.0

        y_hat = np.asarray(y_hat)

        # Erwartet wird 2D: (1, H). Unterschiedliche Rückgaben robust auf (1, k) bringen.
        if y_hat.ndim == 1:
            y_hat = y_hat.reshape(1, -1)
        elif y_hat.ndim > 2:
            y_hat = y_hat.reshape(1, -1)

        H = int(self.config.get("horizon", 1))
        if y_hat.shape[1] == 1 and H > 1:
            logging.warning("RF predict(): erwartete H=%d, bekam (1, 1) – Modell liefert Single-Output.", H)

        return y_hat, t_ms


    def _prepare_retrain_XY(self, df_featured):
        """
        Baut X und Y (Mehrschritt-Ziele) aus einem FE-DataFrame für den RF-Retrain.
        Garantiert: Y.shape == (n_samples, H).
        """
        import os
        import joblib
        import numpy as np
        import pandas as pd

        df = df_featured.copy().sort_index().dropna()
        H = int(self.config.get("horizon", 1))

        # gespeicherte Feature-Liste laden
        features_path = os.path.join(self.config["paths"]["Models"], "features.joblib")
        features = joblib.load(features_path)

        # Zielspalte robust bestimmen
        candidates = [
            self.config.get("target_column"),
            self.config.get("target"),
            (self.config.get("base_features") or [None])[0],
            "target", "y", "value", "true_value"
        ]
        target_col = next((c.lower() for c in candidates if c and c.lower() in df.columns), None)
        if target_col is None:
            raise ValueError("RF-Retrain: Keine Zielspalte gefunden (z.B. config['target_column']).")

        # X: exakt die gespeicherten Features (Schnittmenge)
        feat_cols = [c for c in features if c in df.columns]
        if not feat_cols:
            raise ValueError("RF-Retrain: Keine der gespeicherten Features im DF gefunden.")
        X_all = df[feat_cols].to_numpy(dtype=float)

        # Y: Direkt-Mehrschritt-Ziele Y_{t+1..t+H}
        y = df[target_col].to_numpy(dtype=float)
        Y_all = np.column_stack([np.roll(y, -(k + 1)) for k in range(H)])

        # Am Ende fehlen H Zeilen → kürzen
        valid_len = len(df) - H
        if valid_len <= 0:
            raise ValueError("RF-Retrain: Zu wenig Daten nach Horizon-Ausrichtung.")

        X = X_all[:valid_len]
        Y = Y_all[:valid_len]

        # NaNs filtern (z.B. durch Rolling/Lags)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
        X = X[mask]
        Y = Y[mask]
        used_index = df.index[:valid_len][mask]

        return X, Y, used_index

    def _prepare_input_data(self, payload: dict) -> tuple[np.ndarray | None, any, float | None]:
        """
        Bereitet einen 2D-Feature-Vektor für das RF-Modell vor.
        Diese Version ist robust gegenüber Groß- und Kleinschreibung im Payload.
        """
        if not payload:
            return None, None, None

        # --- NEU: Robuste, Case-Insensitive Behandlung des Payloads ---
        # Erstelle eine temporäre Version des Payloads, bei der alle Schlüssel klein geschrieben sind.
        try:
            payload_lower = {str(k).lower(): v for k, v in payload.items()}
        except AttributeError:
            logging.error("Fehler beim Konvertieren der Payload-Schlüssel in Kleinbuchstaben. Ist das Payload ein Dictionary?")
            return None, None, None
            
        # Der Datenprozessor erhält das Payload mit den nun garantierten Kleinbuchstaben-Schlüsseln.
        # Wichtig: Der data_processor muss so konfiguriert sein, dass er ebenfalls mit Kleinbuchstaben-Features arbeitet.
        featured_buffer = self.data_processor.update_and_process(payload_lower)
        
        if featured_buffer is None or featured_buffer.empty:
            return None, None, None

        # Letzten Vektor extrahieren
        last_vector_full = featured_buffer[self.feature_list].iloc[-1:]

        if last_vector_full.isnull().values.any():
            logging.warning("NaNs im finalen Inferenz-Vektor entdeckt. Überspringe Schritt.")
            return None, None, None

        # Skalieren
        X_live_scaled = self.scaler.transform(last_vector_full.values) if self.scaler else last_vector_full.values
        
        timestamp = last_vector_full.index[-1]
        
        # --- KORREKTUR: Suche in dem Dictionary mit den Kleinbuchstaben-Schlüsseln ---
        key_to_find = self.target_feature.lower()
        true_value = payload_lower.get(key_to_find)

        # Der Debug-Logger bleibt für den Fall, dass die Spalte komplett fehlt
        if true_value is None:
            logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            logging.warning(f"FEHLER: 'true_value' konnte auch nach Umwandlung in Kleinbuchstaben nicht gefunden werden!")
            logging.warning(f"--> Gesuchter Schlüssel: '{key_to_find}'")
            available_keys = list(payload_lower.keys())
            logging.warning(f"--> Verfügbare Schlüssel (klein, Auszug): {available_keys[:10]}")
            logging.warning("--> Bitte prüfen: Ist die Spalte in der CSV/MQTT-Quelle überhaupt vorhanden?")
            logging.warning("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        
        return X_live_scaled, timestamp, true_value
    # --- KORREKTUR 2: Fehlende abstrakte Methode implementiert ---
    def _inverse_transform_prediction(self, prediction_scaled: np.ndarray) -> np.ndarray:
        """
        Für Random Forest sind die Vorhersagen bereits im korrekten, unskalierten Raum.
        Daher geben wir die Vorhersage einfach unverändert zurück.
        """
        return np.asarray(prediction_scaled).flatten()

# Der __main__-Block kann für Standalone-Tests bleiben, wird aber von der pipeline_web_app nicht genutzt.
if __name__ == "__main__":
    from config.config_ml_random_forest import random_forest
    from config.config_general import CONFIG_PATH, MQTT_CONFIG, CONFIG_LOAD_ARTIFACTS
    import argparse 

    parser = argparse.ArgumentParser(description="Standalone Random Forest Inference")
    parser.add_argument("--load_id", type=str, help="Optional: The specific run ID to load artifacts from.")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("--- MODE: Standalone RF Inference (Console Output) ---")
    
    infer_config = random_forest.copy()
    infer_config.update(CONFIG_LOAD_ARTIFACTS)
    infer_config['paths'] = CONFIG_PATH['paths']
    
    if args.load_id:
        infer_config['load_id'] = args.load_id
        infer_config['inference_mode'] = 'load_artifacts_path'
    
    # Erstellen der Instanz ohne MQTT-Parameter, da diese aus der Config kommen
    processor = RFInference(config=infer_config)
    
    # Die .run() Methode muss noch an die neue Iterator-Logik angepasst werden,
    # aber für den Web-App-Kontext ist das nicht notwendig.
    logging.info("Standalone-Ausführung beendet. Für den Pipeline-Betrieb ist dies korrekt.")