# LSTM_Web_App.py
import sys
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf # Behalte TensorFlow-Import für LSTM-Modelle

# Pfade anpassen, um lokale Module und Flask_App.py zu finden.
# Annahme: Flask_App.py liegt im übergeordneten Verzeichnis von ML_Flask_Pipeline.
# Annahme: Utility-Module (Pipeline_Utils etc.) liegen im selben Verzeichnis wie dieses Skript.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Für Flask_App.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))     # Für andere Module im selben Ordner

# Importiere das Flask_App Modul
import Flask_App

# ML-spezifische Importe
import Pipeline_Utils as PipelineUtils
import Load_Prepare_Data as LoadPrepareData
import LSTM_Utils as LSTMUtils
import LSTM_run_Pipeline as LSTMRunPipeline 
from config import CONFIG_PATH, param_LSTM

# Kombinierte Konfiguration
CONFIG_LSTM_ALL = {**CONFIG_PATH, **param_LSTM}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Schnittstellenfunktionen, die vom Flask-Backend aufgerufen werden ---

def setup_and_train_lstm():
    """
    Implementierung der Schnittstelle für das Training.
    Passt die Datenvorbereitung und das Training für das LSTM-Modell an.

    Returns:
        dict: Ein Dictionary mit den für die Inferenz benötigten Artefakten
              (Modell, Scaler, Testdaten, Konfiguration etc.).
    """
    logging.info("LSTM-Pipeline: Setup und Training starten...")
    
    model, train_time, param_LSTM, paths, \
    X_train_3D, y_train_3D, X_test_3D, y_test_3D, \
    scaler_3D, test_df, full_feature_list, history = LSTMRunPipeline.setup_and_train_lstm_model(CONFIG_LSTM_ALL)
    
    
    # 3. Gib ein Dictionary mit allen Artefakten zurück, die für die Inferenz benötigt werden
    return {
        "model": model, "history": history, "train_time": train_time, 
        "param_config": param_LSTM, # Schlüssel zu "param_config" geändert
        "paths": paths,
        "X_test_3D": X_test_3D, "y_test_3D": y_test_3D, "scaler_3D": scaler_3D, 
        "test_df": test_df, "full_feature_list": full_feature_list, "config": param_LSTM, 
        "total_steps": len(X_test_3D) # Sicherstellen, dass dies korrekt ist für LSTM-Schritte
    }

def run_inference_step_lstm(artifacts, step_index):
    """
    Implementierung der Schnittstelle für die Inferenz für einen einzelnen Schritt.
    Angepasst für einen einzelnen LSTM-Inferenzschritt.

    Args:
        artifacts (dict): Das Dictionary von Artefakten, das von setup_and_train_lstm zurückgegeben wurde.
        step_index (int): Der aktuelle Schrittindex für die Inferenz.

    Returns:
        dict: Ein Dictionary mit den Ergebnissen des Inferenzschritts,
              einschließlich Datum, tatsächlichem Wert und Prognose.
    """
    # 1. Benötigte Artefakte aus dem Dictionary extrahieren
    model = artifacts["model"]
    X_test_3D = artifacts["X_test_3D"]
    y_test_3D = artifacts["y_test_3D"]
    scaler = artifacts["scaler_3D"]
    test_df = artifacts["test_df"]
    config = artifacts["config"] # Nutze das volle config-Dict
    features = artifacts["full_feature_list"]
    
    horizon = config.get("horizon", 1)
    
    # Sicherstellen, dass 'base_features' existiert und der erste Eintrag gültig ist
    base_feature = config.get("base_features", [None])
    if not base_feature or base_feature[0] is None:
        logging.error("Konfigurationsfehler: 'base_features' muss spezifiziert sein und darf nicht leer sein.")
        raise ValueError("base_features muss in der Konfiguration spezifiziert sein und darf nicht leer sein.")
    base_feature_name = base_feature[0]
    
    # Prüfen, ob der Basis-Feature-Name in der Feature-Liste vorhanden ist
    if base_feature_name not in features:
        logging.error(f"Feature '{base_feature_name}' nicht in der Feature-Liste gefunden: {features}")
        raise ValueError(f"Basis-Feature '{base_feature_name}' nicht in der Liste der Features gefunden.")
    target_index = features.index(base_feature_name)
    
    # 2. Daten für den aktuellen Schritt vorbereiten
    if step_index >= len(X_test_3D):
        logging.warning(f"Inferenzschritt {step_index} liegt außerhalb der Grenzen von X_test_3D (Länge {len(X_test_3D)}).")
        return {
            "date": "N/A",
            "true_value": None,
            "future_forecast": {"dates": [], "values": []},
            "predicted_value_step_1": None,
            "predicted_value_step_n": None
        }

    X_step = X_test_3D[step_index:step_index+1] # Sollte (1, timesteps, num_features) sein
    y_step_true = y_test_3D[step_index:step_index+1] # Sollte (1, horizon) oder (1, horizon, 1) sein

    # 3. LSTM-Inferenz durchführen
    # LSTMUtils.run_inference_lstm sollte einen Array der Form (1, horizon) oder (1, horizon, 1) zurückgeben.
    # [0] wird verwendet, um die Batch-Dimension zu entfernen, was ein Array der Form (horizon,) oder (horizon, 1) hinterlässt.
    preds_scaled = LSTMUtils.run_inference_lstm(model=model, X_test=X_step)[0] 
    
    # 4. Ergebnisse de-skalieren
    # Wenn preds_scaled (horizon,) ist, wandelt reshape(-1, 1) es in (horizon, 1) um.
    # Wenn preds_scaled (horizon, 1) ist, bleibt es so.
    preds_descaled = PipelineUtils.safe_inverse_transform(scaler, preds_scaled.reshape(-1, 1), target_index)
    
    # Der wahre Wert muss auch descaliert werden. y_step_true sollte (1, horizon) oder (1, horizon, 1) sein.
    # Wir nehmen an, dass der "wahre Wert" der Wert für t+1 ist (erste im Horizont).
    # reshape(-1, 1) stellt sicher, dass es 2D ist für safe_inverse_transform.
    true_orig = PipelineUtils.safe_inverse_transform(scaler, y_step_true[0, 0].reshape(-1, 1), target_index).item()
    
    # 5. Zeitstempel berechnen
    # Lags werden zur step_index hinzugefügt, um den korrekten Startpunkt im originalen test_df zu finden.
    data_start_index = config.get("lags", 0)
    if (data_start_index + step_index) >= len(test_df.index):
        logging.warning(f"Datum für Schritt {step_index} (mit Lags {data_start_index}) liegt außerhalb des Test-DF-Index.")
        # Fallback für den Fall, dass der Index außerhalb der Grenzen liegt.
        return {
            "date": "N/A",
            "true_value": None,
            "future_forecast": {"dates": [], "values": []},
            "predicted_value_step_1": None,
            "predicted_value_step_n": None
        }

    current_date = test_df.index[data_start_index + step_index]
    freq = test_df.index[1] - test_df.index[0] if len(test_df.index) > 1 else pd.Timedelta(minutes=1)
    future_dates = [current_date + (j + 1) * freq for j in range(horizon)]
    
    # 6. Ergebnis-Dictionary für das Frontend zusammenstellen
    preds_descaled_list = preds_descaled.flatten().tolist()
    
    return {
        "date": current_date.strftime('%Y-%m-%d %H:%M:%S'),
        "true_value": true_orig,
        "predicted_value_step_1": preds_descaled_list[0] if preds_descaled_list else None,
        "predicted_value_step_n": preds_descaled_list[-1] if preds_descaled_list else None,
        "future_forecast": {
            "dates": [d.strftime('%Y-%m-%d %H:%M:%S') for d in future_dates],
            "values": preds_descaled_list
        }
    }


# --- Zentraler Startpunkt der Anwendung ---
if __name__ == "__main__":
    logging.info("Starte die LSTM ML-Anwendung (als Hauptskript)...")
    
    try:
        # Initialisiere die Flask-App mit den ML-Pipeline-Funktionen dieses Skripts
        Flask_App.initialize_flask_app(
            setup_and_train_func=setup_and_train_lstm,
            run_inference_step_func=run_inference_step_lstm
        )
        # Starte den Flask-Server
        Flask_App.run_flask_server(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e:
        logging.error(f"Fehler beim Starten der Anwendung: {e}", exc_info=True)
        sys.exit(1)






