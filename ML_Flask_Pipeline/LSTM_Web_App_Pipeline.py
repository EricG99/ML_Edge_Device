import sys
import os
import subprocess
import logging
import time
import numpy as np
import pandas as pd
import tensorflow as tf

# Fügt das Hauptverzeichnis zum Suchpfad hinzu, damit die Utility-Module gefunden werden.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- ML-spezifische Importe ---
import Pipeline_Utils as PipelineUtils
import Load_Prepare_Data as LoadPrepareData
import LSTM_Utils as LSTMUtils
from config import CONFIG_PATH, param_LSTM

# Kombinierte Konfiguration
CONFIG_LSTM_ALL = {**CONFIG_PATH, **param_LSTM}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Schnittstellenfunktionen, die vom Backend aufgerufen werden ---

def setup_and_train():
    """
    Schnittstelle für das Training. Wird von app.py aufgerufen.
    Passt die Datenvorbereitung und das Training für das LSTM-Modell an.
    """
    logging.info("LSTM-Pipeline: Setup und Training starten...")
    
    param_LSTM_config, paths = PipelineUtils.setup_experiment(CONFIG_LSTM_ALL)
    
    # 1. Daten als 3D-Arrays für LSTM vorbereiten
    X_train_3D, y_train_3D, X_test_3D, y_test_3D, scaler_3D, y_scaler, train_df, test_df, _, features = LoadPrepareData._prepare_base_data_3D(param_LSTM_config)
    
    # 2. LSTM-Modell trainieren
    model, history, train_time = LSTMUtils.train_model_LSTM(
        config=param_LSTM_config,
        X_train=X_train_3D,
        y_train=y_train_3D,
        features=features
    )
    
    # 3. Gib ein Dictionary mit allen Artefakten zurück, die für die Inferenz benötigt werden
    return {
        "model": model, "history": history, "train_time": train_time, 
        "param_rf": param_LSTM_config, # Behalte den Schlüssel "param_rf" für Konsistenz mit dem Backend bei
        "paths": paths,
        "X_test_3D": X_test_3D, "y_test_3D": y_test_3D, "scaler_3D": scaler_3D, 
        "test_df": test_df, "full_feature_list": features, "config": param_LSTM_config, 
        "total_steps": len(X_test_3D)
    }

def run_inference_step(artifacts, step_index):
    """
    Schnittstelle für die Inferenz. Wird von app.py in einer Schleife aufgerufen.
    Angepasst für einen einzelnen LSTM-Inferenzschritt.
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
    base_feature = config.get("base_features", [None])[0]
    target_index = features.index(base_feature)
    
    # 2. Daten für den aktuellen Schritt vorbereiten
    X_step = X_test_3D[step_index:step_index+1]
    y_step = y_test_3D[step_index:step_index+1]
    
    # 3. LSTM-Inferenz durchführen
    preds_scaled = LSTMUtils.run_inference_lstm(model=model, X_test=X_step)[0]
    
    # 4. Ergebnisse de-skalieren
    # safe_inverse_transform kann mit 1D-Arrays umgehen
    preds_descaled = PipelineUtils.safe_inverse_transform(scaler, preds_scaled, target_index)
    true_orig = PipelineUtils.safe_inverse_transform(scaler, y_step.flatten(), target_index)
    
    # 5. Zeitstempel berechnen
    current_date = test_df.index[config.get("lags", 0) + step_index]
    freq = test_df.index[1] - test_df.index[0] if len(test_df.index) > 1 else pd.Timedelta(minutes=1)
    future_dates = [current_date + (j + 1) * freq for j in range(horizon)]
    
    # 6. Ergebnis-Dictionary für das Frontend zusammenstellen
    return {
        "date": current_date.strftime('%Y-%m-%d %H:%M:%S'),
        "true_value": true_orig[0], # Nimm den ersten wahren Wert für diesen Zeitschritt
        "predicted_value_step_1": preds_descaled[0],
        "predicted_value_step_n": preds_descaled[-1],
        "future_forecast": {
            "dates": [d.strftime('%Y-%m-%d %H:%M:%S') for d in future_dates],
            "values": preds_descaled.tolist()
        }
    }


# --- Zentraler Startpunkt der Anwendung ---
if __name__ == "__main__":
    """
    Diese main-Methode startet die generische Flask-Anwendung und teilt ihr mit,
    dass sie die Logik aus DIESEM Skript verwenden soll.
    """
    logging.info("Starte die Anwendung über den LSTM-Pipeline-Launcher...")
    
    # Pfad zum generischen Flask-App-Skript (eine Ebene höher)
    flask_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Flask_App.py'))
    
    # Der Name dieses Skripts (ohne .py), der an die Flask-App übergeben wird
    pipeline_name = os.path.splitext(os.path.basename(__file__))[0]
    
    if not os.path.exists(flask_app_path):
        logging.error(f"Fehler: Das Haupt-App-Skript '{flask_app_path}' wurde nicht gefunden.")
        sys.exit(1)

    logging.info(f"Rufe '{flask_app_path}' mit Pipeline '{pipeline_name}' auf...")
    
    # Befehl zum Starten des Web-Servers
    command = [sys.executable, flask_app_path, pipeline_name]

    try:
        # Starte den Flask-Server in einem neuen Prozess
        process = subprocess.Popen(command)
        process.wait()
    except KeyboardInterrupt:
        logging.info("Anwendung wird beendet.")
        process.terminate()
    except Exception as e:
        logging.error(f"Fehler beim Starten der Anwendung: {e}")
        if 'process' in locals():
            process.terminate()

