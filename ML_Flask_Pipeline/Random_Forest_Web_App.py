import sys
import os
import subprocess
import logging
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Fügt das Hauptverzeichnis zum Suchpfad hinzu, damit die Utility-Module gefunden werden.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ML-spezifische Importe
import Pipeline_Utils as PipelineUtils
import Load_Prepare_Data as LoadPrepareData
import RF_Run_Pipeline as RFRunPipeline
import RF_Utils as RFUtils  # Direkter Import von RF_Utils
from config import CONFIG_PATH, param_rf

# Kombinierte Konfiguration
CONFIG_RF_ALL = {**CONFIG_PATH, **param_rf}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Schnittstellenfunktionen, die vom Backend aufgerufen werden ---

# def setup_and_train():
#     """
#     Schnittstelle für das Training. Verwendet jetzt direkt die Funktionen aus den Utility-Skripten.
#     """
#     logging.info("RF-Pipeline: Setup und Training starten...")
#     param_rf_config, paths = PipelineUtils.setup_experiment(CONFIG_RF_ALL)
#     X_train, y_train, X_test, y_test, scaler, _, _, test_df, _, features = LoadPrepareData._prepare_base_data_2D(param_rf_config)
    
#     # KORREKTUR: Ruft die Funktion direkt aus RF_Utils auf, wie im Beispielskript.
#     model, train_time = RFUtils.train_random_forest_model(
#         config=param_rf_config, 
#         X_train=X_train, 
#         y_train=y_train, 
#         features=features
#     )
    
#     return {
#         "model": model, "train_time": train_time, "param_rf": param_rf_config, "paths": paths,
#         "X_test_2D": X_test, "y_test_2D": y_test, "scaler_2D": scaler, "test_df": test_df,
#         "full_feature_list": features, "config": param_rf_config, "total_steps": len(X_test)
#     }

def setup_and_train():
    """
    Schnittstelle für das Training. Verwendet die gekapselte Pipeline-Funktion 
    aus RF_Run_Pipeline.py.
    """
    logging.info("RF-Pipeline: Gekapseltes Setup und Training starten...")

    # Ruft die Haupt-Setup- und Trainingsfunktion aus dem RF_Run_Pipeline-Skript auf.
    # Diese Funktion führt das Setup, die Datenvorbereitung und das Training durch.
    (
        model, train_time, param_rf_config, paths, 
        X_train, y_train, X_test, y_test, 
        scaler, test_df, features
    ) = RFRunPipeline.setup_and_train_rf_model(CONFIG_RF_ALL)

    # Stelle das Dictionary mit den für die Inferenz benötigten Artefakten zusammen.
    return {
        "model": model, 
        "train_time": train_time, 
        "param_rf": param_rf_config, 
        "paths": paths,
        "X_test_2D": X_test, 
        "y_test_2D": y_test, 
        "scaler_2D": scaler, 
        "test_df": test_df,
        "full_feature_list": features, 
        "config": param_rf_config, 
        "total_steps": len(X_test)
    }

def run_inference_step(artifacts, step_index):
    """
    Korrigierte Schnittstelle für die Inferenz mit spezieller Behandlung für horizon=1
    """
    model, X_test, y_test, scaler, test_df, config, features = (
        artifacts["model"], artifacts["X_test_2D"], artifacts["y_test_2D"], artifacts["scaler_2D"],
        artifacts["test_df"], artifacts["param_rf"], artifacts["full_feature_list"]
    )
    horizon = config.get("horizon", 1)
    base_feature = config.get("base_features", [None])[0]
    target_index = features.index(base_feature)
    
    X_step = X_test[step_index:step_index+1]
    
    # Spezielle Behandlung für y_test basierend auf horizon
    if horizon == 1:
        # Bei horizon=1 ist y_test 1D, wir brauchen einen 2D-Array für die inverse Transformation
        y_for_inverse = np.array([[y_test[step_index]]])  # Doppelte Klammern für 2D
    else:
        # Bei horizon>1 nehmen wir den ersten Wert des Horizonts für t+1
        y_step = y_test[step_index:step_index+1]
        y_for_inverse = y_step[:, 0].reshape(-1, 1)  # Sicherstellen dass es 2D ist

    # Führe Vorhersage durch
    preds_scaled = RFUtils.run_inference_random_forest(model=model, X_test=X_step)
    
    # Spezielle Behandlung der Vorhersage basierend auf horizon
    if horizon == 1:
        # Bei horizon=1 ist die Ausgabe ein 1D-Array, wir machen es 2D
        preds_scaled = np.array([preds_scaled]) if preds_scaled.ndim == 1 else preds_scaled
        preds_descaled = PipelineUtils.safe_inverse_transform(scaler, preds_scaled, target_index)
    else:
        preds_descaled = PipelineUtils.safe_inverse_transform(scaler, preds_scaled[0], target_index)

    # Berechne Zeitstempel
    current_date = test_df.index[config.get("lags", 0) + step_index]
    freq = test_df.index[1] - test_df.index[0] if len(test_df.index) > 1 else pd.Timedelta(minutes=1)
    future_dates = [current_date + (j + 1) * freq for j in range(horizon)]
    
    # Vorbereitung der Rückgabewerte
    result = {
        "date": current_date.strftime('%Y-%m-%d %H:%M:%S'),
        "true_value": PipelineUtils.safe_inverse_transform(scaler, y_for_inverse, target_index).item(),
        "future_forecast": {
            "dates": [d.strftime('%Y-%m-%d %H:%M:%S') for d in future_dates],
            "values": preds_descaled.flatten().tolist()  # Sicherstellen dass es eine flache Liste ist
        }
    }

    # Behandlung der Vorhersagewerte basierend auf horizon
    if horizon == 1:
        result["predicted_value_step_1"] = preds_descaled.item()
        result["predicted_value_step_n"] = preds_descaled.item()
    else:
        result["predicted_value_step_1"] = preds_descaled[0].item()
        result["predicted_value_step_n"] = preds_descaled[-1].item()

    return result

# --- Zentraler Startpunkt der Anwendung ---
if __name__ == "__main__":
    """
    Diese main-Methode startet die generische Flask-Anwendung und teilt ihr mit,
    dass sie die Logik aus DIESEM Skript verwenden soll.
    """
    logging.info("Starte die Anwendung über den RF-Pipeline-Launcher...")
    
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
