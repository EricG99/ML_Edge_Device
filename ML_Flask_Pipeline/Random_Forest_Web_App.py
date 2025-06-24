# Random_Forest_Web_App.py
import sys
import os
import logging
import numpy as np
import pandas as pd
# Keine Notwendigkeit für subprocess mehr

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
import RF_Run_Pipeline as RFRunPipeline
import RF_Utils as RFUtils
from config import CONFIG_PATH, param_rf

# Kombinierte Konfiguration
CONFIG_RF_ALL = {**CONFIG_PATH, **param_rf}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Schnittstellenfunktionen, die vom Flask-Backend aufgerufen werden ---

def setup_and_train_rf():
    """
    Implementierung der Schnittstelle für das Training.
    Verwendet die gekapselte Pipeline-Funktion aus RF_Run_Pipeline.py.

    Returns:
        dict: Ein Dictionary mit den für die Inferenz benötigten Artefakten
              (Modell, Scaler, Testdaten, Konfiguration etc.).
    """
    logging.info("RF-Pipeline: Gekapseltes Setup und Training starten...")

    (
        model, train_time, param_rf_config, paths,
        X_train, y_train, X_test, y_test,
        scaler, test_df, features
    ) = RFRunPipeline.setup_and_train_rf_model(CONFIG_RF_ALL)

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

def run_inference_step_rf(artifacts, step_index):
    """
    Implementierung der Schnittstelle für die Inferenz für einen einzelnen Schritt.

    Args:
        artifacts (dict): Das Dictionary von Artefakten, das von setup_and_train_rf zurückgegeben wurde.
        step_index (int): Der aktuelle Schrittindex für die Inferenz.

    Returns:
        dict: Ein Dictionary mit den Ergebnissen des Inferenzschritts,
              einschließlich Datum, tatsächlichem Wert und Prognose.
    """
    model, X_test, y_test, scaler, test_df, config, features = (
        artifacts["model"], artifacts["X_test_2D"], artifacts["y_test_2D"], artifacts["scaler_2D"],
        artifacts["test_df"], artifacts["param_rf"], artifacts["full_feature_list"]
    )
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

    # Sicherstellen, dass der Schritt-Index innerhalb der Grenzen liegt
    if step_index >= len(X_test):
        logging.warning(f"Inferenzschritt {step_index} liegt außerhalb der Grenzen von X_test (Länge {len(X_test)}).")
        # Geeignete Fehlerbehandlung oder leere Daten zurückgeben
        return {
            "date": "N/A",
            "true_value": None,
            "future_forecast": {"dates": [], "values": []},
            "predicted_value_step_1": None,
            "predicted_value_step_n": None
        }

    X_step = X_test[step_index:step_index+1] # Nehmen den aktuellen Datenpunkt für die Vorhersage
    
    # Spezielle Behandlung für y_test basierend auf horizon
    if horizon == 1:
        # Bei horizon=1 ist y_test 1D, wir brauchen einen 2D-Array für die inverse Transformation
        y_for_inverse = np.array([[y_test[step_index]]])  # Doppelte Klammern für 2D-Format
    else:
        # Bei horizon>1 nehmen wir den ersten Wert des Horizonts für t+1 als wahren Wert
        # y_test ist hier (N_samples, horizon)
        y_step = y_test[step_index:step_index+1] # Dies ist (1, horizon)
        y_for_inverse = y_step[:, 0].reshape(-1, 1) # Nur den ersten Wert (für t+1) als 2D-Array

    # Führe Vorhersage durch
    preds_scaled = RFUtils.run_inference_random_forest(model=model, X_test=X_step)
    
    # Spezielle Behandlung der Vorhersage basierend auf horizon
    if horizon == 1:
        # Bei horizon=1 ist die Ausgabe des Modells ein Skalar oder 1D-Array; mache es 2D für inverse_transform
        preds_scaled = np.array([preds_scaled]) if preds_scaled.ndim == 1 else preds_scaled
        preds_descaled = PipelineUtils.safe_inverse_transform(scaler, preds_scaled, target_index)
    else:
        # Für horizon > 1 ist preds_scaled wahrscheinlich (1, horizon)
        # safe_inverse_transform erwartet (n_samples, n_features). Hier ist es 1 Sample und 'horizon' "Features" (Prognosen)
        # Wir müssen sicherstellen, dass die Form passt. preds_scaled[0] ist ein 1D-Array der Länge 'horizon'.
        # reshape es zu (1, horizon) für safe_inverse_transform, wenn es eine skalare Inversion pro "Feature" macht.
        # Beachten Sie: Wenn der Scaler nur für ein einzelnes Feature trainiert wurde, ist dies möglicherweise keine ideale Nutzung.
        # Es wird davon ausgegangen, dass safe_inverse_transform diese Multi-Output-Prediction korrekt behandelt.
        preds_descaled = PipelineUtils.safe_inverse_transform(scaler, preds_scaled[0].reshape(1, -1), target_index)


    # Berechne Zeitstempel für den aktuellen Wert und die zukünftigen Prognosen
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
    # Frequenz berechnen, um zukünftige Datenpunkte zu bestimmen
    freq = test_df.index[1] - test_df.index[0] if len(test_df.index) > 1 else pd.Timedelta(minutes=1)
    future_dates = [current_date + (j + 1) * freq for j in range(horizon)]
    
    # Vorbereitung der Rückgabewerte
    result = {
        "date": current_date.strftime('%Y-%m-%d %H:%M:%S'),
        # .item() wird verwendet, um einen Skalar aus einem 0D- oder 1D-Array zu erhalten
        "true_value": PipelineUtils.safe_inverse_transform(scaler, y_for_inverse, target_index).item(),
        "future_forecast": {
            "dates": [d.strftime('%Y-%m-%d %H:%M:%S') for d in future_dates],
            "values": preds_descaled.flatten().tolist()  # Sicherstellen dass es eine flache Liste ist
        }
    }

    # Behandlung der einzelnen Vorhersagewerte basierend auf horizon
    if horizon == 1:
        result["predicted_value_step_1"] = preds_descaled.item()
        result["predicted_value_step_n"] = preds_descaled.item()
    else:
        preds_list = preds_descaled.flatten().tolist()
        result["predicted_value_step_1"] = preds_list[0] if preds_list else None
        result["predicted_value_step_n"] = preds_list[-1] if preds_list else None

    return result

# --- Zentraler Startpunkt der Anwendung ---
if __name__ == "__main__":
    logging.info("Starte die Random Forest ML-Anwendung (als Hauptskript)...")

    try:
        # Initialisiere die Flask-App mit den ML-Pipeline-Funktionen dieses Skripts
        Flask_App.initialize_flask_app(
            setup_and_train_func=setup_and_train_rf,
            run_inference_step_func=run_inference_step_rf
        )
        # Starte den Flask-Server
        Flask_App.run_flask_server(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e:
        logging.error(f"Fehler beim Starten der Anwendung: {e}", exc_info=True)
        sys.exit(1)

