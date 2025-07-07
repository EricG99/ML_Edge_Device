# LSTM_Loaded_Web_App.py
import sys
import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf # Behalte TensorFlow-Import für LSTM-Modelle
import joblib # Zum Laden des Scalers

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


if project_root not in sys.path:
    sys.path.append(project_root)


from ML_WebApp import Flask_App
from ML_Helpfunctions import Pipeline_Utils as PipelineUtils
from ML_Helpfunctions import Load_Prepare_Data as LoadPrepareData
from ML_Algorithms import RF_Run_Pipeline as RFRunPipeline
from ML_Helpfunctions import RF_Utils as RFUtils

from config import CONFIG_PATH, param_LSTM_EDGE, CONFIG_LSTM_ALL

# # Kombinierte Konfiguration
# CONFIG_LSTM_ALL = {**CONFIG_PATH, **param_LSTM_EDGE}
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LOAD_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'your_lstm_model.h5')
LOAD_SCALER_PATH = os.path.join(BASE_DIR, 'models', 'your_scaler.pkl') 

def load_model_LSTM(model_path, model_name, model_type="quantized"):
    """
    Lädt ein LSTM-Modell, entweder ein normales Keras-Modell oder ein quantisiertes TFLite-Modell.

    Args:
        model_path (str): Der Pfad zur Modelldatei.
        model_type (str): Der Typ des zu ladenden Modells ("normal" für Keras, "quantized" für TFLite).

    Returns:
        tf.keras.Model or tf.lite.Interpreter: Das geladene Modell oder der TFLite Interpreter.

    Raises:
        ValueError: Wenn ein ungültiger model_type angegeben wird.
        RuntimeError: Wenn das Laden des Modells fehlschlägt.
    """
    model_path = os.path.join(model_path, model_name)
    logging.info(f"Versuche, {model_type} Modell von {model_path} zu laden...")

    if model_type == "normal":
        try:
            model = tf.keras.models.load_model(model_path)
            logging.info(f"Normales Keras LSTM-Modell erfolgreich geladen.")
            return model
        except Exception as e:
            logging.error(f"Fehler beim Laden des normalen Keras LSTM-Modells von {model_path}: {e}", exc_info=True)
            raise RuntimeError(f"Normales Keras LSTM-Modell konnte nicht geladen werden: {e}")
    elif model_type == "quantized":
        try:
            # Für TFLite Modelle wird ein Interpreter benötigt
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            logging.info(f"Quantisiertes TFLite LSTM-Modell erfolgreich geladen.")
            return interpreter
        except Exception as e:
            logging.error(f"Fehler beim Laden des quantisierten TFLite LSTM-Modells von {model_path}: {e}", exc_info=True)
            raise RuntimeError(f"Quantisiertes TFLite LSTM-Modell konnte nicht geladen werden: {e}")
    else:
        raise ValueError(f"Ungültiger Modelltyp: {model_type}. Muss 'normal' oder 'quantized' sein.")
    
def run_inference_lstm(model, X_test: np.ndarray) -> np.ndarray:
    """
    Führt die Inferenz für ein LSTM-Modell durch (unterstützt sowohl Keras als auch TFLite).
    
    Args:
        model: Keras Modell oder TFLite Interpreter
        X_test (np.ndarray): Eingabedaten, Form (samples, timesteps, features)
    
    Returns:
        np.ndarray: Vorhersagen
    """
    print("🔍 Starte LSTM-Inferenz...")
    
    if len(X_test.shape) != 3:
        raise ValueError(f"❌ Falsche Eingabeform. Erwartet (samples, timesteps, features), erhalten: {X_test.shape}")

    try:
        # Unterscheidung zwischen Keras und TFLite Modell
        if isinstance(model, tf.lite.Interpreter):
            # TFLite Inferenz
            interpreter = model
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Eingabedaten vorbereiten
            interpreter.set_tensor(input_details[0]['index'], X_test.astype(np.float32))
            
            # Inferenz durchführen
            interpreter.invoke()
            
            # Ergebnisse holen
            preds = interpreter.get_tensor(output_details[0]['index'])
        else:
            # Standard Keras Inferenz
            preds = model.predict(X_test, verbose=0)
            
        print(f"✅ Inferenz abgeschlossen - Ausgabeform: {preds.shape}")
        return np.array(preds)
        
    except Exception as e:
        print(f"❌ Inferenzfehler: {e}")
        import traceback
        print(traceback.format_exc())
        raise


def setup_and_load_lstm_model(CONFIG_LSTM_ALL=None):
    """Lädt ein vortrainiertes LSTM-Modell und bereitet die Daten vor."""

    # 1. Setup
    param_lstm_config, paths = PipelineUtils.setup_experiment(CONFIG_LSTM_ALL)

    # 2. Daten vorbereiten mit erweiterten 3D-Features
    (
        X_train_3D, y_train_3D,
        X_test_3D, y_test_3D,
        scaler_3D, y_scaler,
        train_df, test_df,
        train_features_dict, full_feature_list
    ) = LoadPrepareData._prepare_base_data_3D(param_lstm_config)

    print(f"[DEBUG] Shape y_train_3D: {y_train_3D.shape}, Shape y_test_3D: {y_test_3D.shape}")
    print(f"[DEBUG] Horizon aus config: {param_lstm_config.get('horizon')}")

    # 3. Modell laden
    print("\n" + "="*50)
    print("DEBUGGING: Überprüfung der Werte vor dem Laden des Modells")
    print(f"Typ von 'paths': {type(paths)}")
    print(f"Inhalt von 'paths': {paths}")
    
    print("-" * 20)
    
    # Überprüfen wir die Werte, die wir verwenden wollen
    model_path_value = paths.get("input_data_edge_device")
    model_name_value = param_lstm_config.get("model_name_edge_device")
    
    print(f"Wert für 'model_path': {model_path_value} (Typ: {type(model_path_value)})")
    print(f"Wert für 'model_name': {model_name_value} (Typ: {type(model_name_value)})")
    print("="*50 + "\n")

    # Der eigentliche Aufruf
    model = load_model_LSTM(
        model_path=model_path_value,
        model_name=model_name_value,
    )

    # 3. Gib ein Dictionary mit allen Artefakten zurück, die für die Inferenz benötigt werden
    return {
        "model": model, 
        "param_config": param_lstm_config, # Schlüssel zu "param_config" geändert
        "paths": paths,
        "X_test_3D": X_test_3D, "y_test_3D": y_test_3D, "scaler_3D": scaler_3D, 
        "test_df": test_df, "full_feature_list": full_feature_list, "config": param_lstm_config, 
        "total_steps": len(X_test_3D) # Sicherstellen, dass dies korrekt ist für LSTM-Schritte
    }


def run_inference_step_lstm_loaded(artifacts, step_index):
    """
    Implementierung der Schnittstelle für die Inferenz für einen einzelnen Schritt.
    Angepasst für einen einzelnen LSTM-Inferenzschritt mit geladenem Modell.

    Args:
        artifacts (dict): Das Dictionary von Artefakten, das von setup_and_train_lstm_loaded zurückgegeben wurde.
        step_index (int): Der aktuelle Schrittindex für die Inferenz.

    Returns:
        dict: Ein Dictionary mit den Ergebnissen des Inferenzschritts,
              einschließlich Datum, tatsächlichem Wert und Prognose.
    """
    # 1. Benötigte Artefakte aus dem Dictionary extrahieren
    model = artifacts["model"]
    X_test_3D = artifacts["X_test_3D"]
    y_test_3D = artifacts["y_test_3D"]
    scaler = artifacts["scaler_3D"] # Dies ist der geladene Scaler
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
    try:
        preds_scaled = run_inference_lstm(model=model, X_test=X_step)
        
        # Für TFLite müssen wir möglicherweise die Ausgabeform anpassen
        if isinstance(model, tf.lite.Interpreter):
            preds_scaled = preds_scaled[0]  # Ersten Batch nehmen
            
        preds_scaled = preds_scaled[0]  # Batch-Dimension entfernen
    except Exception as e:
        logging.error(f"Inferenzfehler: {e}", exc_info=True)
        raise
    
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
    logging.info("Starte die LSTM ML-Anwendung...")
    
    try:
        # Initialisiere die Flask-App mit den ML-Pipeline-Funktionen
        Flask_App.initialize_flask_app(
            setup_and_train_func=lambda: setup_and_load_lstm_model(CONFIG_LSTM_ALL),
            run_inference_step_func=run_inference_step_lstm_loaded
        )
        
        Flask_App.run_flask_server(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
    except Exception as e:
        logging.error(f"Fehler beim Starten der Anwendung: {e}", exc_info=True)
        sys.exit(1)

