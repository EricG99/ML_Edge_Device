# xgboost_utils.py
import time
import xgboost as xgb
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import sys
import os

# --- Project Path Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ML_Helpfunctions import Pipeline_Utils

def train_xgboost_model(config: dict, X_train: np.ndarray, y_train: np.ndarray, features: list):
    """
    Trainiert ein XGBoost-Modell, ggf. mit MultiOutputRegressor und Early Stopping.

    Args:
        config (dict): Konfigurationsparameter (n_estimators, learning_rate, etc.).
        X_train (np.ndarray): Trainingsdaten (Input, 2D).
        y_train (np.ndarray): Trainingsdaten (Zielwerte).
        features (list): Liste der Feature-Namen.

    Returns:
        tuple: (model, train_time)
            - model: Das trainierte XGBoost-Modell.
            - train_time: Die Trainingszeit in Sekunden.
    """
    print("🚀 Starte Training für XGBoost-Modell...")
    start_time = time.time()

    # 1. Erstelle einen Validierungssplit für Early Stopping
    X_fit, y_fit, X_val, y_val = Pipeline_Utils.create_timeseries_validation_split(
        X_train, y_train, config
    )
    
    # 2. Modell initialisieren mit Parametern aus der Konfiguration
    xgb_base = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=config.get("n_estimators", 1000),
        learning_rate=config.get("learning_rate", 0.05),
        max_depth=config.get("max_depth", 5),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        gamma=config.get("gamma", 0),
        random_state=config.get("random_state", 42),
        n_jobs=config.get("n_jobs", -1),
        early_stopping_rounds=config.get("early_stopping_rounds", 50) # Early stopping
    )

    # 3. Multi-Output-Strategie anwenden, falls Horizont > 1
    model_to_train = xgb_base
    fit_params = {}
    
    current_horizon = config.get("horizon", 1)
    if current_horizon > 1:
        print(f"XGBoost: MultiOutputRegressor wird für horizon={current_horizon} verwendet.")
        # MultiOutputRegressor unterstützt kein direktes `eval_set`, daher wird es hier nicht verwendet.
        # Early stopping ist in diesem Modus nicht direkt mit dem Wrapper möglich.
        # Das Basismodell wird ohne Early Stopping trainiert.
        xgb_base.early_stopping_rounds = None 
        model_to_train = MultiOutputRegressor(xgb_base)
        
        # Für MultiOutputRegressor sind keine speziellen fit_params nötig.
        print("Warnung: Early stopping ist im Multi-Output-Modus mit dem Scikit-learn Wrapper nicht verfügbar.")
    else:
        # Für Single-Output (Horizon = 1)
        if y_train.ndim == 2 and y_train.shape[1] == 1:
            y_fit = y_fit.ravel()
            if y_val is not None:
                y_val = y_val.ravel()
        
        # `eval_set` für Early Stopping vorbereiten
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False # Unterdrückt die Ausgabe für jede Runde
            print(f"XGBoost: Early Stopping wird mit Validierungsdaten der Form X:{X_val.shape} verwendet.")
        else:
             xgb_base.early_stopping_rounds = None # Deaktiviere Early Stopping, wenn keine Validierungsdaten da sind
             print("XGBoost: Kein Validierungssplit, Training ohne Early Stopping.")


    # 4. Modell trainieren
    print(f"Starte model.fit() für XGBoost auf Daten mit Shape X: {X_fit.shape}, Y: {y_fit.shape}...")
    model_to_train.fit(X_fit, y_fit, **fit_params)
    print("✅ XGBoost-Modell Training abgeschlossen.")

    training_duration_seconds = time.time() - start_time
    print(f"⏱️ Trainingszeit für XGBoost: {training_duration_seconds:.2f} Sekunden.")
    print(f"Model type after training: {type(model_to_train)}")

    return model_to_train, training_duration_seconds