import pandas as pd
import logging
import sys
import os

# --- Systempfad-Setup ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from ML_Helpfunctions import Feature_Engeneering as fe

class RealTimeDataProcessor:
    """
    Eine wiederverwendbare Klasse zur effizienten Verarbeitung von Echtzeit-Datenströmen.
    Sie verwaltet einen internen Datenpuffer mit fester Größe, um eine konstante
    Performance bei der Feature-Berechnung zu gewährleisten.
    """
    def __init__(self, config: dict):
        """
        Initialisiert den Prozessor.
        """
        self.config = config
        self._buffer = pd.DataFrame()

        # --- FINALE KORREKTUR START ---
        # Lese 'lags' aus der Trainingskonfiguration, die zuverlässiger ist.
        # Der fallback auf die Haupt-config bleibt als Sicherheit.
        training_cfg = self.config.get('training_config', {})
        min_for_lags = int(training_cfg.get('lags', self.config.get('lags', 1)))

        rolling_windows = self.config.get('rolling_windows', [])
        if not isinstance(rolling_windows, list):
            rolling_windows = [rolling_windows]
        
        min_for_rolling_features = max(
            self.config.get('rolling_window_size', 1),
            max(rolling_windows) if rolling_windows else 1
        )
        
        # Die tatsächliche Mindestanzahl ist das Maximum aus beiden Anforderungen.
        self._min_data_points = max(min_for_lags, min_for_rolling_features)
        # --- FINALE KORREKTUR ENDE ---

        self._max_buffer_size = self.config.get('max_fe_window', 50) + min_for_lags
        
        logging.info(
            f"RealTimeDataProcessor initialisiert. "
            f"Maximale Puffergröße: {self._max_buffer_size}, "
            f"Minimale Datenpunkte für Start: {self._min_data_points}"
        )

    def update_and_process(self, new_data_point: dict) -> pd.DataFrame | None:
        """
        Fügt einen neuen Datenpunkt hinzu, aktualisiert den Puffer und berechnet die Features.

        Args:
            new_data_point (dict): Ein einzelner, neuer Datenpunkt als Dictionary.

        Returns:
            pd.DataFrame | None: Ein DataFrame mit den neu berechneten Features, wenn genügend
            Daten vorhanden sind. Andernfalls None.
        """
        if not isinstance(new_data_point, dict):
            logging.warning("Ungültiger Datenpunkt empfangen. Erwartet wurde ein Dictionary.")
            return None

        # 1. Neuen Datenpunkt in einen DataFrame umwandeln und an den Puffer anhängen
        try:
            new_row = pd.DataFrame([new_data_point])
            new_row.columns = new_row.columns.str.lower()
            new_row['datetime'] = pd.to_datetime(new_row['datetime'])
            new_row = new_row.set_index('datetime')
        except KeyError:
            logging.error("Der neue Datenpunkt enthält keinen 'datetime'-Schlüssel.")
            return None
        
        self._buffer = pd.concat([self._buffer, new_row]).sort_index()

        # FIX 2: Duplikate & NaNs im Index entfernen, damit der "letzte" Index wirklich fortschreitet
        self._buffer = self._buffer[~self._buffer.index.duplicated(keep='last')]
        self._buffer = self._buffer[~self._buffer.index.isna()]


        # 2. Puffer auf die maximale Größe kürzen (Effizienz)
        if len(self._buffer) > self._max_buffer_size:
            self._buffer = self._buffer.iloc[-self._max_buffer_size:]

        # 3. Prüfen, ob genügend Daten für eine sinnvolle Verarbeitung vorhanden sind
        if len(self._buffer) < self._min_data_points:
            logging.info(f"Datenpuffer wird gefüllt... {len(self._buffer)}/{self._min_data_points}")
            return None

        # 4. Feature Engineering auf dem kleinen, optimierten Puffer ausführen
        try:
            # --- HINZUGEFÜGTE DIAGNOSE-ZEILE ---
            # Diese Log-Ausgabe hilft zu überprüfen, ob die bei der Inferenz verwendete 
            # Konfiguration die korrekten Werte für das Feature Engineering enthält.
            logging.debug(f"Übergebe Konfiguration an FE: lags={self.config.get('lags')}, rolling_windows={self.config.get('rolling_windows')}")
            # --- ENDE DIAGNOSE-ZEILE ---

            featured_buffer, _ = fe.add_all_features(self._buffer.copy(), self.config)
            return featured_buffer
        except Exception as e:
            logging.error(f"Fehler beim Feature Engineering im RealTimeDataProcessor: {e}", exc_info=True)
            return None


        
    def prime_buffer(self, df_like: pd.DataFrame):
            """
            Füllt den internen Puffer initial mit den letzten _min_data_points Zeilen.
            Erwartet Datetime-Index und Spaltennamen in Kleinbuchstaben.
            """
            if df_like is None or df_like.empty:
                return
            df = df_like.copy()
            df.columns = df.columns.str.lower()
            if "datetime" in df.columns:
                df["datetime"] = pd.to_datetime(df["datetime"])
                df = df.set_index("datetime")
            # Puffer mit den letzten N Zeilen füllen, die für das Feature Engineering benötigt werden
            self._buffer = df.sort_index().tail(self._min_data_points)
            logging.info(f"Puffer wurde mit {len(self._buffer)} Zeilen vorgefüllt ('geprimed').")
 