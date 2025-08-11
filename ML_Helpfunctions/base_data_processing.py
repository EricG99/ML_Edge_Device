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

        # KORREKTUR: Bestimme die wahre minimale Anzahl an Datenpunkten, die für alle Features benötigt wird.
        # Dies ist das Maximum aus der Lag-Anzahl und der Fenstergröße für rollierende Features.
        min_for_lags = self.config.get('lags', 1)
        min_for_rolling_features = self.config.get('rolling_window_size', 1)
        self._min_data_points = max(min_for_lags, min_for_rolling_features)

        # === LÖSUNG: Puffergröße drastisch erhöhen für stabile Features ===
        # Ein kleiner Puffer führt zu instabilen statistischen Features (z.B. rolling_mean),
        # die nicht mit den im Training berechneten Features übereinstimmen.
        # Ein größerer Puffer sorgt für repräsentativere Feature-Werte.
        # Der Wert kann über den Key 'max_fe_window' in der Config überschrieben werden.
        default_window_size = 1000
        self._max_buffer_size = self.config.get('max_fe_window', default_window_size) + self.config.get('lags', 1)
        
        logging.info(
            f"RealTimeDataProcessor initialisiert. "
            f"Maximale Puffergröße: {self._max_buffer_size} (Default-Fenster: {default_window_size}), "
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
            # Alle Spaltennamen auf Kleinbuchstaben standardisieren
            new_row.columns = new_row.columns.str.lower()
            
            # Zeitstempel-Spalte robust verarbeiten
            if 'datetime' in new_row.columns:
                new_row['datetime'] = pd.to_datetime(new_row['datetime'])
            elif 'time' in new_row.columns:
                 new_row['datetime'] = pd.to_datetime(new_row['time'], unit='ms')
            else:
                 raise KeyError("Eingehende Daten müssen eine 'datetime' oder 'time' Spalte enthalten.")

            new_row = new_row.set_index('datetime')
        except KeyError as e:
            logging.error(f"Der neue Datenpunkt enthält keinen gültigen Zeitstempel-Schlüssel: {e}")
            return None
        
        self._buffer = pd.concat([self._buffer, new_row]).sort_index()

        # 2. Puffer auf die maximale Größe kürzen (Effizienz)
        if len(self._buffer) > self._max_buffer_size:
            self._buffer = self._buffer.iloc[-self._max_buffer_size:]

        # 3. Prüfen, ob genügend Daten für eine sinnvolle Verarbeitung vorhanden sind
        if len(self._buffer) < self._min_data_points:
            logging.info(f"Datenpuffer wird gefüllt... {len(self._buffer)}/{self._min_data_points}")
            return None

        # 4. Feature Engineering auf dem kleinen, optimierten Puffer ausführen
        try:
            featured_buffer, _ = fe.add_all_features(self._buffer.copy(), self.config)
            return featured_buffer
        except Exception as e:
            logging.error(f"Fehler beim Feature Engineering im RealTimeDataProcessor: {e}", exc_info=True)
            return None