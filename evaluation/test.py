import pandas as pd
import json
import os
import numpy as np
import plotly.graph_objects as go

# ==============================================================================
# Teil 1: Funktion zur Datenaufbereitung
# ==============================================================================

def load_and_process_data(root_path, summary_filename, algorithm, profile, linux_base_path):
    """
    Liest alle Rohdaten, korrigiert die Dateipfade von Linux zu Windows
    und gibt einen verarbeiteten DataFrame zurück.
    """
    summary_path = os.path.join(root_path, 'Error_Metrics', summary_filename)
    try:
        df_summary = pd.read_csv(summary_path, engine='python', encoding='utf-8')
    except FileNotFoundError:
        print(f"FEHLER: Die zentrale Übersichtsdatei wurde nicht gefunden: {summary_path}")
        return None 

    filtered_summary = df_summary[(df_summary['algorithm'] == algorithm) & (df_summary['profile'] == profile)]
    if filtered_summary.empty: 
        print(f"Keine Experimente für Algorithmus='{algorithm}' und Profil='{profile}' gefunden.")
        return None
    
    results_list = []
    print(f"Datenaufbereitung: {len(filtered_summary)} passende Experimente gefunden. Beginne Verarbeitung...")
    for index, summary_row in filtered_summary.iterrows():
        try:
            run_id = str(summary_row['run_id']).strip()
        except AttributeError:
            continue

        run_data = summary_row.to_dict()
        run_data['run_id'] = run_id
        
        try:
            algo_folder_name = algorithm.replace('_', ' ').title().replace(' ', '_')
            run_folder_path = os.path.join(root_path, algo_folder_name, run_id)
            if not os.path.isdir(run_folder_path):
                print(f"  - Warnung: Das Verzeichnis für run_id '{run_id}' wurde nicht gefunden: {run_folder_path}")
                continue

            run_metrics_path = os.path.join(run_folder_path, 'Error_Metrics', 'ErrorMetrics_all_runs.csv')
            df_run_details = pd.read_csv(run_metrics_path, engine='python')
            if df_run_details.empty: continue
            
            run_details_row = df_run_details.iloc[0]
            
            # --- NEU: Pfade von Linux zu Windows anpassen ---
            predictions_path_linux = run_details_row['predictions_file_path']
            json_path_linux = run_details_row['json_path']
            
            predictions_path = predictions_path_linux.replace(linux_base_path, root_path)
            json_path = json_path_linux.replace(linux_base_path, root_path)
            # -----------------------------------------------

            with open(json_path, 'r', encoding='utf-8') as f:
                error_data = json.load(f)
            if 'metrics' in error_data: run_data.update(error_data['metrics'])
            
            pred_df = pd.read_csv(predictions_path, engine='python')
            pred_df_for_avg = pred_df.iloc[1:].copy()
            avg_cols = ['inference_time_s', 'total_time_s', 'cpu_percent', 'ram_percent', 'ram_mb']
            for col in avg_cols:
                if col in pred_df_for_avg.columns:
                    numeric_series = pd.to_numeric(pred_df_for_avg[col], errors='coerce')
                    run_data[f'avg_{col}'] = numeric_series.mean()
            
            for col in pred_df.columns:
                run_data[f'{col}_list'] = pred_df[col].tolist()
                
        except FileNotFoundError as e:
            print(f"  - Warnung: Datei für run_id '{run_id}' nicht gefunden. Wahrscheinlich war der Pfad-Ersatz nicht erfolgreich.")
            print(f"    Original Linux-Pfad: {run_details_row.get('json_path', 'N/A')}")
            print(f"    Umgewandelter Windows-Pfad: {e.filename}")
            continue
        except Exception as e:
            print(f"  - Warnung: Ein unerwarteter Fehler ist bei run_id '{run_id}' aufgetreten: {type(e).__name__} - {e}")
            continue
            
        results_list.append(run_data)
        
    if not results_list:
        print("\nDatenaufbereitung abgeschlossen, aber keine Experimente konnten erfolgreich verarbeitet werden.")
        return pd.DataFrame()

    print(f"\nDatenaufbereitung erfolgreich abgeschlossen. {len(results_list)} Experiment(e) verarbeitet.")
    return pd.DataFrame(results_list)

# ==============================================================================
# Teil 2: Funktion zur Visualisierung (unverändert)
# ==============================================================================
def visualize_data(df, metrics_to_plot):
    if df is None or df.empty:
        print("\nVisualisierung übersprungen, da keine Daten verarbeitet wurden.")
        return
    
    print(f"\nVisualisierung: Erstelle {len(metrics_to_plot)} interaktive Plots...")
    for metric, title in metrics_to_plot.items():
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            print(f"  - Erstelle Plot für: {metric}...")
            try:
                # ... (Rest der Funktion bleibt unverändert) ...
                df['lags'] = pd.to_numeric(df['lags'])
                df['horizon'] = pd.to_numeric(df['horizon'])
                pivot_df = df.pivot_table(index='horizon', columns='lags', values=metric, aggfunc='mean')
                fig = go.Figure(data=[go.Surface(z=pivot_df.values, x=pivot_df.columns, y=pivot_df.index, colorscale='Viridis', colorbar=dict(title=metric.replace('_', ' ').title()))])
                fig.update_layout(title=title, scene=dict(xaxis_title='Lags', yaxis_title='Horizon', zaxis_title=metric), width=800, height=700, margin=dict(l=65, r=50, b=65, t=90))
                output_filename = f"interactive_grid_plot_{metric}.html"
                fig.write_html(output_filename)
                print(f"    -> Interaktiver Plot als '{output_filename}' gespeichert.")
                fig.show()
            except Exception as e:
                print(f"    -> Fehler beim Erstellen des Plots für '{metric}': {e}")
        else:
            print(f"  - Warnung: Spalte '{metric}' kann nicht geplottet werden. Übersprungen.")

# ==============================================================================
# Haupt-Workflow
# ==============================================================================

# --- EINSTELLUNGEN ---
OUTPUT_ROOT_PATH = r'C:\Users\ericg\Documents\Mechatronik M Sc\6. Semster\MA\Dev_Ma\ML_Edge_Device'
SUMMARY_FILENAME = 'Experiment_Summary.csv'
TARGET_ALGORITHM = 'lstm' 
TARGET_PROFILE = 'edge'

# NEU: Der Linux-Basispfad, der in den Pfaden aus der CSV-Datei ersetzt werden soll.
LINUX_BASE_PATH_TO_REPLACE = '/home/pi/ML_Edge_Device/Output'

METRICS_FOR_PLOTTING = {
    'avg_inference_time_s': 'Durchschnittliche Inferenzzeit (s)',
    'avg_total_time_s': 'Durchschnittliche Gesamtzeit (s)',
    'avg_cpu_percent': 'Durchschnittliche CPU-Auslastung (%)',
    'avg_ram_percent': 'Durchschnittliche RAM-Auslastung (%)',
    'model_size_mb': 'Modellgröße (MB)'
}

# --- AUSFÜHRUNG ---
df_processed = load_and_process_data(
    root_path=OUTPUT_ROOT_PATH,
    summary_filename=SUMMARY_FILENAME,
    algorithm=TARGET_ALGORITHM,
    profile=TARGET_PROFILE,
    linux_base_path=LINUX_BASE_PATH_TO_REPLACE
)

visualize_data(df_processed, METRICS_FOR_PLOTTING)