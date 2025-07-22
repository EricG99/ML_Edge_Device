import pandas as pd
import plotly.express as px
import os

# Laden des Datensatzes
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, 'aufzeichnungen', 'mqtt_data_rate_limited.csv')
    df = pd.read_csv(csv_path)

    # Konvertieren der 'datetime'-Spalte
    df['datetime'] = pd.to_datetime(df['datetime'], format='%b %d %Y %H:%M:%S.%f')

    # Spalten, die geplottet werden sollen
    y_columns = ['Group4-2_S6_MassFlowRate', 'Group4-2_S6_FlowVelocity', 'Group4-2_S6_Volume']
    x_column = 'datetime'

    # Umwandeln der Daten vom "wide" ins "long" Format, was für Plotly ideal ist
    df_melted = df.melt(id_vars=[x_column], value_vars=y_columns,
                        var_name='Messgröße', value_name='Wert')

    # Erstellen eines interaktiven Liniendiagramms mit Facetten (Subplots)
    fig = px.line(df_melted,
                  x=x_column,
                  y='Wert',
                  color='Messgröße',       # Jede Messgröße erhält eine eigene Farbe
                  facet_row='Messgröße',  # Erstellt für jede Messgröße einen eigenen Subplot
                  title='Interaktive Datenvisualisierung')

    # Layout anpassen, um redundante Y-Achsen-Titel zu entfernen
    fig.update_yaxes(matches=None, title_text="")
    fig.update_layout(
        xaxis_title='Zeit',
        showlegend=False # Legende ist durch die Subplot-Titel überflüssig
    )

    # Speichern des Plots als interaktive HTML-Datei
    fig.write_html("interactive_data_visualization.html")
    
    # Optional: Plot direkt im Browser öffnen
    # fig.show()

    print("Der interaktive Plot wurde als interactive_data_visualization.html gespeichert.")

except FileNotFoundError:
    print("Die Datei mqtt_data.csv wurde nicht gefunden.")
except KeyError as e:
    print(f"Eine Spalte wurde nicht im CSV gefunden: {e}")
except Exception as e:
    print(f"Ein Fehler ist aufgetreten: {e}")