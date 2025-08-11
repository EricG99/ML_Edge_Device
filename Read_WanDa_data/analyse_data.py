import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel

# 1. Pfad zum Datensatz
script_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(script_dir, 'aufzeichnungen', 'mqtt_data_rate_limited.csv')

# 2. Einlesen
df = pd.read_csv(
    data_file,
    parse_dates=['datetime'],
    dayfirst=False
)

# 3. Elapsed time in Sekunden
df = df.sort_values('datetime').reset_index(drop=True)
df['elapsed_sec'] = (df['datetime'] - df['datetime'].iloc[0]).dt.total_seconds()

# 4. Zyklische Zeit-Features
minutes = df['datetime'].dt.minute + df['datetime'].dt.second / 60.0
df['sin_min'] = np.sin(2 * np.pi * minutes / 60)
df['cos_min'] = np.cos(2 * np.pi * minutes / 60)

# 5. Ziel- und Feature-Matrix
target = 'Group4-2_S6_MassFlowRate'
all_features = [c for c in df.columns if c not in ['datetime', 'recording_timestamp', target]]

X = df[all_features]
y = df[target]

# 6. Zeitreihen-Plot in Sekunden
plt.figure(figsize=(6, 3))
plt.plot(df['elapsed_sec'], y, lw=1)
plt.xlabel('Verstrichene Zeit (s)')
plt.ylabel('Massenstrom [Einheit]')
plt.title('Group4-2_S6_MassFlowRate über verstrichene Zeit')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'massflowrate_time_series_sec.pdf'))
plt.show()

# 7. Korrelationsmatrix nur numerischer Features
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
corr = df[num_cols + [target]].corr()
plt.figure(figsize=(6, 6))
im = plt.imshow(corr, aspect='auto', interpolation='nearest')
plt.colorbar(im, fraction=0.046, pad=0.04)
plt.xticks(range(len(corr)), corr.columns, rotation=90)
plt.yticks(range(len(corr)), corr.columns)
plt.title('Korrelationsmatrix numerischer Features und Zielvariable')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'feature_correlation_matrix.pdf'))
plt.show()

# 8. Scatter-Matrix (optional)
scatter_matrix(df[num_cols + [target]],
               figsize=(8, 8),
               diagonal='kde',
               alpha=0.5,
               marker='o')
plt.suptitle('Scatter-Matrix numerischer Messgrößen')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'feature_scatter_matrix.pdf'))
plt.show()

# 9. ML-Modell: RandomForest mit CV und Feature-Importance
X_train, X_test, y_train, y_test = train_test_split(
    X[num_cols], y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
print(f'Mean R² (5-fold CV): {scores.mean():.3f} ± {scores.std():.3f}')

rf.fit(X_train, y_train)
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Top-10 Feature-Importances
plt.figure(figsize=(6, 3))
top_n = min(10, len(num_cols))
plt.barh(range(top_n),
         importances[indices][:top_n][::-1],
         align='center')
plt.yticks(range(top_n),
           [num_cols[i] for i in indices][:top_n][::-1])
plt.xlabel('Importance')
plt.title('Top 10 Feature-Importances (RandomForest)')
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'feature_importances.pdf'))
plt.show()

# 10. (Optional) Automatische Selektion basierend auf Importance
from sklearn.feature_selection import SelectFromModel
selector = SelectFromModel(rf, prefit=True, threshold='median')
X_reduced = selector.transform(X[num_cols])
print(f'Urspr. Feature-Anzahl: {len(num_cols)}, nach Selektion: {X_reduced.shape[1]}')
