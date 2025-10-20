import pandas as pd
import matplotlib.pyplot as plt

# Excel-Datei einlesen
df = pd.read_excel(
    '/Users/alinasikabonyi/Desktop/BA Forecasting Methods/Thyroid.xlsx',
    sheet_name='thyroid_data',
    engine='openpyxl'
)

# Filtern der Daten für eine bestimmte ID
id_value = 4 # gewünschte ID
df_id = df[df['ID'] == id_value][['Time', 'TSH', 'fT4']]

# Werte für die ID ausgeben
print(f"Werte für ID {id_value}:")
print(df_id)

# Vektoren extrahieren
time_values = df_id['Time'].values
tsh_values = df_id['TSH'].values
ft4_values = df_id['fT4'].values

# Vektoren-DataFrame erstellen
df_vectors = pd.DataFrame({
    'Time': time_values,
    'TSH': tsh_values,
    'fT4': ft4_values
})

# CSV-Datei speichern
df_vectors.to_csv('/Users/alinasikabonyi/PycharmProject/bachelor/vectors.csv', index=False)

# Plotten der Zeitreihe mit Punkten (Marker)
plt.figure(figsize=(10, 6))
plt.plot(df_id['Time'], df_id['TSH'], label='TSH', color='blue', marker='o', linestyle='None')  # Dots for TSH
plt.plot(df_id['Time'], df_id['fT4'], label='fT4', color='red', marker='o', linestyle='None')  # Dots for fT4

# Diagramm-Layout

plt.xlabel('Time')
plt.ylabel('Hormone Concentrations')
plt.legend()
plt.grid(True)
plt.show()



