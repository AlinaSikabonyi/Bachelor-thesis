import pandas as pd
import matplotlib.pyplot as plt

# Excel-Datei einlesen
df = pd.read_excel(
    '/Users/alinasikabonyi/Desktop/BA Forecasting Methods/Thyroid.xlsx',
    sheet_name='thyroid_data',
    engine='openpyxl'
)

# Plot 1: Time vs ID
plt.figure(figsize=(6, 5))
plt.scatter(df['Time'], df['ID'], alpha=0.6, c='blue')
plt.xlabel('Time')
plt.ylabel('ID')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: fT4 vs TSH
plt.figure(figsize=(6, 5))
plt.scatter(df['fT4'], df['TSH'], alpha=0.6, c='green')
plt.xlabel('FT4')
plt.ylabel('TSH')
plt.grid(True)
plt.tight_layout()
plt.show()
