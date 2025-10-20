import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Load CSV file
df = pd.read_csv("vectors.csv")  # Replace with your actual filename

# Extract vectors
time, tsh, ft4 = df["Time"].values, df["TSH"].values, df["fT4"].values

# Generate equidistant time points up to 35 (or max time in data)
num_points = 15
time_new = np.linspace(0, max(15, time.max()), num=num_points)

# Interpolation functions
tsh_new = interp1d(time, tsh, kind="linear", fill_value="extrapolate")(time_new)
ft4_new = interp1d(time, ft4, kind="linear", fill_value="extrapolate")(time_new)

# Create new DataFrame
df_new = pd.DataFrame({"Itime": time_new, "Itsh": tsh_new, "Ift4": ft4_new})
df_new.to_csv("interpolated_data.csv", index=False)

# Plotting
plt.figure(figsize=(10, 6))  # Same size as before

# TSH subplot
plt.subplot(2, 1, 1)
plt.plot(time_new, tsh_new, color='lightcoral', linestyle='--', marker='o',
         markersize=5, alpha=0.6, linewidth=1.2, label='Interpolated TSH')
plt.plot(time, tsh, 'o', color='blue', markeredgecolor='black', markersize=8,
         label='Measured TSH')
plt.ylabel("TSH")
plt.legend()
plt.grid(True)

# fT4 subplot
plt.subplot(2, 1, 2)
plt.plot(time_new, ft4_new, color='plum', linestyle='--', marker='o',
         markersize=5, alpha=0.6, linewidth=1.2, label='Interpolated fT4')
plt.plot(time, ft4, 'o', color='red', markeredgecolor='black', markersize=8,
         label='Measured fT4')
plt.xlabel("Time")
plt.ylabel("fT4")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()





