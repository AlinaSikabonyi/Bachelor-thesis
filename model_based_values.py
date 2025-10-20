import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# Load parameters
params = np.loadtxt("optimized_params.csv", delimiter=",")
S, phi, A, alpha = params

# Load and sort data
df = pd.read_csv("/Users/alinasikabonyi/PycharmProject/bachelor/vectors.csv")
df = df.groupby("Time", as_index=False).mean()
df = df.sort_values(by="Time")

time = df["Time"].values
tsh_data = df["TSH"].values
ft4_data = df["fT4"].values

# Define the ODE system
def ODEsystem(t, z, S, phi, A, alpha):
    x, y = z
    dx_dt = S / np.exp(phi * y) - x
    dy_dt = A - (A / np.exp(alpha * x)) - y
    return [dx_dt, dy_dt]

# Initial condition
x0 = [tsh_data[0], ft4_data[0]]

# Create equidistant time points from 0 to max(15, last time point in original data)
final_time = max(15, int(time[-1]))
equidistant_time = np.arange(0, final_time + 1, 1)  # step of 1

# Solve ODE over the new time span
sol = solve_ivp(ODEsystem, (0, final_time), x0, args=(S, phi, A, alpha), t_eval=equidistant_time)

# Get solution values
t_values = sol.t
tsh_pred = sol.y[0]
ft4_pred = sol.y[1]

# Create interpolated DataFrame
df_interp = pd.DataFrame({
    "Itime": t_values,
    "Itsh": tsh_pred,
    "Ift4": ft4_pred
})

# Replace with observed data where available
for t, tsh, ft4 in zip(time, tsh_data, ft4_data):
    df_interp.loc[df_interp["Itime"] == t, "Itsh"] = tsh
    df_interp.loc[df_interp["Itime"] == t, "Ift4"] = ft4

# Save to a new CSV
df_interp.to_csv("new_vectors.csv", index=False)

# --- Plotting ---
measured_times = set(time)
is_interpolated = ~df_interp["Itime"].isin(measured_times)

plt.figure(figsize=(10, 6))

# TSH plot
plt.subplot(2, 1, 1)
plt.plot(df_interp["Itime"], df_interp["Itsh"], color='blue', alpha=0.4, label='TSH Prediction')
plt.scatter(time, tsh_data, color='blue', marker='x', label='Measured TSH')
plt.scatter(df_interp["Itime"][is_interpolated], df_interp["Itsh"][is_interpolated],
            color='blue', marker='o', s=40, label='Model based TSH')
plt.xlabel('Time')
plt.ylabel('TSH')
plt.legend()
plt.grid(True)

# fT4 plot
plt.subplot(2, 1, 2)
plt.plot(df_interp["Itime"], df_interp["Ift4"], color='red', alpha=0.4, label='fT4 Prediction')
plt.scatter(time, ft4_data, color='red', marker='x', label='Measured fT4')
plt.scatter(df_interp["Itime"][is_interpolated], df_interp["Ift4"][is_interpolated],
            color='red', marker='o', s=40, label='Model based fT4')
plt.xlabel('Time')
plt.ylabel('fT4')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

