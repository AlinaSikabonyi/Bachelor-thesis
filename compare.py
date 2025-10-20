import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

# Load parameters
params = np.loadtxt("/Users/alinasikabonyi/PycharmProject/bachelor/optimized_params.csv", delimiter=",")
S, phi, A, alpha = params

# Load and process original data
df = pd.read_csv("/Users/alinasikabonyi/PycharmProject/bachelor/vectors.csv")
df = df.groupby("Time", as_index=False).mean()
df = df.sort_values(by="Time")

time = df["Time"].values
tsh_data = df["TSH"].values
ft4_data = df["fT4"].values

# Define function and solve ODE
def ODEsystem(t, z, S, phi, A, alpha):
    x, y = z
    dx_dt = S / np.exp(phi * y) - x
    dy_dt = A - (A / np.exp(alpha * x)) - y
    return [dx_dt, dy_dt]

x0 = [tsh_data[0], ft4_data[0]]
t_span = (0, 50)
sol = solve_ivp(ODEsystem, t_span, x0, args=(S, phi, A, alpha))
last_time_point = sol.t[-1] # Last time point
last_x_value = sol.y[0, -1]  # TSH value at last time point
last_y_value = sol.y[1, -1]  # fT4 value at last time point

# Numerical set point
x_n = last_x_value
y_n = last_y_value

# Maximum curvature (HP)
x_mc1 = 1 / (np.sqrt(2)) * phi
y_mc1 = (np.log(np.sqrt(2) * S * phi))/phi

# Maximum curvature (T)
x_mc2 = np.log(np.sqrt(2) * A * alpha) / alpha
y_mc2 = A - (1 / (np.sqrt(2) * alpha))

# Gain factor analysis
x_gf = 1 / alpha
y_gf = A * (1 - np.exp(-1))

# Moving average setpoints
df_mov = pd.read_csv("/Users/alinasikabonyi/PycharmProject/bachelor/mov_av.csv")
df_wmov = pd.read_csv("/Users/alinasikabonyi/PycharmProject/bachelor/w_mov_av.csv")
df_exp = pd.read_csv("/Users/alinasikabonyi/PycharmProject/bachelor/exp_sm.csv")

# Extract values
mov_tsh = df_mov['Last_Itsh'].iloc[0]
mov_ft4 = df_mov['Last_Ift4'].iloc[0]

wmov_tsh = df_wmov['Last_Itsh'].iloc[0]
wmov_ft4 = df_wmov['Last_Ift4'].iloc[0]

exp_tsh = df_exp['Last_Itsh'].iloc[0]
exp_ft4 = df_exp['Last_Ift4'].iloc[0]

# Interpolated values (I prefix)
df_imov = pd.read_csv("/Users/alinasikabonyi/PycharmProject/bachelor/imov_av.csv")
df_iwmov = pd.read_csv("/Users/alinasikabonyi/PycharmProject/bachelor/iw_mov_av.csv")
df_iexp = pd.read_csv("/Users/alinasikabonyi/PycharmProject/bachelor/iexp_sm.csv")

# Extract I-prefixed values
imov_tsh = df_imov['Last_Itsh'].iloc[0]
imov_ft4 = df_imov['Last_Ift4'].iloc[0]

iwmov_tsh = df_iwmov['Last_Itsh'].iloc[0]
iwmov_ft4 = df_iwmov['Last_Ift4'].iloc[0]

iexp_tsh = df_iexp['Last_Itsh'].iloc[0]
iexp_ft4 = df_iexp['Last_Ift4'].iloc[0]

# Print values
print("Moving average setpoint:", mov_tsh)
print("Moving average setpoint fT4:", mov_ft4)
print("Weighted Moving average setpoint:", wmov_tsh)
print("Weighted Moving average setpoint fT4:", wmov_ft4)
print("Exponential smoothing:", exp_tsh)
print("Exponential smoothing fT4:", exp_ft4)

print("I-Moving average setpoint:", imov_tsh)
print("I-Moving average setpoint fT4:", imov_ft4)
print("I-Weighted Moving average setpoint:", iwmov_tsh)
print("I-Weighted Moving average setpoint fT4:", iwmov_ft4)
print("I-Exponential smoothing:", iexp_tsh)
print("I-Exponential smoothing fT4:", iexp_ft4)

# Scatter plot
plt.scatter([x_n], [y_n], label='Numerical Set Point')
plt.scatter([x_mc1], [y_mc1], label='Maximum Curvature (HP)')
plt.scatter([x_mc2], [y_mc2], label='Maximum Curvature (T)')
plt.scatter([x_gf], [y_gf], label='Gain Factor')

# Original set points
plt.scatter([mov_tsh], [mov_ft4], label='Moving Average')
plt.scatter([wmov_tsh], [wmov_ft4], label='Weighted Moving Average')
plt.scatter([exp_tsh], [exp_ft4], label='Exponential Smoothing')

# I-prefixed set points
plt.scatter([imov_tsh], [imov_ft4], label='I-Moving Average')
plt.scatter([iwmov_tsh], [iwmov_ft4], label='I-Weighted Moving Average')
plt.scatter([iexp_tsh], [iexp_ft4], label='I-Exponential Smoothing')

# Add reference range rectangle (TSH: 2.5–4, FT4: 7–18)
plt.gca().add_patch(
    plt.Rectangle((2.5, 7), 1.5, 11,
                  linewidth=0,
                  edgecolor=None,
                  facecolor='lightgrey',
                  alpha=0.3,
                  label='Reference Range')
)

# Move legend outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Set Points", fontsize=10)

plt.xlabel('TSH')
plt.ylabel('FT4')
plt.grid(True)
plt.tight_layout()

# Show plot
plt.show()
