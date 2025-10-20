import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/alinasikabonyi/PycharmProject/bachelor/vectors.csv")
df = df.groupby("Time", as_index=False).mean()
df = df.sort_values(by="Time")


time = df["Time"].values
tsh_data = df["TSH"].values
ft4_data = df["fT4"].values


x0 = [tsh_data[0], ft4_data[0]]


def ODEsystem(t, z, S, phi, A, alpha):
    x, y = z
    dx_dt = S / np.exp(phi * y) - x
    dy_dt = A - (A / np.exp(alpha * x)) - y
    return [dx_dt, dy_dt]


def simulate(params):
    S, phi, A, alpha = params
    sol = solve_ivp(
        ODEsystem,
        [time[0], time[-1]],
        x0,
        args=(S, phi, A, alpha),
        t_eval=time,
        method='RK45'
    )
    return sol.y  # [TSH, fT4]


def loss(params):
    sim = simulate(params)
    sim_tsh, sim_ft4 = sim
    error = np.mean((sim_tsh - tsh_data)**2 + (sim_ft4 - ft4_data)**2)
    return error


bounds = [
    (1000, 1500),   # S
    (0.2, 0.5),     # phi
    (20, 100),      # A
    (0.1, 0.4)      # alpha
]

initial_guess = [1200, 0.3, 50, 0.2]

result = minimize(loss, initial_guess, bounds=bounds, method='L-BFGS-B')

print("Optimale Parameter:")
print("S =", result.x[0])
print("phi =", result.x[1])
print("A =", result.x[2])
print("alpha =", result.x[3])


sim_result = simulate(result.x)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(time, sim_result[0], label="TSH (simuliert)")
plt.scatter(time, tsh_data, color='red', label="TSH (Messung)")
plt.xlabel("Zeit")
plt.ylabel("TSH")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time, sim_result[1], label="fT4 (simuliert)")
plt.scatter(time, ft4_data, color='red', label="fT4 (Messung)")
plt.xlabel("Zeit")
plt.ylabel("fT4")
plt.legend()

plt.tight_layout()
plt.show()

np.savetxt("optimized_params.csv", result.x, delimiter=",")

#pd.DataFrame([result.x], columns=["S", "phi", "A", "alpha"]).to_csv("optimized_parameters.csv", index=False)
