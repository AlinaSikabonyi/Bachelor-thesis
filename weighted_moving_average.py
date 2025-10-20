import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Weighted moving average function (for visualization only)
def weighted_moving_average(data, window_size=3):
    """Compute weighted moving average using linearly decreasing weights."""
    weighted_avg = []
    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window_data = data[start_index:i + 1]
        weights = np.linspace(1, 0.1, len(window_data))  # Avoid 0 weights
        weights /= np.sum(weights)
        weighted_avg.append(np.sum(window_data * weights))
    return np.array(weighted_avg)

# Forecast using last raw data values (not the smoothed series)
def forecast_next_values(original_data, window_size=5, num_predictions=10):
    """Forecast future values based on the last raw values with rolling window."""
    predicted_values = []
    for i in range(num_predictions):
        if i == 0:
            last_window = original_data[-window_size:]
        else:
            last_window = np.append(original_data[-window_size + i:], predicted_values[-(window_size - i):])
        predicted_values.append(np.mean(last_window))
    return np.array(predicted_values)

# Load data
df = pd.read_csv("new_vectors.csv")
#df = pd.read_csv("interpolated_data.csv")
df['Itime'] = np.ceil(df['Itime']).astype(int)

# Extract time series
time = df['Itime'].values
tsh = df['Itsh'].values
ft4 = df['Ift4'].values

# Compute weighted moving averages
tsh_weighted_avg = weighted_moving_average(tsh)
ft4_weighted_avg = weighted_moving_average(ft4)

# Forecast next values
num_forecast = 10
tsh_forecast = forecast_next_values(tsh, window_size=5, num_predictions=num_forecast)
ft4_forecast = forecast_next_values(ft4, window_size=5, num_predictions=num_forecast)

# Create future time steps
future_time = np.arange(time[-1] + 1, time[-1] + 1 + num_forecast)

# Plot results
plt.figure(figsize=(10, 6))

# Plot Itsh
plt.subplot(2, 1, 1)
plt.plot(time, tsh, 'o', label='Itsh Data', color='blue')
plt.plot(time, tsh_weighted_avg, '-', label='Weighted MA (Itsh)', color='blue')
plt.plot(future_time, tsh_forecast, 'o', label='Forecasted Itsh', color='red')
plt.plot(np.append(time, future_time), np.append(tsh, tsh_forecast), '--', color='red')
plt.title('Itsh: Weighted Moving Average & Forecast')
plt.xlabel('Time')
plt.ylabel('Itsh')
plt.legend()

# Plot Ift4
plt.subplot(2, 1, 2)
plt.plot(time, ft4, 'o', label='Ift4 Data', color='green')
plt.plot(time, ft4_weighted_avg, '-', label='Weighted MA (Ift4)', color='green')
plt.plot(future_time, ft4_forecast, 'o', label='Forecasted Ift4', color='red')
plt.plot(np.append(time, future_time), np.append(ft4, ft4_forecast), '--', color='red')
plt.title('Ift4: Weighted Moving Average & Forecast')
plt.xlabel('Time')
plt.ylabel('Ift4')
plt.legend()

plt.tight_layout()
plt.show()

# Save final forecasted values
print("\nLast forecasted Itsh value:", tsh_forecast[-1])
print("Last forecasted Ift4 value:", ft4_forecast[-1])
df_last_forecast = pd.DataFrame({
    'Last_Itsh': [tsh_forecast[-1]],
    'Last_Ift4': [ft4_forecast[-1]]
})
df_last_forecast.to_csv('w_mov_av.csv', index=False)
#df_last_forecast.to_csv('iw_mov_av.csv', index=False)
