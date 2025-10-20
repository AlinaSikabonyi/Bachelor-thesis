import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Moving average function (for visualization only)
def moving_average(data, window_size=6):
    """Compute the moving average for the data."""
    moving_avg = []
    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window_data = data[start_index:i + 1]
        moving_avg.append(np.mean(window_data))
    return np.array(moving_avg)

# Forecast directly from raw data instead of moving average
def forecast_next_values(original_data, window_size=5, num_predictions=10, damping_factor=0.9):
    """Forecast the next values using the last raw values with damping."""
    predicted_values = []

    for i in range(num_predictions):
        if i == 0:
            last_window = original_data[-window_size:]
        else:
            last_window = np.append(original_data[-window_size + i:], predicted_values[-(window_size - i):])
        forecast = np.mean(last_window) * (damping_factor ** i)
        predicted_values.append(forecast)

    return np.array(predicted_values)

# Load the data
df = pd.read_csv("new_vectors.csv")
#df = pd.read_csv("interpolated_data.csv")
df['Itime'] = np.ceil(df['Itime']).astype(int)

# Extract time series
time = df['Itime'].values
tsh = df['Itsh'].values
ft4 = df['Ift4'].values

# Compute moving averages for visualization
tsh_moving_avg = moving_average(tsh)
ft4_moving_avg = moving_average(ft4)

# Forecast based on raw data
num_forecast = 10
tsh_forecast = forecast_next_values(tsh, window_size=5, num_predictions=num_forecast, damping_factor=0.9)
ft4_forecast = forecast_next_values(ft4, window_size=5, num_predictions=num_forecast, damping_factor=0.9)

# Generate future time points
future_time = np.arange(time[-1] + 1, time[-1] + 1 + num_forecast)

# Plotting
plt.figure(figsize=(10, 6))

# Plot Itsh
plt.subplot(2, 1, 1)
plt.plot(time, tsh, 'o', label='Itsh Data Points', color='blue')
plt.plot(time, tsh_moving_avg, '-', label='Moving Average (Itsh)', color='blue')
plt.plot(future_time, tsh_forecast, 'o', label='Forecasted Itsh', color='red')
plt.plot(np.append(time, future_time), np.append(tsh, tsh_forecast), '--', color='red')  # raw + forecast
plt.title('Moving Average for TSH')
plt.xlabel('Time')
plt.ylabel('TSH')
plt.legend()

# Plot Ift4
plt.subplot(2, 1, 2)
plt.plot(time, ft4, 'o', label='Ift4 Data Points', color='green')
plt.plot(time, ft4_moving_avg, '-', label='Moving Average (Ift4)', color='green')
plt.plot(future_time, ft4_forecast, 'o', label='Forecasted Ift4', color='red')
plt.plot(np.append(time, future_time), np.append(ft4, ft4_forecast), '--', color='red')  # raw + forecast
plt.title('Moving Average for FT4')
plt.xlabel('Time')
plt.ylabel('FT4')
plt.legend()

plt.tight_layout()
plt.show()


print("\nLast forecasted Itsh value:", tsh_forecast[-1])
print("Last forecasted Ift4 value:", ft4_forecast[-1])
df_last_values = pd.DataFrame({
    'Last_Itsh': [tsh_forecast[-1]],
    'Last_Ift4': [ft4_forecast[-1]]
})
df_last_values.to_csv('mov_av.csv', index=False)
#df_last_values.to_csv('imov_av.csv', index=False)