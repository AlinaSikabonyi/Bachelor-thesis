import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the data
#df = pd.read_csv("interpolated_data.csv")
df = pd.read_csv("new_vectors.csv")

# Check for NaN values
print("Check for NaN values in the data:")
print(df.isna().sum())

# Drop any rows with NaN values if necessary
df = df.dropna()

# Reset the index after dropping rows
df = df.set_index(pd.Index(range(len(df))))

# Extract series
time = df.index.values
tsh = df['Itsh'].values
ft4 = df['Ift4'].values

# Fit Holt-Winters model for TSH and FT4
tsh_model = ExponentialSmoothing(df['Itsh'], trend='add', damped_trend=True, seasonal=None).fit()
ft4_model = ExponentialSmoothing(df['Ift4'], trend='add', damped_trend=True, seasonal=None).fit()

# Forecast next 10 steps
forecast_steps = 10
tsh_forecast = tsh_model.forecast(steps=forecast_steps)
ft4_forecast = ft4_model.forecast(steps=forecast_steps)

# Future time points for forecasting
future_time = np.arange(time[-1] + 1, time[-1] + 1 + forecast_steps)

# Plotting the observed and forecasted data
plt.figure(figsize=(10, 5))
plt.plot(time, tsh, 'ro-', label='Observed TSH')
plt.plot(time, ft4, 'bo-', label='Observed FT4')
plt.plot(future_time, tsh_forecast, 'r--', label='Forecasted TSH')
plt.plot(future_time, ft4_forecast, 'b--', label='Forecasted FT4')
plt.xlabel('Time')
plt.ylabel('Hormone Levels')
plt.title('TSH and FT4 Forecasting (Exponential Smoothing)')
plt.legend()
plt.tight_layout()
plt.show()

# Save the last forecasted values (use .iloc[-1] for safe indexing)
df_last_forecast = pd.DataFrame({
    'Last_Itsh': [tsh_forecast.iloc[-1]],  # Last value from forecasted data (using .iloc[-1])
    'Last_Ift4': [ft4_forecast.iloc[-1]]   # Last value from forecasted data (using .iloc[-1])
})
#df_last_forecast.to_csv("iexp_sm.csv", index=False)
df_last_forecast.to_csv("exp_sm.csv", index=False)
# Print the last forecasted values
print("Last forecasted TSH:", tsh_forecast.iloc[-1])
print("Last forecasted FT4:", ft4_forecast.iloc[-1])


