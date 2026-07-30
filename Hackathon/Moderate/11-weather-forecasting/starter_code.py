"""Starter code for Weather Forecasting."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('data/weather_data.csv', parse_dates=['Date'], index_col='Date')

print("=== Weather Dataset ===")
print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Temp_Max'], label='Temp_Max', alpha=0.7)
plt.plot(df.index, df['Temp_Min'], label='Temp_Min', alpha=0.7)
plt.title('Daily Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.legend()
plt.show()

df['Temp_Max_MA7'] = df['Temp_Max'].rolling(window=7).mean()
df['Precipitation'].fillna(0, inplace=True)

train = df.iloc[:800]
test = df.iloc[800:]

last_values = train['Temp_Max'].iloc[-7:].mean()
test['Forecast'] = last_values

rmse = np.sqrt(mean_squared_error(test['Temp_Max'], test['Forecast']))
mae = mean_absolute_error(test['Temp_Max'], test['Forecast'])
print(f"\nSimple Moving Average Forecast:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
