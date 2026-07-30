"""Generate synthetic weather forecasting dataset."""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

np.random.seed(42)

n = 1095
dates = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(n)]
day_of_year = np.array([d.timetuple().tm_yday for d in dates])

base_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
temp_max = base_temp + np.random.normal(0, 4, n) + 3
temp_min = base_temp + np.random.normal(0, 4, n) - 5
temp_min = np.minimum(temp_min, temp_max - 1)

humidity = 60 + 15 * np.sin(2 * np.pi * (day_of_year - 180) / 365) + np.random.normal(0, 10, n)
humidity = np.clip(humidity, 10, 100).round(1)

wind_speed = np.random.lognormal(2.0, 0.4, n).round(1)
precip_prob = 0.20 + 0.25 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
precip = np.where(np.random.random(n) < precip_prob, np.random.exponential(5, n).round(1), 0.0)
precip = np.clip(precip, 0, 80)

pressure = 1013 + 20 * np.sin(2 * np.pi * (day_of_year - 120) / 365) + np.random.normal(0, 8, n)
pressure = pressure.round(1)

missing_mask = np.random.random(n) < 0.03
precip[missing_mask] = np.nan

df = pd.DataFrame({
    'Date': [d.strftime('%Y-%m-%d') for d in dates],
    'Temp_Max': temp_max.round(1),
    'Temp_Min': temp_min.round(1),
    'Humidity': humidity,
    'WindSpeed': wind_speed,
    'Precipitation': precip,
    'Pressure': pressure
})

out_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'weather_data.csv')
df.to_csv(out_path, index=False)
print(f"Generated {out_path} ({len(df)} rows)")
