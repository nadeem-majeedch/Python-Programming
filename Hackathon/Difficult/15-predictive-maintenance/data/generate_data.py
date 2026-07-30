#!/usr/bin/env python3
"""Generate synthetic predictive maintenance sensor data."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_machines = 100
n_records_per_machine = 50  # 5000 total
failure_rate = 0.05

records = []
for mid in range(1, n_machines + 1):
    base_temp = np.random.uniform(60, 80)
    base_vibration = np.random.uniform(0.5, 2.0)
    base_pressure = np.random.uniform(80, 120)
    base_rpm = np.random.uniform(800, 1500)
    
    machine_health = 1.0  # starts healthy, degrades over time
    failure_threshold = np.random.uniform(0.15, 0.25)
    in_degradation = False
    degradation_start = 0
    will_fail = np.random.random() < failure_rate
    
    for hour in range(n_records_per_machine):
        # Simulate degradation
        if will_fail and hour > 20 and not in_degradation:
            if np.random.random() < 0.3:
                in_degradation = True
                degradation_start = hour
        
        if in_degradation:
            progress = (hour - degradation_start) / (n_records_per_machine - degradation_start)
            machine_health = max(0.05, 1.0 - progress * 1.2)
        else:
            machine_health = min(1.0, machine_health + np.random.uniform(-0.02, 0.03))
        
        # Sensor readings with noise
        temp = base_temp + (1 - machine_health) * np.random.uniform(10, 30) + np.random.normal(0, 2)
        vibration = base_vibration + (1 - machine_health) * np.random.uniform(1, 4) + np.random.normal(0, 0.3)
        pressure = base_pressure + (1 - machine_health) * np.random.uniform(-15, 15) + np.random.normal(0, 3)
        rpm = base_rpm + (1 - machine_health) * np.random.uniform(-200, 100) + np.random.normal(0, 30)
        
        operating_hours = (mid * n_records_per_machine + hour) * np.random.uniform(0.9, 1.1)
        last_maint_days = np.random.poisson(30) if not in_degradation else max(0, 60 - (hour - degradation_start) * 3)
        
        # Failure in next 24h?
        failure_in_24h = 0
        if in_degradation and machine_health < failure_threshold:
            failure_in_24h = 1
        
        records.append({
            "MachineID": mid,
            "Timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=hour),
            "Temperature": round(temp, 2),
            "Vibration": round(vibration, 3),
            "Pressure": round(pressure, 2),
            "RPM": round(rpm, 1),
            "OperatingHours": round(operating_hours, 1),
            "LastMaintenanceDays": round(last_maint_days, 1),
            "Failure": failure_in_24h
        })

df = pd.DataFrame(records)
out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "predictive_maintenance.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} sensor readings (failures: {df['Failure'].sum()}) -> {out_path}")
