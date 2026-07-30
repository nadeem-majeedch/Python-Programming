"""Generate synthetic credit card fraud detection dataset."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 5000
fraud_ratio = 0.03
n_fraud = int(n * fraud_ratio)
n_legit = n - n_fraud

classes = [0] * n_legit + [1] * n_fraud
np.random.shuffle(classes)

time_legit = np.random.exponential(72, n_legit)
time_fraud = np.random.exponential(24, n_fraud)
times = np.concatenate([time_legit, time_fraud])
times = np.clip(times, 0, 168).round(2)

V = np.random.randn(n, 28) * 0.8

fraud_indices = [i for i, c in enumerate(classes) if c == 1]
for i in fraud_indices:
    V[i, :8] += np.random.uniform(-3, 3, 8)
    V[i, 8:15] += np.random.uniform(-1.5, 1.5, 7)

amount_legit = np.random.lognormal(3.5, 0.8, n_legit)
amount_fraud = np.random.lognormal(4.2, 1.2, n_fraud)
amounts = np.concatenate([amount_legit, amount_fraud]).round(2)

df = pd.DataFrame(
    np.column_stack([times] + [V[:, i] for i in range(28)] + [amounts, classes]),
    columns=['Time'] + [f'V{i+1}' for i in range(28)] + ['Amount', 'Class']
)

for col in [f'V{i+1}' for i in range(28)]:
    mask = np.random.random(n) < 0.025
    df.loc[mask, col] = np.nan

out_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'credit_card_fraud.csv')
df.to_csv(out_path, index=False)
print(f"Generated {out_path} ({len(df)} rows, {n_fraud} fraud cases)")
