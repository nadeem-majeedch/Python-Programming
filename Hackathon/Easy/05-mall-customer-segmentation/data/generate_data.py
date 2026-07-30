#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 300

# Generate 5 distinct cluster centers for realism
centers = [
    (20, 20, 80),   # young, low income, high spend
    (25, 80, 80),   # young, high income, high spend
    (45, 60, 50),   # middle-aged, medium
    (55, 40, 20),   # older, low-medium income, low spend
    (35, 90, 30),   # middle-aged, high income, low spend
]
pts_per_center = n // 5

ages = []
incomes = []
scores = []

for age_c, inc_c, score_c in centers:
    ages.extend(np.random.normal(age_c, 5, pts_per_center))
    incomes.extend(np.random.normal(inc_c, 8, pts_per_center))
    scores.extend(np.random.normal(score_c, 7, pts_per_center))

df = pd.DataFrame({
    "CustomerID": [f"CUST_{i:04d}" for i in range(1, n + 1)],
    "Age": np.clip(np.round(ages).astype(int), 18, 70),
    "AnnualIncome": np.clip(np.round(incomes, 1), 15, 120),
    "SpendingScore": np.clip(np.round(scores, 1), 1, 100),
})

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out = os.path.join(os.getcwd(), "mall_customers.csv")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
