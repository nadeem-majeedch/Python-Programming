#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    "acidity": np.round(np.random.uniform(3, 8, n), 2),
    "sugar": np.round(np.random.uniform(1, 20, n), 2),
    "alcohol": np.round(np.random.uniform(8, 15, n), 2),
    "pH": np.round(np.random.uniform(2.8, 4.2, n), 2),
    "sulphates": np.round(np.random.uniform(0.3, 0.8, n), 3),
})

# Quality formula
raw = (3
       + 0.5 * df["alcohol"]
       + 0.3 * df["sugar"]
       - 0.4 * df["acidity"]
       + 0.2 * df["pH"]
       + np.random.normal(0, 1.0, n))

df["quality"] = np.clip(np.round(raw).astype(int), 3, 8)
df["quality_label"] = np.where(df["quality"] >= 7, "good", "bad")

out = os.path.join(os.getcwd(), "wine_quality.csv")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
