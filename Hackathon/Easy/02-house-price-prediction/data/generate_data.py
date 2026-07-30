#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 1500

sqft = np.random.uniform(800, 5000, n)
bedrooms = np.random.randint(1, 7, n)
bathrooms = np.random.randint(1, 6, n)
year_built = np.random.randint(1950, 2024, n)
lot_size = np.random.uniform(2000, 20000, n)
garage = np.random.randint(0, 4, n)

price = (
    100 * sqft
    + 15000 * bedrooms
    + 10000 * bathrooms
    + 500 * (2023 - year_built)
    + 20 * lot_size
    + 8000 * garage
    + np.random.normal(0, 20000, n)
)

df = pd.DataFrame({
    "SqFt": np.round(sqft, 0).astype(int),
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "YearBuilt": year_built,
    "LotSize": np.round(lot_size, 0).astype(int),
    "Garage": garage,
    "Price": np.round(price, 2),
})

out = os.path.join(os.getcwd(), "house_prices.csv")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
