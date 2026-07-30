"""Generate synthetic real estate price prediction dataset."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 3000

sqft = np.random.lognormal(7.5, 0.35, n).round(-1).clip(500, 10000).astype(int)
bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n, p=[0.02, 0.10, 0.35, 0.33, 0.15, 0.05])
bathrooms = np.where(bedrooms == 1, 1,
                     np.where(bedrooms <= 3,
                              np.random.choice([1, 1.5, 2], n, p=[0.05, 0.10, 0.85]),
                              np.random.choice([2, 2.5, 3, 4], n, p=[0.30, 0.30, 0.30, 0.10])))
year_built = np.random.randint(1950, 2024, n)
lot_size = np.random.lognormal(8.5, 0.5, n).round(-2).clip(1500, 50000).astype(int)
has_garage = np.random.choice([0, 1], n, p=[0.15, 0.85])
has_pool = np.random.choice([0, 1], n, p=[0.88, 0.12])
zip_codes = np.random.choice([85001, 85002, 85003, 85004, 85005], n)
dist_downtown = np.random.lognormal(2.0, 0.8, n).round(1).clip(0.5, 50)
crime_rate = np.random.exponential(8, n).round(1).clip(0.5, 50)
school_rating = np.random.uniform(1, 10, n).round(1)

age = 2024 - year_built
base_price = (
    50000 + sqft * 120 + bedrooms * 15000 + bathrooms * 8000 + (lot_size * 0.5)
    + has_garage * 15000 + has_pool * 25000
    + np.where(zip_codes == 85001, 30000, np.where(zip_codes == 85003, -10000, 0))
    + (10 - school_rating) * (-5000) - dist_downtown * 2000 - crime_rate * 1000
    + np.where(age > 50, age * (-200), np.where(age < 10, age * 200, 0))
    + np.random.normal(0, 25000, n)
)
price = np.maximum(base_price, 30000).round(-3).astype(int)

df = pd.DataFrame({
    'SqFt': sqft, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms,
    'YearBuilt': year_built, 'LotSize': lot_size, 'HasGarage': has_garage,
    'HasPool': has_pool, 'ZipCode': zip_codes, 'DistanceToDowntown': dist_downtown,
    'CrimeRate': crime_rate, 'SchoolRating': school_rating, 'Price': price
})

out_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'real_estate.csv')
df.to_csv(out_path, index=False)
print(f"Generated {out_path} ({len(df)} rows)")
