#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 1000

genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror"]
genre_bonus = {"Action": 0.0, "Comedy": 0.5, "Drama": 1.5, "Sci-Fi": 0.3, "Horror": -0.5}

genre_choices = np.random.choice(genres, n, p=[0.25, 0.20, 0.25, 0.15, 0.15])
runtime = np.random.uniform(80, 200, n)
budget = np.random.uniform(1, 200, n)
release_year = np.random.randint(1990, 2024, n)

rating = (
    3.0
    + 0.3 * np.log(budget)
    - 0.01 * runtime
    + np.array([genre_bonus[g] for g in genre_choices])
    + np.random.normal(0, 0.5, n)
)

df = pd.DataFrame({
    "MovieID": [f"MOV_{i:04d}" for i in range(1, n + 1)],
    "Genre": genre_choices,
    "Runtime": np.round(runtime, 0).astype(int),
    "Budget": np.round(budget, 1),
    "ReleaseYear": release_year,
    "IMDbRating": np.clip(np.round(rating, 1), 1.0, 10.0),
})

out = os.path.join(os.getcwd(), "movie_ratings.csv")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
