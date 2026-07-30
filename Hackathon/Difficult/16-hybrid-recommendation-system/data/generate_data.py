#!/usr/bin/env python3
"""Generate synthetic recommendation system data."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_users = 500
n_items = 200
n_interactions = 10000
n_cold_users = 30
n_cold_items = 20

# Generate user demographics
regions = ["North", "South", "East", "West"]
categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]

users = []
for uid in range(1, n_users + 1):
    users.append({
        "UserID": uid,
        "Age": int(np.random.normal(35, 12)),
        "Region": np.random.choice(regions, p=[0.25, 0.25, 0.25, 0.25])
    })
df_users = pd.DataFrame(users)

# Generate item metadata
items = []
for iid in range(1, n_items + 1):
    cat = np.random.choice(categories)
    items.append({
        "ItemID": iid,
        "Category": cat,
        "Price": round(np.random.lognormal(3.5, 0.5), 2),
        "Popularity": np.random.uniform(0, 1)
    })
df_items = pd.DataFrame(items)

# Create latent factors for users and items (underlying preference structure)
n_factors = 10
np.random.seed(42)
user_factors = np.random.randn(n_users, n_factors)
item_factors = np.random.randn(n_items, n_factors)

# Region and category biases
region_bias = {"North": 0.2, "South": -0.1, "East": 0.3, "West": -0.2}
cat_bias = {"Electronics": 0.3, "Clothing": -0.2, "Books": 0.1, "Home": 0.0, "Sports": -0.1}

# Generate ratings
ratings = []
interaction_count = 0
attempts = 0
max_attempts = 100000

while interaction_count < n_interactions and attempts < max_attempts:
    attempts += 1
    uid = np.random.randint(1, n_users + 1)
    # Skip cold-start users (they won't have ratings)
    if uid > n_users - n_cold_users:
        continue
    iid = np.random.randint(1, n_items + 1)
    # Skip cold-start items (they won't have ratings)
    if iid > n_items - n_cold_items:
        continue
    
    u_idx = uid - 1
    i_idx = iid - 1
    
    # Rating = latent dot product + biases + noise
    rating = np.dot(user_factors[u_idx], item_factors[i_idx])
    user_row = df_users.iloc[u_idx]
    item_row = df_items.iloc[i_idx]
    rating += region_bias.get(user_row["Region"], 0)
    rating += cat_bias.get(item_row["Category"], 0)
    rating += np.random.normal(0, 0.5)
    
    rating = np.clip(rating, 0.5, 5.0)
    rating = round(rating)
    
    ratings.append({
        "UserID": uid,
        "ItemID": iid,
        "Rating": int(rating)
    })
    interaction_count += 1

df_ratings = pd.DataFrame(ratings)

# Ensure cold-start users have no ratings
cold_uids = list(range(n_users - n_cold_users + 1, n_users + 1))
df_ratings = df_ratings[~df_ratings["UserID"].isin(cold_uids)]

# Ensure cold-start items have no ratings
cold_iids = list(range(n_items - n_cold_items + 1, n_items + 1))
df_ratings = df_ratings[~df_ratings["ItemID"].isin(cold_iids)]

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)

df_ratings.to_csv(os.path.join(out_dir, "ratings.csv"), index=False)
df_users.to_csv(os.path.join(out_dir, "users.csv"), index=False)
df_items.to_csv(os.path.join(out_dir, "items.csv"), index=False)

print(f"Ratings: {len(df_ratings)}, Users: {len(df_users)}, Items: {len(df_items)}")
print(f"Cold-start users: {n_cold_users}, Cold-start items: {n_cold_items}")
