#!/usr/bin/env python3
"""Starter code: Hybrid Recommendation System"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# Load data
ratings = pd.read_csv("data/ratings.csv")
users = pd.read_csv("data/users.csv")
items = pd.read_csv("data/items.csv")

print(f"Ratings: {len(ratings)}")
print(f"Users: {len(users)} (cold-start: {users[users['UserID'].isin(ratings['UserID'].unique())].shape[0]} active)")
print(f"Items: {len(items)} (cold-start: {items[~items['ItemID'].isin(ratings['ItemID'].unique())].shape[0]} new)")

# SVD via numpy
R = ratings.pivot_table(index="UserID", columns="ItemID", values="Rating").values
R_mean = np.nanmean(R)
R_demeaned = R - R_mean
R_demeaned[np.isnan(R_demeaned)] = 0

U, S, Vt = np.linalg.svd(R_demeaned, full_matrices=False)
k = 20
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]

# Reconstruct
R_pred = np.dot(U_k, np.dot(S_k, Vt_k)) + R_mean

# Evaluate on known ratings
mask = ~np.isnan(R)
rmse = sqrt(np.mean((R[mask] - R_pred[mask]) ** 2))
print(f"\nBasic SVD RMSE: {rmse:.4f}")

# TODO:
# 1. Split data properly for evaluation (training set vs test set)
# 2. Implement proper matrix factorization (SVD ++ or ALS)
# 3. Add content-based component using user/item metadata
# 4. Blend collaborative + content-based predictions
# 5. Handle cold-start users and items
# 6. Evaluate diversity and precision@k
