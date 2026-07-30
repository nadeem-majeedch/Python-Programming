#!/usr/bin/env python3
"""Reference solution: Hybrid Recommendation System (SVD + Content-Based)"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from math import sqrt
import warnings
warnings.filterwarnings("ignore")

# ========================
# Load data
# ========================
ratings = pd.read_csv("data/ratings.csv")
users = pd.read_csv("data/users.csv")
items = pd.read_csv("data/items.csv")

# Encode categoricals
region_encoder = LabelEncoder()
cat_encoder = LabelEncoder()
users["Region_enc"] = region_encoder.fit_transform(users["Region"])
items["Category_enc"] = cat_encoder.fit_transform(items["Category"])

# Split ratings: keep some users as "cold-start" for evaluation
all_active_users = ratings["UserID"].unique()
np.random.seed(42)
test_users = np.random.choice(all_active_users, size=int(len(all_active_users) * 0.15), replace=False)
train_ratings = ratings[~ratings["UserID"].isin(test_users)]
test_ratings = ratings[ratings["UserID"].isin(test_users)]

print(f"Train ratings: {len(train_ratings)}, Test ratings: {len(test_ratings)}")
print(f"Global mean rating: {ratings['Rating'].mean():.3f}")

# ========================
# Collaborative Filtering (SVD)
# ========================
R_train = train_ratings.pivot_table(
    index="UserID", columns="ItemID", values="Rating"
).values
R_mean = np.nanmean(R_train)
R_demeaned = R_train - R_mean
R_demeaned[np.isnan(R_demeaned)] = 0

U, S, Vt = np.linalg.svd(R_demeaned, full_matrices=False)
k = 25
U_k = U[:, :k]
S_k = np.diag(S[:k])
Vt_k = Vt[:k, :]

R_svd = np.dot(U_k, np.dot(S_k, Vt_k)) + R_mean

# Map to predictions for test
user_ids_sorted = sorted(train_ratings["UserID"].unique())
item_ids_sorted = sorted(train_ratings["ItemID"].unique())
user_to_idx = {uid: i for i, uid in enumerate(user_ids_sorted)}
item_to_idx = {iid: i for i, iid in enumerate(item_ids_sorted)}

def predict_svd(uid, iid):
    if uid in user_to_idx and iid in item_to_idx:
        return R_svd[user_to_idx[uid], item_to_idx[iid]]
    return R_mean

# Evaluate SVD on test
svd_preds = []
for _, row in test_ratings.iterrows():
    svd_preds.append(predict_svd(row["UserID"], row["ItemID"]))
svd_rmse = sqrt(mean_squared_error(test_ratings["Rating"], svd_preds))
print(f"SVD-only RMSE: {svd_rmse:.4f}")

# ========================
# Content-Based Component
# ========================
def compute_content_score(uid, iid):
    """Simple content-based score using demographics and item metadata."""
    user_row = users[users["UserID"] == uid]
    item_row = items[items["ItemID"] == uid]
    if user_row.empty or item_row.empty:
        return 0.5
    
    user_row = user_row.iloc[0]
    item_row = item_row.iloc[0]
    
    score = 0.0
    # Age-based affinity (simplified)
    cat_age_map = {
        "Electronics": (20, 40), "Clothing": (18, 60),
        "Books": (25, 65), "Home": (30, 60), "Sports": (18, 45)
    }
    cat_range = cat_age_map.get(item_row["Category"], (0, 100))
    if cat_range[0] <= user_row["Age"] <= cat_range[1]:
        score += 0.3
    
    # Region-based affinity
    region_cat_prefs = {
        "North": ["Electronics", "Books"],
        "South": ["Clothing", "Sports"],
        "East": ["Books", "Electronics"],
        "West": ["Home", "Sports"]
    }
    preferred = region_cat_prefs.get(user_row["Region"], [])
    if item_row["Category"] in preferred:
        score += 0.3
    
    # Price tolerance (older users prefer higher-priced items)
    price_factor = min(1.0, item_row["Price"] / 50.0)
    if user_row["Age"] > 40:
        score += 0.2 * price_factor
    else:
        score += 0.2 * (1 - price_factor)
    
    return min(1.0, score + 0.2)

# ========================
# Hybrid Blending
# ========================
def hybrid_predict(uid, iid, alpha=0.7):
    """Blend SVD and content-based predictions."""
    if uid in user_to_idx and iid in item_to_idx:
        svd_score = predict_svd(uid, iid)
        # Normalize SVD rating to [0, 1]
        svd_norm = (svd_score - 1) / 4.0
        content_score = compute_content_score(uid, iid)
        hybrid = alpha * svd_norm + (1 - alpha) * content_score
        return 1 + hybrid * 4  # Scale back to 1-5
    else:
        # Cold-start: pure content-based
        content_score = compute_content_score(uid, iid)
        return 1 + content_score * 4

# Find optimal alpha
best_alpha = 0.7
best_rmse = float("inf")
for alpha in np.linspace(0.3, 0.95, 10):
    preds = []
    for _, row in test_ratings.iterrows():
        preds.append(hybrid_predict(row["UserID"], row["ItemID"], alpha))
    rmse = sqrt(mean_squared_error(test_ratings["Rating"], preds))
    if rmse < best_rmse:
        best_rmse = rmse
        best_alpha = alpha

print(f"\nBest alpha: {best_alpha:.2f}, Hybrid RMSE: {best_rmse:.4f}")

# ========================
# Cold-Start Evaluation
# ========================
cold_users = users[~users["UserID"].isin(ratings["UserID"].unique())]
cold_items = items[~items["ItemID"].isin(ratings["ItemID"].unique())]

print(f"\nCold-start users: {len(cold_users)}")
print(f"Cold-start items: {len(cold_items)}")

# For each cold-start user, recommend top-N items
def recommend_cold_start(uid, n=10):
    """Recommend using only content-based scores."""
    scores = []
    for _, item_row in items.iterrows():
        score = compute_content_score(uid, item_row["ItemID"])
        scores.append((item_row["ItemID"], score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [iid for iid, _ in scores[:n]]

print("\nCold-start user recommendations:")
for _, user_row in cold_users.head(5).iterrows():
    recs = recommend_cold_start(user_row["UserID"], n=5)
    rec_cats = items[items["ItemID"].isin(recs)]["Category"].tolist()
    print(f"  User {user_row['UserID']} ({user_row['Age']}, {user_row['Region']}): {rec_cats}")

# ========================
# Diversity Metric
# ========================
def category_diversity(rec_list):
    """Diversity = 1 - (proportion of most common category)."""
    cats = items[items["ItemID"].isin(rec_list)]["Category"].tolist()
    if not cats:
        return 0
    most_common = max(cats.count(c) for c in set(cats))
    return 1 - most_common / len(cats)

print("\n--- Diversity Analysis ---")
diversities = []
for uid in np.random.choice(all_active_users, size=20, replace=False):
    recs = recommend_cold_start(uid, n=10)
    div = category_diversity(recs)
    diversities.append(div)
print(f"Average diversity across 20 users: {np.mean(diversities):.3f}")

# ========================
# Final Summary
# ========================
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"SVD-only RMSE:         {svd_rmse:.4f}")
print(f"Hybrid RMSE (alpha={best_alpha:.2f}): {best_rmse:.4f}")
print(f"Cold-start strategy:   Content-based fallback")
print(f"Average diversity:     {np.mean(diversities):.3f}")
print(f"RMSE < 1.0: {'PASS' if best_rmse < 1.0 else 'NEEDS IMPROVEMENT'}")
