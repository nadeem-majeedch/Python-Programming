import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/mall_customers.csv")

X = df[["Age", "AnnualIncome", "SpendingScore"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
inertias = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

# Plot elbow
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, "bo-")
ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Method")
ax2.plot(K_range, sil_scores, "ro-")
ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette Score"); ax2.set_title("Silhouette Analysis")
plt.tight_layout()
plt.savefig("elbow_analysis.png")
print("Saved elbow_analysis.png")

# Optimal k
optimal_k = K_range[np.argmax(sil_scores)]
print(f"\nOptimal k: {optimal_k} (silhouette score: {max(sil_scores):.4f})")

# K-Means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"], cmap="viridis", edgecolors="k")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.title(f"PCA Visualization (k={optimal_k})")
plt.tight_layout()
plt.savefig("pca_clusters.png")
print("Saved pca_clusters.png")

# Cluster profiling
print(f"\n=== Cluster Profiles (k={optimal_k}) ===")
for cluster in sorted(df["Cluster"].unique()):
    subset = df[df["Cluster"] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"  Size: {len(subset)}")
    print(f"  Avg Age: {subset['Age'].mean():.1f}")
    print(f"  Avg Income: {subset['AnnualIncome'].mean():.1f}")
    print(f"  Avg Spending: {subset['SpendingScore'].mean():.1f}")

# Silhouette score on original scaled data
print(f"\nSilhouette Score for k={optimal_k}: {silhouette_score(X_scaled, kmeans.labels_):.4f}")
