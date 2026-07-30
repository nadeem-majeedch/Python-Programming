import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("data/mall_customers.csv")

X = df[["AnnualIncome", "SpendingScore"]]

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

plt.scatter(X["AnnualIncome"], X["SpendingScore"], c=df["Cluster"], cmap="viridis")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customer Segments (k=5)")
plt.savefig("clusters.png")
print("Saved clusters.png")
