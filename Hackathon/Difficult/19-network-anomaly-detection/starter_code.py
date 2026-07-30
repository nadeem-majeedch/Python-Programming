#!/usr/bin/env python3
"""Starter code: Network Anomaly Detection"""

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix

# Load data
df = pd.read_csv("data/network_traffic.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Anomaly distribution:\n{df['AnomalyType'].value_counts()}")

y_true = (df["AnomalyType"] != "None").astype(int)

# Basic threshold-based detection
# Simple heuristic: Z-score on BytesSent and NumPackets
features = ["BytesSent", "NumPackets", "PacketSize", "ErrorRate"]
anomaly_scores = np.zeros(len(df))

for col in features:
    mean_v = df[col].mean()
    std_v = df[col].std()
    if std_v > 0:
        anomaly_scores += np.abs((df[col] - mean_v) / std_v)
    else:
        print(f"Warning: {col} has zero std")

threshold = anomaly_scores.quantile(0.95)
y_pred = (anomaly_scores > threshold).astype(int)

print(f"\nThreshold-based detection:")
print(classification_report(y_true, y_pred))
print(f"F1: {f1_score(y_true, y_pred):.4f}")

# TODO:
# 1. Implement proper anomaly detection (IsolationForest, Autoencoder)
# 2. Engineer rolling/rate features for temporal patterns
# 3. Add post-processing for temporal consistency
# 4. Classify anomaly types (multi-class)
# 5. Measure detection latency
