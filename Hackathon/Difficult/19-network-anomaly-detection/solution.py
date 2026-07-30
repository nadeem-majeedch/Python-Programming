#!/usr/bin/env python3
"""Reference solution: Network Anomaly Detection
IsolationForest + feature engineering + temporal consistency.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

try:
    from sklearn.neural_network import MLPClassifier
    AUTOENCODER = True
except ImportError:
    AUTOENCODER = False

# ========================
# Load and prepare data
# ========================
df = pd.read_csv("data/network_traffic.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

# Ground truth
y_binary = (df["AnomalyType"] != "None").astype(int)
y_multi = df["AnomalyType"].values

print(f"Data: {len(df)} records")
print(f"Anomalies: {y_binary.sum()} ({y_binary.mean()*100:.2f}%)")

# ========================
# Feature engineering
# ========================
def engineer_features(df):
    df = df.copy()
    
    # Encode protocol
    le = LabelEncoder()
    df["Protocol_enc"] = le.fit_transform(df["Protocol"])
    
    # Rolling statistics (temporal features, respecting time order)
    window = 20
    for col in ["BytesSent", "BytesReceived", "NumPackets", "PacketSize", "ErrorRate"]:
        df[f"{col}_roll_mean"] = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window, min_periods=1).std()
        df[f"{col}_roll_max"] = df[col].rolling(window, min_periods=1).max()
        df[f"{col}_rate"] = df[col].diff().fillna(0)
    
    # Rate features (change per time step)
    df["bytes_sent_rate"] = df["BytesSent"].diff().fillna(0)
    df["packet_rate"] = df["NumPackets"].diff().fillna(0)
    
    # Ratio features
    df["send_recv_ratio"] = df["BytesSent"] / (df["BytesReceived"] + 1)
    df["bytes_per_packet"] = df["BytesSent"] / (df["NumPackets"] + 1)
    
    # IP features (frequency of same source/dest)
    df["src_freq"] = df.groupby("SourceIP")["SourceIP"].transform("count")
    df["dest_freq"] = df.groupby("DestIP")["DestIP"].transform("count")
    
    # Interaction features
    df["bytes_x_packets"] = df["BytesSent"] * df["NumPackets"] / 10000
    df["error_x_duration"] = df["ErrorRate"] * df["Duration"]
    
    # Sin/cos time encoding
    df["time_sin"] = np.sin(2 * np.pi * df["TimeOfDay"] / 24)
    df["time_cos"] = np.cos(2 * np.pi * df["TimeOfDay"] / 24)
    
    return df.fillna(0)

df_feat = engineer_features(df)
feature_cols = [c for c in df_feat.columns if c not in
                ["Timestamp", "SourceIP", "DestIP", "Protocol", "AnomalyType"]]

print(f"Features engineered: {len(feature_cols)}")

# ========================
# IsolationForest for anomaly detection
# ========================
print("\n" + "=" * 50)
print("ISOLATION FOREST ANOMALY DETECTION")
print("=" * 50)

# Train on first 60%, test on remaining 40% (temporal split)
split_idx = int(len(df_feat) * 0.6)
train_df = df_feat.iloc[:split_idx]
test_df = df_feat.iloc[split_idx:]

X_train = train_df[feature_cols].values
X_test = test_df[feature_cols].values
y_test_bin = y_binary[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train IsolationForest
iso = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42,
    n_jobs=-1
)
iso.fit(X_train_s)

# Predict: -1 = anomaly, 1 = normal
y_pred_iso = iso.predict(X_test_s)
y_pred_iso_bin = (y_pred_iso == -1).astype(int)

print("\nIsolationForest Results:")
print(classification_report(y_test_bin, y_pred_iso_bin))
print(f"F1: {f1_score(y_test_bin, y_pred_iso_bin):.4f}")
print(f"FP rate: {(~y_test_bin.astype(bool) & y_pred_iso_bin.astype(bool)).sum() / (~y_test_bin).sum():.4f}")

# Score-based detection with tunable threshold
scores = iso.score_samples(X_test_s)
threshold = np.percentile(scores, 2)
y_pred_score = (scores < threshold).astype(int)

print(f"\nScore-based (threshold={threshold:.3f}):")
print(classification_report(y_test_bin, y_pred_score))
f1_score_val = f1_score(y_test_bin, y_pred_score)
print(f"F1: {f1_score_val:.4f}")

# ========================
# Anomaly Type Classification
# ========================
print("\n" + "=" * 50)
print("ANOMALY TYPE CLASSIFICATION")
print("=" * 50)

# Among detected anomalies, classify type
from sklearn.ensemble import RandomForestClassifier

detected_mask = y_pred_score == 1
if detected_mask.sum() > 0:
    y_test_types = test_df["AnomalyType"].values[detected_mask]
    # Only use train data anomalies for training type classifier
    train_anom_mask = train_df["AnomalyType"] != "None"
    
    if train_anom_mask.sum() > 10:
        type_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        type_clf.fit(
            X_train[train_anom_mask.values],
            train_df["AnomalyType"].values[train_anom_mask]
        )
        y_pred_types = type_clf.predict(X_test[detected_mask])
        print(classification_report(y_test_types, y_pred_types))

# ========================
# Detection Latency
# ========================
print("\n" + "=" * 50)
print("DETECTION LATENCY")
print("=" * 50)

latencies = []
current_anomaly_start = None
for i in range(len(y_test_bin)):
    if y_test_bin[i] == 1 and current_anomaly_start is None:
        current_anomaly_start = i
    elif y_test_bin[i] == 1 and current_anomaly_start is not None:
        if y_pred_score[i] == 1:
            latencies.append(i - current_anomaly_start)
            current_anomaly_start = None
    else:
        if current_anomaly_start is not None and y_test_bin[i] == 0:
            # Anomaly ended without detection
            latencies.append(i - current_anomaly_start)
            current_anomaly_start = None

if latencies:
    print(f"Average detection latency: {np.mean(latencies):.2f} time steps")
    print(f"Median detection latency: {np.median(latencies):.2f} time steps")
    print(f"Latency < 3: {(np.array(latencies) < 3).mean()*100:.1f}%")
else:
    print("No anomalies detected in test set")

# ========================
# Temporal Consistency Post-Processing
# ========================
print("\n" + "=" * 50)
print("TEMPORAL CONSISTENCY POST-PROCESSING")
print("=" * 50)

def temporal_smoothing(predictions, min_anomaly_window=3):
    """Remove isolated anomaly predictions."""
    smoothed = predictions.copy()
    for i in range(len(predictions)):
        if predictions[i] == 1:
            # Check if surrounded by normal
            window_before = predictions[max(0, i-min_anomaly_window):i]
            window_after = predictions[i+1:min(len(predictions), i+min_anomaly_window+1)]
            if window_before.sum() < 1 and window_after.sum() < 1:
                smoothed[i] = 0
    return smoothed

y_pred_smooth = temporal_smoothing(y_pred_score, min_anomaly_window=3)
f1_smooth = f1_score(y_test_bin, y_pred_smooth)
print(f"\nAfter temporal smoothing:")
print(classification_report(y_test_bin, y_pred_smooth))
print(f"F1: {f1_smooth:.4f}")

# ========================
# Final Summary
# ========================
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
fpr = (~y_test_bin.astype(bool) & y_pred_smooth.astype(bool)).sum() / (~y_test_bin).sum()
print(f"IsolationForest F1: {f1_score_val:.4f}")
print(f"After smoothing F1: {f1_smooth:.4f}")
print(f"False positive rate: {fpr:.4f}")
print(f"F1 > 0.85: {'PASS' if f1_smooth > 0.85 else 'NEEDS IMPROVEMENT'}")
print(f"FPR < 5%: {'PASS' if fpr < 0.05 else 'NEEDS IMPROVEMENT'}")
