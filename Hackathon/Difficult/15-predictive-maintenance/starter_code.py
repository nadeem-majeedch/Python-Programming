#!/usr/bin/env python3
"""Starter code: Predictive Maintenance"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score

# Load data
df = pd.read_csv("data/predictive_maintenance.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Basic features
features = ["Temperature", "Vibration", "Pressure", "RPM",
            "OperatingHours", "LastMaintenanceDays"]
X = df[features]
y = df["Failure"]

# Train/test split (temporal - use first 80% for train)
df_sorted = df.sort_values("Timestamp")
split_idx = int(len(df_sorted) * 0.8)
train_df = df_sorted.iloc[:split_idx]
test_df = df_sorted.iloc[split_idx:]

X_train = train_df[features]
y_train = train_df["Failure"]
X_test = test_df[features]
y_test = test_df["Failure"]

print(f"Train failures: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Test failures: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# Train RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nRandomForest Baseline:")
print(classification_report(y_test, y_pred))
print(f"F1 (failure class): {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")

# TODO:
# 1. Engineer temporal features: rolling windows, rate of change, lag features
# 2. Handle class imbalance better (SMOTE, scale_pos_weight, threshold tuning)
# 3. Implement early warning time metric
# 4. Try XGBoost/LightGBM with proper tuning
