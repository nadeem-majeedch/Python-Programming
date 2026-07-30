#!/usr/bin/env python3
"""Starter code: End-to-End ML Pipeline with Deployment Constraints"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load data
df = pd.read_csv("data/production_pipeline_data.csv")
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\n{df.isna().sum()}")
print(f"Target distribution:\n{df['Target'].value_counts()}")

# Simple split
X = df.drop(columns=["Target", "IsDrifted"])
y = df["Target"]

# Basic preprocessing
X = pd.get_dummies(X, columns=["CustomerTier", "Region", "Feedback"], drop_first=True)
X["SignupDate"] = pd.to_datetime(X["SignupDate"])
X["SignupYear"] = X["SignupDate"].dt.year
X["SignupMonth"] = X["SignupDate"].dt.month
X = X.drop(columns=["SignupDate"])
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"\nAccuracy: {(y_pred == y_test).mean():.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.4f}")
print(classification_report(y_test, y_pred))

# TODO:
# 1. Build proper sklearn Pipeline with ColumnTransformer
# 2. Handle missing values, categoricals, text, dates in a single pipeline
# 3. Meet size (<50MB) and speed (<100ms) constraints
# 4. Add drift detection on the last 2000 rows
# 5. Create Flask/FastAPI serving endpoint
# 6. Ensure reproducibility
