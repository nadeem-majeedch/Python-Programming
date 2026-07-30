#!/usr/bin/env python3
"""Starter code: Medical Diagnosis with Limited Data"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

# Load data
df = pd.read_csv("data/medical_diagnosis.csv")
X = df.drop(columns=["Diagnosis"])
y = df["Diagnosis"]

print(f"Data shape: {X.shape}")
print(f"Positive rate: {y.mean()*100:.1f}%")

# Simple train-test split (not ideal for small data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Basic LogisticRegression (will overfit with 30 features on 300 samples)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print(f"AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Brier Score: {brier_score_loss(y_test, y_prob):.4f}")

# TODO:
# 1. Add regularization (Ridge, Lasso, ElasticNet)
# 2. Perform feature selection (RFE, SelectKBest, L1 regularization)
# 3. Use proper validation (LOO-CV, bootstrap)
# 4. Calibrate probabilities (Platt scaling)
# 5. Report confidence intervals on AUC
