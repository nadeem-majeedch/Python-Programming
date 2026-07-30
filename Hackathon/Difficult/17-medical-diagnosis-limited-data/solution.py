#!/usr/bin/env python3
"""Reference solution: Medical Diagnosis with Limited Data
Ridge regression + feature selection + bootstrap confidence intervals.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import cross_val_predict, LeaveOneOut, StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ========================
# Load data
# ========================
df = pd.read_csv("data/medical_diagnosis.csv")
X = df.drop(columns=["Diagnosis"]).values
feature_names = df.drop(columns=["Diagnosis"]).columns.tolist()
y = df["Diagnosis"].values

print(f"Data: {X.shape}, Positive: {y.sum()} ({y.mean()*100:.1f}%)")

# ========================
# Approach 1: L1-regularized LogisticRegression
# ========================
print("\n" + "=" * 50)
print("APPROACH 1: L1-regularized LogisticRegression")
print("=" * 50)

# Use LOOCV for honest evaluation
loo = LeaveOneOut()
y_prob_l1 = np.zeros(len(y))

for train_idx, test_idx in loo.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    model = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=5000, random_state=42)
    model.fit(X_tr_s, y_tr)
    y_prob_l1[test_idx] = model.predict_proba(X_te_s)[:, 1]

auc_l1 = roc_auc_score(y, y_prob_l1)
brier_l1 = brier_score_loss(y, y_prob_l1)
print(f"LOOCV AUC: {auc_l1:.4f}")
print(f"LOOCV Brier: {brier_l1:.4f}")

# ========================
# Approach 2: Ridge (L2) + Feature Selection
# ========================
print("\n" + "=" * 50)
print("APPROACH 2: Ridge + SelectKBest Feature Selection")
print("=" * 50)

# Cross-validated feature selection + Ridge
k_values = [5, 8, 10, 12, 15]
best_k = 8
best_auc = 0

for k in k_values:
    y_prob_k = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        selector = SelectKBest(f_classif, k=k)
        X_tr_sel = selector.fit_transform(X_tr_s, y_tr)
        X_te_sel = selector.transform(X_te_s)
        
        model = LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=42)
        model.fit(X_tr_sel, y_tr)
        y_prob_k[test_idx] = model.predict_proba(X_te_sel)[:, 1]
    
    auc_k = roc_auc_score(y, y_prob_k)
    if auc_k > best_auc:
        best_auc = auc_k
        best_k = k

print(f"Best k: {best_k}, LOOCV AUC: {best_auc:.4f}")

# Refit with best k using all data to show selected features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
selector = SelectKBest(f_classif, k=best_k)
X_sel = selector.fit_transform(X_scaled, y)
selected_mask = selector.get_support()
selected_features = [f for f, sel in zip(feature_names, selected_mask) if sel]
print(f"Selected features ({len(selected_features)}): {selected_features}")

final_model = LogisticRegression(penalty="l2", C=1.0, max_iter=5000, random_state=42)
final_model.fit(X_sel, y)

# ========================
# Approach 3: ElasticNet
# ========================
print("\n" + "=" * 50)
print("APPROACH 3: ElasticNet (L1 + L2)")
print("=" * 50)

y_prob_en = np.zeros(len(y))
for train_idx, test_idx in loo.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr = y[train_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    model = LogisticRegression(penalty="elasticnet", solver="saga",
                                C=0.1, l1_ratio=0.5, max_iter=5000, random_state=42)
    model.fit(X_tr_s, y_tr)
    y_prob_en[test_idx] = model.predict_proba(X_te_s)[:, 1]

auc_en = roc_auc_score(y, y_prob_en)
brier_en = brier_score_loss(y, y_prob_en)
print(f"ElasticNet LOOCV AUC: {auc_en:.4f}")
print(f"ElasticNet LOOCV Brier: {brier_en:.4f}")

# ========================
# Bootstrap Confidence Intervals
# ========================
print("\n" + "=" * 50)
print("BOOTSTRAP CONFIDENCE INTERVALS (ElasticNet)")
print("=" * 50)

n_bootstrap = 1000
bootstrap_aucs = []

for b in range(n_bootstrap):
    idx = np.random.choice(len(y), len(y), replace=True)
    X_boot, y_boot = X[idx], y[idx]
    
    # Leave-pair-out within bootstrap
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=b)
    aucs_fold = []
    for train_idx, test_idx in skf.split(X_boot, y_boot):
        X_tr, X_te = X_boot[train_idx], X_boot[test_idx]
        y_tr, y_te = y_boot[train_idx], y_boot[test_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        model = LogisticRegression(penalty="elasticnet", solver="saga",
                                    C=0.1, l1_ratio=0.5, max_iter=5000)
        model.fit(X_tr_s, y_tr)
        y_prob_b = model.predict_proba(X_te_s)[:, 1]
        if len(np.unique(y_te)) > 1:
            aucs_fold.append(roc_auc_score(y_te, y_prob_b))
    
    if aucs_fold:
        bootstrap_aucs.append(np.mean(aucs_fold))

bootstrap_aucs = np.array(bootstrap_aucs)
ci_lower = np.percentile(bootstrap_aucs, 2.5)
ci_upper = np.percentile(bootstrap_aucs, 97.5)
print(f"Bootstrap AUC: {np.mean(bootstrap_aucs):.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

# ========================
# Probability Calibration
# ========================
print("\n" + "=" * 50)
print("CALIBRATION ANALYSIS")
print("=" * 50)

# Use the best approach (ElasticNet) and calibrate
calibrator = CalibratedClassifierCV(
    LogisticRegression(penalty="elasticnet", solver="saga", C=0.1, l1_ratio=0.5, max_iter=5000),
    cv=5, method="sigmoid"
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_prob_cal = cross_val_predict(calibrator, X_scaled, y, cv=5, method="predict_proba")[:, 1]
brier_cal = brier_score_loss(y, y_prob_cal)
auc_cal = roc_auc_score(y, y_prob_cal)

print(f"After Platt scaling:")
print(f"AUC: {auc_cal:.4f}")
print(f"Brier Score: {brier_cal:.4f}")
print(f"Brier < 0.20: {'PASS' if brier_cal < 0.20 else 'NEEDS IMPROVEMENT'}")

# ========================
# Final Comparison
# ========================
print("\n" + "=" * 50)
print("FINAL COMPARISON")
print("=" * 50)
comparison = pd.DataFrame({
    "Method": ["L1 LogisticRegression", "Ridge + SelectKBest",
               "ElasticNet", "ElasticNet (Calibrated)"],
    "AUC": [f"{auc_l1:.4f}", f"{best_auc:.4f}", f"{auc_en:.4f}", f"{auc_cal:.4f}"],
    "Brier": [f"{brier_l1:.4f}", "-", f"{brier_en:.4f}", f"{brier_cal:.4f}"],
})
print(comparison.to_string(index=False))
print(f"\n95% CI for ElasticNet AUC: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Selected biomarkers: {selected_features}")
