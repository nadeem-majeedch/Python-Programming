"""Reference solution for Credit Card Fraud Detection.

Approach:
1. Handle missing values with median imputation
2. Scale features with StandardScaler
3. Apply SMOTE to handle class imbalance
4. Train multiple models: LogisticRegression, RandomForest, IsolationForest
5. Tune decision threshold for optimal F1 score
6. Evaluate with precision-recall curves and cost-benefit analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, average_precision_score, f1_score,
    roc_curve, auc
)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

df = pd.read_csv('data/credit_card_fraud.csv')

print("=" * 60)
print("CREDIT CARD FRAUD DETECTION - SOLUTION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Fraud rate: {df['Class'].mean():.4f} ({df['Class'].sum()} cases)")

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))
X_test_processed = scaler.transform(imputer.transform(X_test))

smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_res, y_train_res = smote.fit_resample(X_train_processed, y_train)

print(f"\nAfter SMOTE - Train shape: {X_train_res.shape}")
print(f"Class distribution after SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1),
}

results = []
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    y_pred = model.predict(X_test_processed)
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_pr = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    results.append({'Model': name, 'ROC-AUC': roc_auc, 'PR-AUC': avg_pr, 'F1': f1})
    print(f"\n--- {name} ---")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {avg_pr:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = f1_scores.max()
    print(f"Best threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")

    y_pred_tuned = (y_proba >= best_threshold).astype(int)
    print(f"Tuned Confusion Matrix:\n{confusion_matrix(y_test, y_pred_tuned)}")
    print(f"Tuned F1: {f1_score(y_test, y_pred_tuned):.4f}")

    fp_cost = 25
    fn_cost = 250
    base_cost = y_pred_tuned.sum() * fp_cost + (y_test.sum() - (y_pred_tuned & y_test).sum()) * fn_cost
    no_model_cost = y_test.sum() * fn_cost
    savings = no_model_cost - base_cost
    print(f"Cost analysis (FP=${fp_cost}, FN=${fn_cost}):")
    print(f"  No-model cost: ${no_model_cost}")
    print(f"  Model cost: ${base_cost}")
    print(f"  Savings: ${savings}")

iso_forest = IsolationForest(random_state=42, contamination=0.03, n_estimators=200)
iso_pred = iso_forest.fit_predict(X_train_processed)
iso_pred_test = iso_forest.predict(X_test_processed)
iso_pred_test = np.where(iso_pred_test == -1, 1, 0)
print(f"\n--- IsolationForest (Anomaly Detection) ---")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, iso_pred_test)}")
print(f"Classification Report:\n{classification_report(y_test, iso_pred_test)}")

results_df = pd.DataFrame(results)
print(f"\n=== Summary ===")
print(results_df.to_string(index=False))
