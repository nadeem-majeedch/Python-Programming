"""Reference solution for Customer Purchase Prediction.

Approach:
1. Feature engineering (VisitToPurchase ratio, engagement score)
2. Encode categorical variables
3. RFE + feature selection
4. RandomForest + GradientBoosting with hyperparameter tuning
5. ROC curve and threshold optimization
6. Business-focused threshold selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score, roc_curve
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

df = pd.read_csv('data/customer_purchases.csv')

print("=" * 60)
print("CUSTOMER PURCHASE PREDICTION - SOLUTION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
purchase_rate = df['MadePurchase'].value_counts(normalize=True).get('Yes', 0)
print(f"Purchase rate: {purchase_rate:.1%}")

print("\n=== Feature Engineering ===")
df['VisitToPurchaseRatio'] = df['PreviousPurchases'] / (df['WebsiteVisits'] + 1)
df['PagesPerVisit'] = df['PagesVisited'] / (df['WebsiteVisits'] + 1)
df['TimePerPage'] = df['TimeOnSite'] / (df['PagesVisited'] + 1)
df['EngagementScore'] = (
    df['TimeOnSite'] / 60 * 0.3 + df['PagesVisited'] / 30 * 0.3
    + df['PreviousPurchases'] / 20 * 0.2 + (df['WebsiteVisits'] / 50) * 0.2
)
df['IncomePerAge'] = df['Income'] / df['Age']
df['IsHighValue'] = (df['Income'] > 70000).astype(int)

le = LabelEncoder()
df['MadePurchase_enc'] = le.fit_transform(df['MadePurchase'])
df['Gender_enc'] = le.fit_transform(df['Gender'])
df['EmailOpened_enc'] = le.fit_transform(df['EmailOpened'])
df['AdClicked_enc'] = le.fit_transform(df['AdClicked'])
df['DiscountUsed_enc'] = le.fit_transform(df['DiscountUsed'])

feature_cols = ['Age', 'Income', 'WebsiteVisits', 'TimeOnSite', 'PagesVisited',
                'PreviousPurchases', 'Gender_enc', 'EmailOpened_enc', 'AdClicked_enc',
                'DiscountUsed_enc', 'VisitToPurchaseRatio', 'PagesPerVisit',
                'TimePerPage', 'EngagementScore', 'IncomePerAge', 'IsHighValue']
X = df[feature_cols]
y = df['MadePurchase_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n=== Feature Selection (RFE) ===")
rf_selector = RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1)
rfe = RFE(rf_selector, n_features_to_select=10)
rfe.fit(X_train, y_train)
selected_features = [f for f, s in zip(feature_cols, rfe.support_) if s]
print(f"Selected features: {selected_features}")

X_train_sel = X_train_scaled[:, rfe.support_]
X_test_sel = X_test_scaled[:, rfe.support_]

models = {
    'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(random_state=42, n_estimators=200),
}

if HAS_XGB:
    models['XGBoost'] = XGBClassifier(
        random_state=42, eval_metric='logloss',
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
    )

rf_params = {'n_estimators': [100, 200], 'max_depth': [6, 10], 'min_samples_split': [5, 10]}
gb_params = {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.05, 0.1]}

results = []
for name, model in models.items():
    params = rf_params if name == 'RandomForest' else (gb_params if name == 'GradientBoosting' else {'n_estimators': [100, 200], 'max_depth': [4, 6]})
    grid = GridSearchCV(model, params, cv=3, scoring='roc_auc', n_jobs=-1)
    grid.fit(X_train_sel, y_train)
    best = grid.best_estimator_
    y_proba = best.predict_proba(X_test_sel)[:, 1]
    y_pred = best.predict(X_test_sel)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    results.append({'Model': name, 'ROC-AUC': roc_auc, 'F1': f1})
    print(f"\n--- {name} (best params: {grid.best_params_}) ---")
    print(f"ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    best_idx = np.argmax(2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10))
    best_t = thresholds[best_idx]
    print(f"Optimal threshold for F1: {best_t:.3f}")

results_df = pd.DataFrame(results)
print(f"\n=== Summary ===")
print(results_df.to_string(index=False))
