"""Reference solution for Employee Attrition Analysis.

Approach:
1. Comprehensive EDA with visualizations
2. Feature engineering (IncomePerYear, Tenure ratios, Engagement score)
3. SMOTE for class imbalance
4. RandomForest + XGBoost with hyperparameter tuning
5. Feature importance analysis and business interpretation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score
)
from imblearn.over_sampling import SMOTE

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

df = pd.read_csv('data/employee_attrition.csv')

print("=" * 60)
print("EMPLOYEE ATTRITION ANALYSIS - SOLUTION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Attrition rate: {df['Attrition'].value_counts(normalize=True).get('Yes', 0):.2%}")

print("\n=== EDA ===")
print(f"\nAttrition by Department:\n{df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack()}")
print(f"\nAttrition by Overtime:\n{df.groupby('OverTime')['Attrition'].value_counts(normalize=True).unstack()}")

print("\n=== Feature Engineering ===")
df['IncomePerYear'] = df['MonthlyIncome'] * 12
df['IncomePerYearAtCompany'] = df['IncomePerYear'] / (df['YearsAtCompany'] + 1)
df['TenureRatio'] = df['YearsAtCompany'] / (df['Age'] + 1)
df['PromotionLagRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
df['EngagementScore'] = (
    df['JobSatisfaction'] * 0.3 + (df['PerformanceRating'] / 4.0) * 0.3
    + (df['PercentSalaryHike'] / 25.0) * 0.2 + (1 - df['YearsSinceLastPromotion'] / df['YearsAtCompany'].clip(upper=10)) * 0.2
)

df['Attrition_encoded'] = LabelEncoder().fit_transform(df['Attrition'])
df['OverTime_encoded'] = LabelEncoder().fit_transform(df['OverTime'])
df = pd.get_dummies(df, columns=['Department'], drop_first=True)

feature_cols = [c for c in df.columns if c not in ['Attrition', 'Attrition_encoded']]
X = df[feature_cols]
y = df['Attrition_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

smote = SMOTE(random_state=42, sampling_strategy=0.8)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Positive class: {y_train_res.sum()} / {len(y_train_res)} ({y_train_res.mean():.1%})")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [8, 12, None],
    'min_samples_split': [5, 10]
}
rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train_res)
rf_best = rf_grid.best_estimator_

print(f"\nBest RF params: {rf_grid.best_params_}")

y_proba_rf = rf_best.predict_proba(X_test_scaled)[:, 1]
y_pred_rf = rf_best.predict(X_test_scaled)

print(f"\n--- RandomForest ---")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_rf)}")

if HAS_XGB:
    xgb = XGBClassifier(random_state=42, eval_metric='logloss', scale_pos_weight=len(y_train_res[y_train_res==0])/len(y_train_res[y_train_res==1]))
    xgb_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1]
    }
    xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    xgb_grid.fit(X_train_scaled, y_train_res)
    xgb_best = xgb_grid.best_estimator_
    y_proba_xgb = xgb_best.predict_proba(X_test_scaled)[:, 1]
    y_pred_xgb = xgb_best.predict(X_test_scaled)

    print(f"\n--- XGBoost ---")
    print(f"Best params: {xgb_grid.best_params_}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_xgb):.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_xgb)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred_xgb)}")

print("\n=== Feature Importance (RandomForest) ===")
importances = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_best.feature_importances_
}).sort_values('importance', ascending=False)

print(importances.head(10).to_string(index=False))
print("\nTop 5 attrition drivers identified successfully!")
