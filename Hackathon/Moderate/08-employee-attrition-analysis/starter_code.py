"""Starter code for Employee Attrition Analysis."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv('data/employee_attrition.csv')

print("=== Employee Attrition Dataset ===")
print(f"Shape: {df.shape}")
print(f"\nAttrition distribution:\n{df['Attrition'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nBasic statistics:\n{df.describe()}")

le = LabelEncoder()
df['Attrition_encoded'] = le.fit_transform(df['Attrition'])
df['OverTime_encoded'] = le.fit_transform(df['OverTime'])
df = pd.get_dummies(df, columns=['Department'], drop_first=True)

feature_cols = [c for c in df.columns if c not in ['Attrition', 'Attrition_encoded']]
X = df[feature_cols]
y = df['Attrition_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
