"""Starter code for Customer Purchase Prediction."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv('data/customer_purchases.csv')

print("=== Customer Purchase Dataset ===")
print(f"Shape: {df.shape}")
print(f"\nPurchase distribution:\n{df['MadePurchase'].value_counts(normalize=True)}")
print(f"\nCorrelation with MadePurchase:")
for col in df.select_dtypes(include=[np.number]).columns:
    if col != 'CustomerID':
        corr = df[col].corr(pd.factorize(df['MadePurchase'])[0])
        if abs(corr) > 0.05:
            print(f"  {col}: {corr:.3f}")

le = LabelEncoder()
df['MadePurchase_enc'] = le.fit_transform(df['MadePurchase'])
df['Gender_enc'] = le.fit_transform(df['Gender'])
df['EmailOpened_enc'] = le.fit_transform(df['EmailOpened'])
df['AdClicked_enc'] = le.fit_transform(df['AdClicked'])
df['DiscountUsed_enc'] = le.fit_transform(df['DiscountUsed'])

feature_cols = ['Age', 'Income', 'WebsiteVisits', 'TimeOnSite', 'PagesVisited',
                'PreviousPurchases', 'Gender_enc', 'EmailOpened_enc', 'AdClicked_enc', 'DiscountUsed_enc']
X = df[feature_cols]
y = df['MadePurchase_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(f"\nROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
