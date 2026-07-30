"""Starter code for Real Estate Price Prediction."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv('data/real_estate.csv')

print("=== Real Estate Dataset ===")
print(f"Shape: {df.shape}")
print(f"\nPrice statistics:\n{df['Price'].describe()}")
print(f"\nCorrelation with Price:\n{df.corr(numeric_only=True)['Price'].sort_values(ascending=False)}")

df = pd.get_dummies(df, columns=['ZipCode'], prefix='Zip', drop_first=True)

feature_cols = [c for c in df.columns if c != 'Price']
X = df[feature_cols]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f"\nR2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):,.0f}")
print(f"MAPE: {np.mean(np.abs((y_test - y_pred) / y_test)) * 100:.2f}%")
