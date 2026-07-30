import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/house_prices.csv")

# Feature engineering
df["Age"] = 2023 - df["YearBuilt"]
df["PricePerSqFt"] = df["Price"] / df["SqFt"]
df["Rooms"] = df["Bedrooms"] + df["Bathrooms"]

feature_cols = ["SqFt", "Bedrooms", "Bathrooms", "LotSize", "Garage", "Age", "Rooms"]

X = df[feature_cols]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("=== Linear Regression ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
print(f"R2:   {r2_score(y_test, y_pred_lr):.4f}")

# RidgeCV
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

print("\n=== Ridge Regression ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.2f}")
print(f"R2:   {r2_score(y_test, y_pred_ridge):.4f}")
print(f"Best alpha: {ridge.alpha_}")

# Coefficients
print("\n=== Coefficients ===")
for col, coef in zip(feature_cols, ridge.coef_):
    print(f"{col}: {coef:.4f}")

# Residual plot
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_ridge, y_test - y_pred_ridge, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("residual_plot.png")
print("\nSaved residual_plot.png")
