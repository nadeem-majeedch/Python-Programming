"""Reference solution for Real Estate Price Prediction.

Approach:
1. Feature engineering (PricePerSqFt, Age, RoomRatio, log-Price)
2. Handle outliers with IQR clipping
3. PolynomialFeatures for non-linear relationships
4. Ridge/Lasso regularization
5. Compare multiple models: Linear, Ridge, Lasso, RandomForest
6. Residual diagnostics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

df = pd.read_csv('data/real_estate.csv')

print("=" * 60)
print("REAL ESTATE PRICE PREDICTION - SOLUTION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Price range: ${df['Price'].min():,.0f} - ${df['Price'].max():,.0f}")

print("\n=== Feature Engineering ===")
df['Age'] = 2024 - df['YearBuilt']
df['PricePerSqFt'] = df['Price'] / df['SqFt']
df['RoomRatio'] = df['Bedrooms'] / (df['Bathrooms'] + 1e-5)
df['TotalRooms'] = df['Bedrooms'] + df['Bathrooms']
df['LotSqFtRatio'] = df['LotSize'] / df['SqFt']
df['HasAmenities'] = df['HasGarage'] + df['HasPool']
df['LogPrice'] = np.log(df['Price'])

print("Added: Age, PricePerSqFt, RoomRatio, TotalRooms, LotSqFtRatio, HasAmenities, LogPrice")

quartiles = df['Price'].quantile([0.01, 0.99])
print(f"\nPrice 1st percentile: ${quartiles[0.01]:,.0f}")
print(f"Price 99th percentile: ${quartiles[0.99]:,.0f}")

zip_dummies = pd.get_dummies(df['ZipCode'], prefix='Zip', drop_first=True)
feature_cols = ['SqFt', 'Bedrooms', 'Bathrooms', 'Age', 'LotSize', 'HasGarage',
                'HasPool', 'DistanceToDowntown', 'CrimeRate', 'SchoolRating',
                'RoomRatio', 'TotalRooms', 'LotSqFtRatio']
X = pd.concat([df[feature_cols], zip_dummies], axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge(alpha=10)': Ridge(alpha=10, random_state=42),
    'Lasso(alpha=1)': Lasso(alpha=1, random_state=42, max_iter=5000),
    'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
}

results = []
for name, model in models.items():
    if name == 'RandomForest':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    results.append({'Model': name, 'R2': r2, 'RMSE': rmse, 'MAE': mae, 'MAPE': f'{mape:.1f}%'})

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"\n--- {name} ---")
    print(f"R2: {r2:.4f} | RMSE: ${rmse:,.0f} | MAE: ${mae:,.0f} | MAPE: {mape:.1f}%")
    print(f"CV R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

print("\n=== Polynomial Features with Ridge ===")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
ridge_poly = Ridge(alpha=50, random_state=42)
ridge_poly.fit(X_train_poly, y_train)
y_pred_poly = ridge_poly.predict(X_test_poly)
r2_poly = r2_score(y_test, y_pred_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
print(f"Polynomial+Ridge R2: {r2_poly:.4f} | RMSE: ${rmse_poly:,.0f}")

best_model = max(results, key=lambda r: r['R2'])
print(f"\n=== Best Model: {best_model['Model']} ===")
print(f"R2: {best_model['R2']:.4f} | RMSE: ${best_model['RMSE']:,.0f} | MAE: ${best_model['MAE']:,.0f}")
