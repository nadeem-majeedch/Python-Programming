"""Reference solution for Weather Forecasting.

Approach:
1. Time series decomposition (trend + seasonal + residual)
2. ARIMA model with auto_arima or manual order selection
3. Baseline: LinearRegression with lag features (7, 14, 21, 28 days)
4. RMSE comparison across models
5. Residual diagnostics (autocorrelation, normality)
6. Forecast visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

try:
    from pmdarima import auto_arima
    HAS_AUTO_ARIMA = True
except ImportError:
    HAS_AUTO_ARIMA = False

df = pd.read_csv('data/weather_data.csv', parse_dates=['Date'], index_col='Date')
df['Precipitation'].fillna(0, inplace=True)

print("=" * 60)
print("WEATHER FORECASTING - SOLUTION")
print("=" * 60)

print(f"\nDate range: {df.index.min()} to {df.index.max()}")
print(f"Total days: {len(df)}")

target = 'Temp_Max'

decomp = seasonal_decompose(df[target].dropna(), model='additive', period=365)
print(f"\nTime Series Decomposition completed")
print(f"Trend range: {decomp.trend.min():.1f} to {decomp.trend.max():.1f}")
print(f"Seasonal range: {decomp.seasonal.min():.1f} to {decomp.seasonal.max():.1f}")
print(f"Residual std: {decomp.resid.std():.2f}")

def create_lag_features(series, n_lags=7):
    df_lag = pd.DataFrame({'target': series})
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        df_lag[f'lag_{lag}'] = series.shift(lag)
    df_lag['rolling_mean_7'] = series.rolling(7).mean()
    df_lag['rolling_std_7'] = series.rolling(7).std()
    df_lag['day_of_year'] = series.index.dayofyear
    return df_lag.dropna()

print("\n=== LinearRegression with Lag Features ===")
df_lag = create_lag_features(df[target])
feature_cols = [c for c in df_lag.columns if c != 'target']
X = df_lag[feature_cols].values
y = df_lag['target'].values

split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
mae_lr = mean_absolute_error(y_test, y_pred_lr)
print(f"RMSE: {rmse_lr:.2f} | MAE: {mae_lr:.2f}")

print("\n=== ARIMA Model ===")
train_arima = df[target].iloc[:800]
test_arima = df[target].iloc[800:]

if HAS_AUTO_ARIMA:
    print("Using auto_arima for order selection...")
    auto_model = auto_arima(train_arima, seasonal=True, m=365,
                            start_p=0, max_p=5, start_q=0, max_q=5,
                            d=1, trace=False, error_action='ignore',
                            suppress_warnings=True)
    order = auto_model.order
    print(f"Selected ARIMA order: {order}")
else:
    plot_pacf(train_arima.diff().dropna(), lags=20)
    plt.title('PACF - helps determine p')
    order = (2, 0, 2)
    print(f"Using manual ARIMA order: {order}")

arima_model = ARIMA(train_arima, order=order)
arima_fit = arima_model.fit()
print(f"AIC: {arima_fit.aic:.2f}")
print(f"BIC: {arima_fit.bic:.2f}")

forecast = arima_fit.forecast(steps=len(test_arima))
rmse_arima = np.sqrt(mean_squared_error(test_arima, forecast))
mae_arima = mean_absolute_error(test_arima, forecast)
print(f"RMSE: {rmse_arima:.2f} | MAE: {mae_arima:.2f}")

residuals = arima_fit.resid
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(f"\nResidual Ljung-Box p-value: {lb_test['lb_pvalue'].values[0]:.4f}")

print(f"\n=== Model Comparison ===")
print(f"{'Model':<30} {'RMSE':<10} {'MAE':<10}")
print(f"{'LinearRegression (lags)':<30} {rmse_lr:<10.2f} {mae_lr:<10.2f}")
print(f"{'ARIMA' + str(order):<30} {rmse_arima:<10.2f} {mae_arima:<10.2f}")
