# Weather Forecasting

## Problem Statement
Forecast daily weather metrics using time series analysis techniques.

## Dataset
Generated synthetic dataset with 3 years of daily weather data (1095 rows):
- **Date**: Daily timestamp (2021-2023)
- **Temp_Max, Temp_Min**: Temperature extremes with seasonal patterns
- **Humidity**: Daily average with seasonal variation
- **WindSpeed**: Wind speed with lognormal distribution
- **Precipitation**: Rainfall (sparse, ~3% missing values)
- **Pressure**: Atmospheric pressure with seasonal patterns

## Objectives
1. Plot and analyze time series (trend, seasonality, residuals)
2. Decompose time series into components
3. Build ARIMA model (select p,d,q orders)
4. Use lag features with LinearRegression as a baseline
5. Compare forecast accuracy across models
6. Perform residual diagnostics

## Success Criteria
- RMSE < 5 degrees for temperature forecasting
- Residuals are approximately white noise
- Forecast visualization matches actual patterns

## Evaluation Metrics
- RMSE, MAE, MAPE
- AIC/BIC for ARIMA selection
- Residual autocorrelation (Ljung-Box test)
- Forecast vs actual visualization
