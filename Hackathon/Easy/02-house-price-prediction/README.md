# Problem 02 – House Price Prediction

**Domain**: Regression  
**Goal**: Predict house sale prices based on structural and lot features.

## Objectives
- Build a regression model to predict house prices.
- Engineer new features (Age, PricePerSqFt).
- Compare LinearRegression vs. RidgeCV.

## Success Criteria
- Achieve RMSE < 50,000 on the test set.
- R² score > 0.80.
- Produce a residual plot to evaluate model fit.

## Dataset: `data/house_prices.csv`
~1500 rows with columns:
- SqFt, Bedrooms, Bathrooms, YearBuilt, LotSize, Garage
- Price (target)
