# Real Estate Price Prediction

## Problem Statement
Predict residential property prices based on property features and location characteristics.

## Dataset
Generated synthetic dataset with 3000 properties:
- **Property features**: SqFt, Bedrooms, Bathrooms, YearBuilt, LotSize, HasGarage, HasPool
- **Location features**: ZipCode (5 codes), DistanceToDowntown, CrimeRate, SchoolRating
- **Target**: Price (complex formula with interactions and non-linear terms)

## Objectives
1. Engineer meaningful features (PricePerSqFt, Age, RoomRatio)
2. Handle outliers in price and feature distributions
3. Apply PolynomialFeatures and regularization (Ridge/Lasso)
4. Compare multiple regression models
5. Perform residual diagnostics
6. Incorporate geospatial features

## Success Criteria
- R2 Score > 0.80
- RMSE < 20% of mean price
- Demonstrate proper residual analysis

## Evaluation Metrics
- R2 Score, RMSE, MAE, MAPE
- Residual plots
- Feature coefficients analysis
- Cross-validation scores
