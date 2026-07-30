# Employee Attrition Analysis

## Problem Statement
Predict which employees are likely to leave the company and identify key drivers of attrition.

## Dataset
Generated synthetic dataset with 2000 employees:
- **Demographics**: Age, Department, DistanceFromHome, Education
- **Job factors**: JobSatisfaction, MonthlyIncome, NumCompaniesWorked, OverTime, PercentSalaryHike, PerformanceRating
- **Tenure**: YearsAtCompany, YearsSinceLastPromotion
- **Target**: Attrition (Yes/No, ~16% attrition rate)

## Objectives
1. Perform comprehensive EDA and visualize key patterns
2. Engineer meaningful features (income-per-year ratio, tenure ratios)
3. Handle class imbalance with SMOTE
4. Train and compare RandomForest and XGBoost classifiers
5. Analyze feature importance and interpret attrition drivers

## Success Criteria
- ROC-AUC > 0.80
- Identify top 5 attrition drivers
- Provide actionable business recommendations

## Evaluation Metrics
- ROC-AUC, Precision, Recall, F1-Score
- Feature Importance ranking
- Confusion Matrix
- Business insights and recommendations
