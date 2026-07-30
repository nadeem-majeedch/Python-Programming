# Problem 01 – Customer Churn Prediction

**Domain**: Classification  
**Goal**: Predict whether a customer will churn (Yes/No) based on account and demographic information.

## Objectives
- Build a classification model to predict customer churn.
- Compare Logistic Regression vs. Random Forest performance.
- Evaluate models using accuracy, precision, recall, and F1-score.

## Success Criteria
- Achieve accuracy > 75% on the test set.
- Identify top 3 most important features influencing churn.
- Generate a confusion matrix and classification report.

## Dataset: `data/customer_churn.csv`
~2000 rows with columns:
- CustomerID, Tenure, MonthlyCharges, TotalCharges
- ContractType (Month-to-month / One year / Two year)
- PaymentMethod (Electronic check / Credit card / Bank transfer)
- TechSupport (Yes / No / No internet service)
- SeniorCitizen (0/1)
- Churn (Yes/No – target)
