#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 2000

data = {
    "CustomerID": [f"CUST_{i:05d}" for i in range(1, n + 1)],
    "Tenure": np.random.randint(1, 73, n),
    "MonthlyCharges": np.round(np.random.uniform(20, 120, n), 2),
    "TotalCharges": np.round(np.random.uniform(100, 9000, n), 2),
    "ContractType": np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]
    ),
    "PaymentMethod": np.random.choice(
        ["Electronic check", "Credit card", "Bank transfer"], n, p=[0.40, 0.30, 0.30]
    ),
    "TechSupport": np.random.choice(
        ["Yes", "No", "No internet service"], n, p=[0.30, 0.40, 0.30]
    ),
    "SeniorCitizen": np.random.choice([0, 1], n, p=[0.80, 0.20]),
}

df = pd.DataFrame(data)

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map = {"Electronic check": 0, "Credit card": 1, "Bank transfer": 2}
tech_map = {"No": 0, "Yes": 1, "No internet service": 2}

score = (
    -0.05 * df["Tenure"]
    + 0.02 * df["MonthlyCharges"]
    - 0.5 * df["SeniorCitizen"]
    + np.where(df["ContractType"].map(contract_map) == 0, 1.0, -0.5)
    + np.where(df["PaymentMethod"].map(payment_map) == 0, 0.3, -0.2)
    + np.where(df["TechSupport"].map(tech_map) == 1, -0.8, 0.2)
    + np.random.normal(0, 0.5, n)
)

df["Churn"] = np.where(score > 0, "Yes", "No")

out = os.path.join(os.getcwd(), "customer_churn.csv")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
