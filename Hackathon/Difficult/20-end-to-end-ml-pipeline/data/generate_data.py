#!/usr/bin/env python3
"""Generate synthetic production pipeline data with drift."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_normal = 6000
n_drift = 2000
n_total = n_normal + n_drift

# Normal data
records = []
for _ in range(n_normal):
    age = np.random.normal(40, 12)
    income = np.random.lognormal(4.5, 0.6)
    credit_score = np.random.normal(700, 50)
    account_age = np.random.exponential(5)
    num_transactions = np.random.poisson(15)
    avg_transaction = np.random.lognormal(3, 0.8)
    customer_tier = np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], p=[0.4, 0.3, 0.2, 0.1])
    region = np.random.choice(["North", "South", "East", "West"], p=[0.25, 0.25, 0.25, 0.25])
    feedback = np.random.choice(["Good", "Average", "Poor"], p=[0.5, 0.3, 0.2])
    signup_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=np.random.randint(0, 1500))
    
    # Target
    p = 1 / (1 + np.exp(-(-2 + 0.01 * income + 0.003 * credit_score +
                           0.1 * account_age - 0.02 * age +
                           {"Bronze": -0.5, "Silver": 0, "Gold": 0.5, "Platinum": 1.0}[customer_tier])))
    target = int(np.random.random() < p)
    
    # Missing values
    if np.random.random() < 0.03:
        income = np.nan
    if np.random.random() < 0.02:
        credit_score = np.nan
    if np.random.random() < 0.01:
        age = np.nan
    
    records.append({
        "Age": round(age, 1),
        "Income": round(income, 2) if not np.isnan(income) else np.nan,
        "CreditScore": round(credit_score, 1) if not np.isnan(credit_score) else np.nan,
        "AccountAgeMonths": round(account_age, 1),
        "NumTransactions": num_transactions,
        "AvgTransactionAmount": round(avg_transaction, 2),
        "CustomerTier": customer_tier,
        "Region": region,
        "Feedback": feedback,
        "SignupDate": signup_date,
        "Target": target
    })

# Drifted data (different distributions)
for _ in range(n_drift):
    age = np.random.normal(55, 15)  # Older
    income = np.random.lognormal(4.0, 0.5)  # Lower income
    credit_score = np.random.normal(650, 60)  # Lower credit
    account_age = np.random.exponential(3)  # Newer accounts
    num_transactions = np.random.poisson(8)  # Fewer transactions
    avg_transaction = np.random.lognormal(2.5, 0.6)  # Lower transaction amounts
    customer_tier = np.random.choice(["Bronze", "Silver", "Gold", "Platinum"], p=[0.5, 0.3, 0.15, 0.05])
    region = np.random.choice(["North", "South", "East", "West"], p=[0.4, 0.2, 0.3, 0.1])  # Different region dist
    feedback = np.random.choice(["Good", "Average", "Poor"], p=[0.3, 0.4, 0.3])  # More negative
    signup_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=np.random.randint(0, 800))  # Newer
    
    # Different target generation (slightly lower)
    p = 1 / (1 + np.exp(-(-2.5 + 0.008 * income + 0.002 * credit_score +
                           0.05 * account_age - 0.01 * age +
                           {"Bronze": -0.3, "Silver": 0.1, "Gold": 0.3, "Platinum": 0.6}[customer_tier])))
    target = int(np.random.random() < p)
    
    # Missing values (higher rate in drift)
    if np.random.random() < 0.05:
        income = np.nan
    if np.random.random() < 0.04:
        credit_score = np.nan
    if np.random.random() < 0.02:
        age = np.nan
    
    records.append({
        "Age": round(age, 1) if not np.isnan(age) else np.nan,
        "Income": round(income, 2) if not np.isnan(income) else np.nan,
        "CreditScore": round(credit_score, 1) if not np.isnan(credit_score) else np.nan,
        "AccountAgeMonths": round(account_age, 1),
        "NumTransactions": num_transactions,
        "AvgTransactionAmount": round(avg_transaction, 2),
        "CustomerTier": customer_tier,
        "Region": region,
        "Feedback": feedback,
        "SignupDate": signup_date,
        "Target": target
    })

df = pd.DataFrame(records)
df["IsDrifted"] = [0] * n_normal + [1] * n_drift

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "production_pipeline_data.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} records (drifted: {n_drift}) -> {out_path}")
print(f"Normal target rate: {df[df['IsDrifted']==0]['Target'].mean():.3f}")
print(f"Drifted target rate: {df[df['IsDrifted']==1]['Target'].mean():.3f}")
