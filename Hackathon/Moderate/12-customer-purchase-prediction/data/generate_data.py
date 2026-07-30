"""Generate synthetic customer purchase prediction dataset."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 5000

ages = np.random.randint(18, 70, n)
genders = np.random.choice(['Male', 'Female'], n)
income = np.random.lognormal(10.8, 0.5, n).round(-3).clip(15000, 250000).astype(int)
site_visits = np.random.poisson(5, n).clip(1, 50)
time_on_site = np.random.exponential(8, n).round(1).clip(0.5, 60)
pages_visited = np.random.poisson(4, n).clip(1, 30)
prev_purchases = np.random.poisson(2, n).clip(0, 30)
email_opened = np.random.choice(['Yes', 'No'], n, p=[0.4, 0.6])
ad_clicked = np.random.choice(['Yes', 'No'], n, p=[0.15, 0.85])
discount_used = np.random.choice(['Yes', 'No'], n, p=[0.2, 0.8])

purchase_prob = (
    0.05 + 0.15 * (email_opened == 'Yes').astype(int)
    + 0.08 * (ad_clicked == 'Yes').astype(int)
    + 0.05 * (time_on_site > 10).astype(int)
    + 0.04 * (pages_visited > 5).astype(int)
    + 0.03 * (prev_purchases > 3).astype(int)
    - 0.02 * (ages < 25).astype(int)
    + 0.03 * (income > 70000).astype(int)
    + np.random.uniform(-0.05, 0.05, n)
)
purchase_prob = np.clip(purchase_prob, 0.05, 0.50)
made_purchase = ['Yes' if np.random.random() < p else 'No' for p in purchase_prob]

df = pd.DataFrame({
    'CustomerID': [f'C{i+1:04d}' for i in range(n)],
    'Age': ages, 'Gender': genders, 'Income': income,
    'WebsiteVisits': site_visits, 'TimeOnSite': time_on_site,
    'PagesVisited': pages_visited, 'PreviousPurchases': prev_purchases,
    'EmailOpened': email_opened, 'AdClicked': ad_clicked,
    'DiscountUsed': discount_used, 'MadePurchase': made_purchase
})

out_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'customer_purchases.csv')
df.to_csv(out_path, index=False)
purchase_rate = df['MadePurchase'].value_counts(normalize=True).get('Yes', 0)
print(f"Generated {out_path} ({len(df)} rows, purchase rate: {purchase_rate:.1%})")
