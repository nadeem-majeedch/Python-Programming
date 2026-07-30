"""Generate synthetic employee attrition dataset."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n = 2000
departments = ['Sales', 'Research & Development', 'Human Resources']
dept_probs = [0.35, 0.50, 0.15]

ages = np.random.randint(22, 60, n)
departments_list = np.random.choice(departments, n, p=dept_probs)
distance = np.random.exponential(10, n).round(1)
education = np.random.choice([1, 2, 3, 4], n, p=[0.15, 0.30, 0.35, 0.20])
job_sat = np.random.choice([1, 2, 3, 4], n, p=[0.12, 0.20, 0.38, 0.30])

income = np.where(departments_list == 'Sales',
                  np.random.lognormal(10.5, 0.4, n),
                  np.where(departments_list == 'Research & Development',
                           np.random.lognormal(10.7, 0.35, n),
                           np.random.lognormal(10.3, 0.3, n))).round(-2)
income = np.clip(income, 20000, 200000).astype(int)

num_companies = np.random.poisson(2.5, n).clip(0, 10)
overtime = np.random.choice(['Yes', 'No'], n, p=[0.28, 0.72])
salary_hike = np.random.uniform(8, 25, n).round(1)
perf_rating = np.random.choice([1, 2, 3, 4], n, p=[0.05, 0.15, 0.55, 0.25])

years_at = np.random.exponential(6, n).round(1).clip(0, 40)
years_since_promo = np.where(years_at < 1, 0, np.random.exponential(3, n).round(1).clip(0, 25))

attrition_prob = (
    0.05 + 0.20 * (job_sat <= 2).astype(int) + 0.15 * (overtime == 'Yes').astype(int)
    + 0.10 * (years_since_promo > 5).astype(int) + 0.05 * (years_at < 2).astype(int)
    - 0.05 * (job_sat >= 4).astype(int) - 0.03 * (perf_rating >= 4).astype(int)
    + np.random.uniform(-0.05, 0.05, n)
)
attrition_prob = np.clip(attrition_prob, 0.05, 0.45)
attrition = ['Yes' if np.random.random() < p else 'No' for p in attrition_prob]

df = pd.DataFrame({
    'Age': ages, 'Department': departments_list, 'DistanceFromHome': distance,
    'Education': education, 'JobSatisfaction': job_sat, 'MonthlyIncome': income,
    'NumCompaniesWorked': num_companies, 'OverTime': overtime,
    'PercentSalaryHike': salary_hike, 'PerformanceRating': perf_rating,
    'YearsAtCompany': years_at, 'YearsSinceLastPromotion': years_since_promo,
    'Attrition': attrition
})

out_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'employee_attrition.csv')
df.to_csv(out_path, index=False)
print(f"Generated {out_path} ({len(df)} rows, {df['Attrition'].value_counts().get('Yes', 0)} attrition cases)")
