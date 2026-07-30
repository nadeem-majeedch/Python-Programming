#!/usr/bin/env python3
"""Unified data generator for all 7 Moderate-level hackathon scenarios."""

import numpy as np
import pandas as pd
import random
import os
import string
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

BASE = os.path.dirname(os.path.abspath(__file__))


def problem_07_credit_card_fraud():
    print("  Generating credit card fraud data...")
    n = 5000
    fraud_ratio = 0.03
    n_fraud = int(n * fraud_ratio)
    n_legit = n - n_fraud
    classes = [0] * n_legit + [1] * n_fraud
    random.shuffle(classes)

    time_legit = np.random.exponential(72, n_legit)
    time_fraud = np.random.exponential(24, n_fraud)
    times = np.concatenate([time_legit, time_fraud])
    times = np.clip(times, 0, 168).round(2)

    V = np.random.randn(n, 28) * 0.8
    fraud_idx = [i for i, c in enumerate(classes) if c == 1]
    for i in fraud_idx:
        V[i, :8] += np.random.uniform(-3, 3, 8)
        V[i, 8:15] += np.random.uniform(-1.5, 1.5, 7)

    amount_legit = np.random.lognormal(3.5, 0.8, n_legit)
    amount_fraud = np.random.lognormal(4.2, 1.2, n_fraud)
    amounts = np.concatenate([amount_legit, amount_fraud]).round(2)

    df = pd.DataFrame(np.column_stack([times] + [V[:, i] for i in range(28)] + [amounts, classes]),
                      columns=['Time'] + [f'V{i+1}' for i in range(28)] + ['Amount', 'Class'])

    for col in [f'V{i+1}' for i in range(28)]:
        mask = np.random.random(n) < 0.025
        df.loc[mask, col] = np.nan

    out = os.path.join(BASE, 'Moderate', '07-credit-card-fraud-detection', 'data', 'credit_card_fraud.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"    -> {out} ({len(df)} rows)")


def problem_08_employee_attrition():
    print("  Generating employee attrition data...")
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
        0.05
        + 0.20 * (job_sat <= 2).astype(int)
        + 0.15 * (overtime == 'Yes').astype(int)
        + 0.10 * (years_since_promo > 5).astype(int)
        + 0.05 * (years_at < 2).astype(int)
        - 0.05 * (job_sat >= 4).astype(int)
        - 0.03 * (perf_rating >= 4).astype(int)
        + np.random.uniform(-0.05, 0.05, n)
    )
    attrition_prob = np.clip(attrition_prob, 0.05, 0.45)
    attrition = ['Yes' if np.random.random() < p else 'No' for p in attrition_prob]
    attrition = np.array(attrition)

    df = pd.DataFrame({
        'Age': ages, 'Department': departments_list, 'DistanceFromHome': distance,
        'Education': education, 'JobSatisfaction': job_sat, 'MonthlyIncome': income,
        'NumCompaniesWorked': num_companies, 'OverTime': overtime,
        'PercentSalaryHike': salary_hike, 'PerformanceRating': perf_rating,
        'YearsAtCompany': years_at, 'YearsSinceLastPromotion': years_since_promo,
        'Attrition': attrition
    })

    out = os.path.join(BASE, 'Moderate', '08-employee-attrition-analysis', 'data', 'employee_attrition.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"    -> {out} ({len(df)} rows, ~{df['Attrition'].value_counts().get('Yes', 0)/n*100:.1f}% attrition)")


def problem_09_real_estate():
    print("  Generating real estate data...")
    n = 3000
    sqft = np.random.lognormal(7.5, 0.35, n).round(-1).clip(500, 10000).astype(int)
    bedrooms = np.random.choice([1, 2, 3, 4, 5, 6], n, p=[0.02, 0.10, 0.35, 0.33, 0.15, 0.05])
    bathrooms = np.where(bedrooms == 1, np.random.choice([1], n),
                         np.where(bedrooms <= 3, np.random.choice([1, 1.5, 2], n, p=[0.05, 0.10, 0.85]),
                                  np.random.choice([2, 2.5, 3, 4], n, p=[0.30, 0.30, 0.30, 0.10])))
    year_built = np.random.randint(1950, 2024, n)
    lot_size = np.random.lognormal(8.5, 0.5, n).round(-2).clip(1500, 50000).astype(int)
    has_garage = np.random.choice([0, 1], n, p=[0.15, 0.85])
    has_pool = np.random.choice([0, 1], n, p=[0.88, 0.12])
    zip_codes = np.random.choice([85001, 85002, 85003, 85004, 85005], n)
    dist_downtown = np.random.lognormal(2.0, 0.8, n).round(1).clip(0.5, 50)
    crime_rate = np.random.exponential(8, n).round(1).clip(0.5, 50)
    school_rating = np.random.uniform(1, 10, n).round(1)

    age = 2024 - year_built
    base_price = (
        50000
        + sqft * 120
        + bedrooms * 15000
        + bathrooms * 8000
        + (lot_size * 0.5)
        + has_garage * 15000
        + has_pool * 25000
        + np.where(zip_codes == 85001, 30000, np.where(zip_codes == 85003, -10000, 0))
        + (10 - school_rating) * (-5000)
        - dist_downtown * 2000
        - crime_rate * 1000
        + np.where(age > 50, age * (-200), np.where(age < 10, age * 200, 0))
        + np.random.normal(0, 25000, n)
    )
    price = np.maximum(base_price, 30000).round(-3).astype(int)

    df = pd.DataFrame({
        'SqFt': sqft, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms,
        'YearBuilt': year_built, 'LotSize': lot_size, 'HasGarage': has_garage,
        'HasPool': has_pool, 'ZipCode': zip_codes, 'DistanceToDowntown': dist_downtown,
        'CrimeRate': crime_rate, 'SchoolRating': school_rating, 'Price': price
    })

    out = os.path.join(BASE, 'Moderate', '09-real-estate-price-prediction', 'data', 'real_estate.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"    -> {out} ({len(df)} rows)")


def problem_10_sentiment():
    print("  Generating product review data...")

    positive_templates = [
        "Absolutely love this {}! It exceeded all my expectations.",
        "This {} is fantastic. Best purchase I have made all year.",
        "I am so happy with my {} purchase. Works perfectly and looks great.",
        "Great {} for the price. Highly recommend to anyone looking for a quality product.",
        "Five stars! This {} is amazing and arrived quickly.",
        "My {} is wonderful. The quality is outstanding for the price point.",
        "I cannot believe how good this {} is. Totally worth every penny.",
        "Excellent {}! Easy to use, well designed, and durable.",
        "Very satisfied with this {}. Exactly what I needed and more.",
        "This {} is a game changer. So glad I decided to buy it.",
        "Perfect {}! It does everything I wanted and more.",
        "Best {} I have ever owned. The quality is top notch.",
        "I am thrilled with my {}. Shipping was fast and packaging was great.",
        "This {} works like a charm. Easy setup and intuitive controls.",
        "Amazing quality {} for such a reasonable price. Will buy again.",
    ]

    neutral_templates = [
        "The {} is okay. It does the job but nothing special.",
        "Decent {} for the money. Not great but not terrible either.",
        "I have mixed feelings about this {}. It has some good features but also some drawbacks.",
        "Average {} performs as expected. No complaints but no surprises.",
        "This {} works fine. It is not the best but it gets the job done.",
        "The {} arrived on time and was as described. Nothing more to say really.",
        "It is an okay {}. I might look for something better in the future.",
        "Not bad but not great either. The {} serves its purpose.",
        "This {} is alright. I have seen better but I have also seen worse.",
        "My {} is functional. It could use some improvements in design though.",
        "The {} does what it is supposed to do. That is about it.",
        "Pretty standard {}. Nothing really stands out about it.",
    ]

    negative_templates = [
        "Terrible {}! It broke within the first week of use.",
        "Very disappointed with this {}. The quality is just not there.",
        "Do not waste your money on this {}. It is poorly made and does not work well.",
        "This {} is a piece of junk. Cheap materials and bad design.",
        "I regret buying this {}. It stopped working after just a month.",
        "Worst {} I have ever purchased. Totally useless product.",
        "The {} is horrible. Customer service was no help either.",
        "I hate this {}. It is nothing like the description said it would be.",
        "Stay away from this {}! It is overpriced and underperforms.",
        "This is the worst {} ever. Save your money and buy something else.",
        "Completely dissatisfied with this {}. Do not recommend it to anyone.",
        "What a waste of money this {} turned out to be. Very poor quality.",
        "Awful {} with terrible build quality. I want my money back.",
        "This {} does not work properly at all. Extremely frustrating experience.",
        "The {} is defective. Tried to get a replacement but no luck.",
    ]

    products = ['wireless headphones', 'smartphone case', 'bluetooth speaker', 'laptop backpack',
                'coffee maker', 'yoga mat', 'running shoes', 'tablet stand',
                'phone charger', 'water bottle', 'desk lamp', 'meal prep containers',
                'portable fan', 'mouse pad', 'screen protector']

    misspellings = {'absolutely': 'absolutly', 'amazing': 'amazng', 'fantastic': 'fantaastic',
                    'disappointed': 'dissapointed', 'recommend': 'reccomend', 'purchase': 'purchace',
                    'quality': 'quallity', 'defective': 'defektive', 'experience': 'experiance',
                    'satisfied': 'satisified', 'wonderful': 'wunderful', 'terrible': 'terible',
                    'features': 'feautures', 'performance': 'perfomance', 'bought': 'brought'}

    n = 2000
    sentiments = np.random.choice(['Positive', 'Neutral', 'Negative'], n, p=[0.45, 0.30, 0.25])
    reviews = []
    sentiments_list = []
    review_ids = []

    for i in range(n):
        product = random.choice(products)
        sent = sentiments[i]
        if sent == 'Positive':
            template = random.choice(positive_templates)
        elif sent == 'Neutral':
            template = random.choice(neutral_templates)
        else:
            template = random.choice(negative_templates)

        review = template.format(product)

        if random.random() < 0.15:
            wrong_word, correct_word = random.choice(list(misspellings.items()))
            if correct_word in review.lower():
                review = review.lower().replace(correct_word, wrong_word, 1)
                review = review[0].upper() + review[1:] if review else review

        if random.random() < 0.08:
            review = review[:-1] + '!!' if review.endswith('.') else review + '!!'

        reviews.append(review)
        sentiments_list.append(sent)
        review_ids.append(f"R{1000+i:04d}")

    df = pd.DataFrame({'review_id': review_ids, 'review_text': reviews, 'sentiment': sentiments_list})

    out = os.path.join(BASE, 'Moderate', '10-sentiment-analysis-reviews', 'data', 'product_reviews.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"    -> {out} ({len(df)} rows)")


def problem_11_weather():
    print("  Generating weather data...")
    n = 1095
    dates = [datetime(2021, 1, 1) + timedelta(days=i) for i in range(n)]
    day_of_year = np.array([d.timetuple().tm_yday for d in dates])

    base_temp = 15 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp_max = base_temp + np.random.normal(0, 4, n) + 3
    temp_min = base_temp + np.random.normal(0, 4, n) - 5
    temp_min = np.minimum(temp_min, temp_max - 1)

    humidity = 60 + 15 * np.sin(2 * np.pi * (day_of_year - 180) / 365) + np.random.normal(0, 10, n)
    humidity = np.clip(humidity, 10, 100).round(1)

    wind_speed = np.random.lognormal(2.0, 0.4, n).round(1)
    precip_prob = 0.20 + 0.25 * np.sin(2 * np.pi * (day_of_year - 60) / 365)
    precip = np.where(np.random.random(n) < precip_prob,
                      np.random.exponential(5, n).round(1), 0.0)
    precip = np.clip(precip, 0, 80)

    pressure = 1013 + 20 * np.sin(2 * np.pi * (day_of_year - 120) / 365) + np.random.normal(0, 8, n)
    pressure = pressure.round(1)

    missing_mask = np.random.random(n) < 0.03
    precip[missing_mask] = np.nan

    df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in dates],
        'Temp_Max': temp_max.round(1),
        'Temp_Min': temp_min.round(1),
        'Humidity': humidity,
        'WindSpeed': wind_speed,
        'Precipitation': precip,
        'Pressure': pressure
    })

    out = os.path.join(BASE, 'Moderate', '11-weather-forecasting', 'data', 'weather_data.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"    -> {out} ({len(df)} rows)")


def problem_12_customer_purchase():
    print("  Generating customer purchase data...")
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
        0.05
        + 0.15 * (email_opened == 'Yes').astype(int)
        + 0.08 * (ad_clicked == 'Yes').astype(int)
        + 0.05 * ((time_on_site > 10).astype(int))
        + 0.04 * ((pages_visited > 5).astype(int))
        + 0.03 * ((prev_purchases > 3).astype(int))
        - 0.02 * (ages < 25).astype(int)
        + 0.03 * ((income > 70000).astype(int))
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

    purchase_rate = df['MadePurchase'].value_counts().get('Yes', 0) / n
    print(f"    Purchase rate: {purchase_rate:.1%}")

    out = os.path.join(BASE, 'Moderate', '12-customer-purchase-prediction', 'data', 'customer_purchases.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"    -> {out} ({len(df)} rows)")


def problem_13_image_classification():
    print("  Generating MNIST-like digit data...")
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data, digits.target
    indices = np.random.RandomState(42).choice(len(X), 500, replace=False)
    df = pd.DataFrame(X[indices])
    df['label'] = y[indices]
    df.columns = [f'pixel{i}' for i in range(64)] + ['label']

    out = os.path.join(BASE, 'Moderate', '13-image-classification-mnist', 'data', 'digits.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"    -> {out} ({len(df)} rows)")


def main():
    print("Generating data for all 7 Moderate-level problems...\n")

    functions = [
        ("07 - Credit Card Fraud Detection", problem_07_credit_card_fraud),
        ("08 - Employee Attrition Analysis", problem_08_employee_attrition),
        ("09 - Real Estate Price Prediction", problem_09_real_estate),
        ("10 - Sentiment Analysis on Reviews", problem_10_sentiment),
        ("11 - Weather Forecasting", problem_11_weather),
        ("12 - Customer Purchase Prediction", problem_12_customer_purchase),
        ("13 - Image Classification MNIST", problem_13_image_classification),
    ]

    for name, func in functions:
        print(f"Problem {name}")
        func()
        print()

    print("All data generated successfully!")


if __name__ == '__main__':
    main()
