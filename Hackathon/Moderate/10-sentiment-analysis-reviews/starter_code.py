"""Starter code for Sentiment Analysis on Product Reviews."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('data/product_reviews.csv')

print("=== Product Reviews Dataset ===")
print(f"Shape: {df.shape}")
print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")
print(f"\nSample reviews:")
for i, row in df.head(3).iterrows():
    print(f"  [{row['sentiment']}] {row['review_text'][:80]}...")

X_train, X_test, y_train, y_test = train_test_split(
    df['review_text'], df['sentiment'], test_size=0.3, random_state=42, stratify=df['sentiment']
)

vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
