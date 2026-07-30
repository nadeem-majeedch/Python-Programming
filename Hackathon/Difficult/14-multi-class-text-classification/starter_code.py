#!/usr/bin/env python3
"""Starter code: Multi-Class Text Classification with Interpretability"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv("data/text_documents.csv")
X = df["text"]
y = df["category"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train LogisticRegression
model = LogisticRegression(max_iter=1000, multi_class="multinomial")
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# TODO:
# 1. Improve accuracy with better preprocessing and hyperparameter tuning
# 2. Add interpretability (LIME, SHAP, or custom feature importance)
# 3. Implement a deep learning alternative (e.g., DistilBERT)
# 4. Compare approaches on accuracy, speed, and interpretability
