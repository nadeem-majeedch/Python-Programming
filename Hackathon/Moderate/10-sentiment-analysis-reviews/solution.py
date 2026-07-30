"""Reference solution for Sentiment Analysis on Product Reviews.

Approach:
1. Text preprocessing: lowercase, remove punctuation/numbers, stopwords, lemmatization
2. TF-IDF vectorization with unigrams + bigrams
3. Multiple classifiers: LogisticRegression, MultinomialNB, LinearSVC
4. Confusion matrix and misclassification analysis
5. Word cloud visualization for each sentiment class
"""

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score
)
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

df = pd.read_csv('data/product_reviews.csv')

print("=" * 60)
print("SENTIMENT ANALYSIS - SOLUTION")
print("=" * 60)

print(f"\nDataset shape: {df.shape}")
print(f"Class distribution:\n{df['sentiment'].value_counts()}")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df['clean_text'] = df['review_text'].apply(preprocess_text)
print(f"\nSample after preprocessing:")
print(f"  Original: {df['review_text'].iloc[0][:80]}...")
print(f"  Cleaned:  {df['clean_text'].iloc[0][:80]}...")

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['sentiment'], test_size=0.3, random_state=42, stratify=df['sentiment']
)

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

classifiers = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    'MultinomialNB': MultinomialNB(alpha=0.5),
    'LinearSVC': LinearSVC(max_iter=2000, random_state=42, C=1.0),
}

results = []
for name, clf in classifiers.items():
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    results.append({'Model': name, 'Accuracy': acc, 'Macro F1': f1_macro})

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

    misclassified = X_test[y_pred != y_test]
    actual = y_test[y_pred != y_test]
    predicted = y_pred[y_pred != y_test]
    if len(misclassified) > 0:
        print("\nMisclassification examples:")
        for j in range(min(3, len(misclassified))):
            print(f"  Actual: {actual.iloc[j]:<10} Predicted: {predicted.iloc[j]:<10} | {misclassified.iloc[j][:70]}...")

print("\n=== Feature Importance (Top LogisticRegression Coefficients) ===")
feature_names = vectorizer.get_feature_names_out()
lr = classifiers['LogisticRegression']

for i, class_name in enumerate(lr.classes_):
    coef = lr.coef_[i]
    top_positive = pd.Series(coef, index=feature_names).nlargest(10)
    print(f"\nTop words for '{class_name}':")
    for word, val in top_positive.items():
        print(f"  {word}: {val:.4f}")

results_df = pd.DataFrame(results)
print(f"\n=== Summary ===")
print(results_df.to_string(index=False))
