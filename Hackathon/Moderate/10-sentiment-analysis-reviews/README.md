# Sentiment Analysis on Product Reviews

## Problem Statement
Classify product reviews as Positive, Neutral, or Negative using NLP techniques.

## Dataset
Generated synthetic dataset with 2000 product reviews:
- **review_id**: Unique identifier
- **review_text**: Product review text (contains sentiment words, negation patterns, domain terms, misspellings)
- **sentiment**: Label (Positive ~45%, Neutral ~30%, Negative ~25%)

## Objectives
1. Perform comprehensive text preprocessing (lowercase, punctuation removal, stopwords, lemmatization)
2. Extract features using TF-IDF with n-grams
3. Train and compare multiple classifiers (LogisticRegression, NaiveBayes, LinearSVC)
4. Analyze misclassifications
5. Generate word cloud visualizations

## Success Criteria
- Accuracy > 0.80
- Macro F1 > 0.75
- Identify common misclassification patterns

## Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score (per class and macro)
- Confusion Matrix
- Misclassification analysis
- Word cloud visualizations
