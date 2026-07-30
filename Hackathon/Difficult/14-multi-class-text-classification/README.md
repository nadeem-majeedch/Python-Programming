# Problem 14: Multi-Class Text Classification with Interpretability

## Domain
NLP + Explainable AI (XAI)

## Problem Statement
You are building a content moderation and categorization engine for a news aggregation platform. The system must automatically classify short text documents into one of five categories (Technology, Health, Finance, Sports, Politics) and provide human-readable explanations for each classification decision. Stakeholders need to trust the model's decisions, especially for borderline cases where documents contain vocabulary that overlaps between categories.

The challenge is deliberately ambiguous: you must decide between traditional machine learning approaches (TF-IDF + linear models) that offer inherent interpretability but potentially lower accuracy, and deep learning approaches (Transformers) that achieve higher accuracy but are harder to explain. Your final submission must justify the trade-off you choose.

## Objectives
1. Achieve >85% classification accuracy on the held-out test set
2. Provide interpretable explanations for individual predictions (feature attribution or attention visualization)
3. Compare at least two fundamentally different approaches and present a quantitative trade-off analysis
4. Report inference latency vs accuracy vs interpretability quality

## Dataset
- 3000 short text documents across 5 categories
- Documents are 20-100 words in length
- Vocabulary overlaps between categories (e.g., "apple" appears in Technology and Finance)
- CSV at `data/text_documents.csv` with columns: doc_id, text, category

## Success Criteria
- **Accuracy**: >85% on test set (macro F1 > 0.83)
- **Interpretability**: For at least 10 test samples, provide feature-attribution explanations that clearly highlight which words drove the prediction
- **Trade-off Analysis**: Quantitative comparison (table/chart) of two approaches showing accuracy, training time, inference time, and interpretability quality score
- **Report**: 1-page analysis of the speed-accuracy-interpretability trade-off

## Starter Code
`starter_code.py` loads the data and provides a basic TF-IDF + LogisticRegression pipeline. You must extend this significantly.

## Constraints
- You may use any open-source library
- You must implement at least one "interpretable" approach AND one "black-box + explanation" approach
- The final model must be serializable (pickle/joblib) for deployment
