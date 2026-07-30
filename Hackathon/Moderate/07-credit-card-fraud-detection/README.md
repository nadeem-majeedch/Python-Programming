# Credit Card Fraud Detection

## Problem Statement
Build a classifier to detect fraudulent credit card transactions from a highly imbalanced dataset.

## Dataset
Generated synthetic dataset with 5000 transactions:
- **Time**: Hours elapsed since first transaction
- **V1-V28**: PCA-transformed features
- **Amount**: Transaction amount
- **Class**: Target (0 = legitimate, 1 = fraud, ~3% fraud rate)
- Contains ~2.5% missing values in V1-V28 features

## Objectives
1. Handle class imbalance effectively (SMOTE or class_weight)
2. Build and compare multiple classification models
3. Use precision-recall curve (not accuracy) for evaluation
4. Tune decision threshold for optimal F1 score
5. Perform cost-benefit analysis (fraud costs more than false alarms)

## Success Criteria
- ROC-AUC > 0.85
- F1 Score > 0.60
- Demonstrate proper handling of imbalanced data
- Include cost-benefit analysis

## Evaluation Metrics
- Precision, Recall, F1-Score
- ROC-AUC and PR-AUC
- Confusion Matrix
- Cost savings analysis
