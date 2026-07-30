# Customer Purchase Prediction

## Problem Statement
Predict whether a website visitor will make a purchase based on their browsing behavior and demographics.

## Dataset
Generated synthetic dataset with 5000 customer sessions:
- **Demographics**: Age, Gender, Income
- **Behavior**: WebsiteVisits, TimeOnSite, PagesVisited, PreviousPurchases
- **Marketing**: EmailOpened (Yes/No), AdClicked (Yes/No), DiscountUsed (Yes/No)
- **Target**: MadePurchase (Yes/No, ~25% positive rate)

## Objectives
1. Perform EDA and feature correlation analysis
2. Engineer features (VisitToPurchase ratio, engagement score)
3. Apply feature selection (RFE or importance-based)
4. Train RandomForest and GradientBoosting classifiers
5. Tune hyperparameters
6. Select business-appropriate decision threshold

## Success Criteria
- ROC-AUC > 0.80
- Precision > 0.65 at Recall > 0.50
- Identify top 5 purchase drivers

## Evaluation Metrics
- ROC-AUC, Precision, Recall, F1-Score
- Precision-Recall curve
- Feature Importance ranking
- Business-focused threshold analysis
