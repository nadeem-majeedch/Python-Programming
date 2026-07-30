# Problem 17: Medical Diagnosis with Limited Data

## Domain
Classification (Small Data)

## Problem Statement
A rare disease affects approximately 10% of the population. Your team has collected patient biomarker data from only 300 patients (30 diagnosed, 270 healthy). Clinical data collection is expensive and slow, so you must build the best possible diagnostic model with what you have. The 30 features include many irrelevant ones, and several features are multicollinear.

Neural networks are explicitly forbidden -- with only 300 samples, a model with even modest capacity would have more parameters than data points and would severely overfit. You must use classical statistical learning with proper regularization, feature selection, and rigorous validation.

## Objectives
1. Build a reliable diagnostic model achieving AUC > 0.80
2. Ensure well-calibrated probabilities (Brier score < 0.20)
3. Select informative features while discarding noise
4. Use appropriate validation (LOO-CV, bootstrap) rather than simple train-test split

## Dataset
- 300 samples, 30 biomarker features
- Binary diagnosis: rare disease (~10% prevalence)
- Many irrelevant features, multicollinearity present
- CSV at `data/medical_diagnosis.csv`

## Success Criteria
- **AUC**: >0.80 on held-out predictions
- **Brier Score**: <0.20 (calibrated probabilities)
- **Feature Selection**: Identify at least the top 5 most important biomarkers
- **Confidence Intervals**: Report 95% CI for AUC using bootstrap

## Starter Code
`starter_code.py` loads data and trains a basic LogisticRegression (which will overfit). You must add regularization and proper validation.

## Constraints
- Neural networks are NOT allowed (too many parameters for 300 samples)
- Must use proper validation strategy (leave-one-out or bootstrap recommended)
- Must report confidence intervals on performance metrics
