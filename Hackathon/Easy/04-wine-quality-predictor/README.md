# Problem 04 – Wine Quality Predictor

**Domain**: Classification / Regression  
**Goal**: Predict wine quality (good/bad) or the numeric quality score based on physicochemical properties.

## Objectives
- Build both regression (predict numeric quality) and classification (good/bad) models.
- Tune Random Forest hyperparameters.
- Analyze feature importance and tune classification thresholds.

## Success Criteria
- Achieve classification accuracy > 80%.
- Identify the top features influencing wine quality.
- Demonstrate threshold tuning to balance precision and recall.

## Dataset: `data/wine_quality.csv`
~2000 rows with columns:
- acidity, sugar, alcohol, pH, sulphates
- quality (3–8 integer)
- quality_label (good / bad – target for classification)
