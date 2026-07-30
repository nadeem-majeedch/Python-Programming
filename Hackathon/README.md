# Data Science & AI Hackathon Competition Framework

A curated collection of **20 data science and AI problem scenarios** designed to assess student problem-solving capabilities in a competitive hackathon format. Scenarios span classification, regression, time series, NLP, computer vision, anomaly detection, clustering, and recommendation systems.

## Difficulty Levels

| Level | Count | Description |
|-------|-------|-------------|
| **Easy** | 6 | Single-concept problems with clean data. Students apply one or two standard techniques. |
| **Moderate** | 7 | Multi-step workflows requiring data exploration and method selection. Messier, more realistic data. |
| **Difficult** | 7 | Ambiguous problem statements with trade-offs between performance and interpretability. Novel technique combinations. |

## Scenario Index

### Easy

| # | Problem | Domain | Techniques |
|---|---------|--------|------------|
| 01 | [Customer Churn Prediction](Easy/01-customer-churn-prediction) | Classification | Logistic Regression, Random Forest |
| 02 | [House Price Prediction](Easy/02-house-price-prediction) | Regression | Linear Regression, feature selection |
| 03 | [Iris Species Classifier](Easy/03-iris-species-classifier) | Classification | KNN, Decision Trees |
| 04 | [Wine Quality Predictor](Easy/04-wine-quality-predictor) | Classification/Regression | Ensemble methods, threshold tuning |
| 05 | [Mall Customer Segmentation](Easy/05-mall-customer-segmentation) | Clustering | K-Means, PCA visualisation |
| 06 | [Movie Rating Predictor](Easy/06-movie-rating-predictor) | Regression | Linear models, feature encoding |

### Moderate

| # | Problem | Domain | Techniques |
|---|---------|--------|------------|
| 07 | [Credit Card Fraud Detection](Moderate/07-credit-card-fraud-detection) | Anomaly Detection | Class imbalance, threshold tuning, ROC |
| 08 | [Employee Attrition Analysis](Moderate/08-employee-attrition-analysis) | Classification | Imbalanced data, feature importance |
| 09 | [Real Estate Price Prediction](Moderate/09-real-estate-price-prediction) | Regression | Feature engineering, spatial features |
| 10 | [Sentiment Analysis on Reviews](Moderate/10-sentiment-analysis-reviews) | NLP | Text preprocessing, TF-IDF, word vectors |
| 11 | [Weather Forecasting](Moderate/11-weather-forecasting) | Time Series | ARIMA, LSTM basics, seasonal decomposition |
| 12 | [Customer Purchase Prediction](Moderate/12-customer-purchase-prediction) | Classification | Feature engineering, class imbalance |
| 13 | [Image Classification with MNIST](Moderate/13-image-classification-mnist) | Computer Vision | CNN basics, data augmentation |

### Difficult

| # | Problem | Domain | Techniques |
|---|---------|--------|------------|
| 14 | [Multi-Class Text Classification with Interpretability](Difficult/14-multi-class-text-classification) | NLP + XAI | Transformers, LIME/SHAP, attention viz |
| 15 | [Predictive Maintenance](Difficult/15-predictive-maintenance) | Time Series + Classification | Sensor fusion, survival analysis |
| 16 | [Hybrid Recommendation System](Difficult/16-hybrid-recommendation-system) | Recommendation | Matrix factorisation, content-based, cold start |
| 17 | [Medical Diagnosis with Limited Data](Difficult/17-medical-diagnosis-limited-data) | Classification | Small data, transfer learning, augmentation |
| 18 | [Autonomous Vehicle Sensor Fusion](Difficult/18-autonomous-vehicle-sensor-fusion) | Multi-modal | Data fusion, object detection concepts |
| 19 | [Network Anomaly Detection](Difficult/19-network-anomaly-detection) | Time Series + Anomaly | Isolation Forest, autoencoders |
| 20 | [End-to-End ML Pipeline with Deployment Constraints](Difficult/20-end-to-end-ml-pipeline) | Full Pipeline | MLOps, model serving, monitoring |

## Each Scenario Contains

```
scenario-name/
├── README.md          # Problem statement, objectives, success criteria
├── data/              # Sample datasets (synthetic or real-world)
├── starter_code.py    # Basic scaffolding to begin the task
├── solution.py        # Reference solution (for evaluation)
└── requirements.txt   # Python dependencies
```

## Evaluation Guidelines

Hackathon submissions are evaluated on four equally weighted criteria:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Correctness** | 30% | Does the solution meet the stated objectives? Are metrics appropriate and correctly computed? |
| **Reasoning** | 30% | Are design choices justified? Are trade-offs acknowledged? |
| **Code Quality** | 20% | Is the code readable, modular, and well-documented? |
| **Performance** | 20% | How well does the solution perform on held-out evaluation data? |

## How to Run a Hackathon

1. **Preparation**: Fork this repository and distribute scenarios to teams.
2. **Distribution**: Provide each team with their assigned `README.md`, `data/`, and `starter_code.py`. Keep `solution.py` for evaluation.
3. **Timeline**: Recommended 3-4 hours for Easy, 4-6 hours for Moderate, 6-8 hours for Difficult.
4. **Submission**: Teams submit their completed `solution.py` and a brief report.
5. **Evaluation**: Use the reference `solution.py` to verify correctness, and evaluate against the rubric above.

## Requirements

Install scenario-specific dependencies from each `requirements.txt`, or install all at once:

```bash
pip install -r Easy/01-customer-churn-prediction/requirements.txt
```

## License

This hackathon framework is provided for educational and competitive assessment purposes.
