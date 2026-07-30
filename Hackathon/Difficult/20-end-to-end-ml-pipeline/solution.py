#!/usr/bin/env python3
"""Reference solution: End-to-End ML Pipeline with Deployment Constraints
Includes: sklearn Pipeline, model serving, drift detection, reproducibility.
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

try:
    from scipy.stats import ks_2samp
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

# ========================
# Load and explore data
# ========================
df = pd.read_csv("data/production_pipeline_data.csv")
df["SignupDate"] = pd.to_datetime(df["SignupDate"])
print(f"Data: {len(df)} rows, {len(df.columns)} columns")
print(f"Drift indicator present (for evaluation only)")

# Separate drift detection
n_normal = 6000
n_drift = 2000
df_normal = df.iloc[:n_normal]
df_drift = df.iloc[n_normal:]

# ========================
# Custom transformers
# ========================
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract year, month, dayofyear from date column."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        dates = pd.to_datetime(X.iloc[:, 0])
        return np.column_stack([
            dates.dt.year,
            dates.dt.month,
            dates.dt.dayofyear,
            dates.dt.quarter
        ])

# ========================
# Build the preprocessing pipeline
# ========================
numeric_features = ["Age", "Income", "CreditScore", "AccountAgeMonths",
                    "NumTransactions", "AvgTransactionAmount"]
categorical_features = ["CustomerTier", "Region", "Feedback"]
date_features = ["SignupDate"]

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

date_transformer = Pipeline([
    ("extractor", DateFeatureExtractor()),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
    ("date", date_transformer, date_features)
])

# Full pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    ))
])

# ========================
# Training (reproducible)
# ========================
print("\n" + "=" * 50)
print("TRAINING PIPELINE")
print("=" * 50)

y = df_normal["Target"]
X = df_normal.drop(columns=["Target", "IsDrifted"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

t0 = time.time()
pipeline.fit(X_train, y_train)
train_time = time.time() - t0

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {auc:.4f}")
print(f"Training time: {train_time:.2f}s")
print(classification_report(y_test, y_pred))

# ========================
# Model size check
# ========================
print("\n" + "=" * 50)
print("MODEL SIZE CHECK")
print("=" * 50)

pickle_path = "model.pkl"
with open(pickle_path, "wb") as f:
    pickle.dump(pipeline, f)

model_size_mb = os.path.getsize(pickle_path) / (1024 * 1024)
print(f"Model size: {model_size_mb:.2f} MB")
print(f"Size < 50MB: {'PASS' if model_size_mb < 50 else 'FAIL'}")

# ========================
# Inference speed
# ========================
print("\n" + "=" * 50)
print("INFERENCE SPEED CHECK")
print("=" * 50)

# Single prediction
single_sample = X_test.iloc[:1]
n_runs = 100
times = []
for _ in range(n_runs):
    t0 = time.time()
    pipeline.predict(single_sample)
    times.append((time.time() - t0) * 1000)  # ms

avg_time = np.mean(times)
print(f"Average single inference: {avg_time:.2f} ms")
print(f"< 100ms: {'PASS' if avg_time < 100 else 'FAIL'}")

# ========================
# Drift Detection
# ========================
print("\n" + "=" * 50)
print("DATA DRIFT DETECTION")
print("=" * 50)

if STATS_AVAILABLE:
    drift_results = {}
    for col in numeric_features:
        normal_data = df_normal[col].dropna()
        drift_data = df_drift[col].dropna()
        stat, p_value = ks_2samp(normal_data, drift_data)
        drift_results[col] = {"KS_stat": stat, "p_value": p_value, "drifted": p_value < 0.05}
    
    drift_count = sum(1 for v in drift_results.values() if v["drifted"])
    print(f"Features with detected drift: {drift_count}/{len(numeric_features)}")
    for col, result in drift_results.items():
        print(f"  {col}: KS={result['KS_stat']:.4f}, p={result['p_value']:.4f}, Drift={'YES' if result['drifted'] else 'NO'}")
    
    if drift_count >= 3:
        print("\nDrift detection PASS: Distribution shift correctly identified")
    else:
        print("\nDrift detection WARNING: Consider more sensitive detection")
else:
    # Simple mean comparison
    for col in numeric_features:
        normal_mean = df_normal[col].mean()
        drift_mean = df_drift[col].mean()
        diff_pct = abs(drift_mean - normal_mean) / normal_mean * 100
        print(f"  {col}: normal={normal_mean:.2f}, drift={drift_mean:.2f}, diff={diff_pct:.1f}%")

# ========================
# Reproducibility check
# ========================
print("\n" + "=" * 50)
print("REPRODUCIBILITY CHECK")
print("=" * 50)

pipeline2 = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    ))
])
pipeline2.fit(X_train, y_train)

y_pred2 = pipeline2.predict(X_test)
identical = (y_pred == y_pred2).all()
print(f"Reproducible predictions: {'YES' if identical else 'NO'}")

# ========================
# API Server Code (commented - uncomment to run)
# ========================
print("\n" + "=" * 50)
print("API SERVER (FastAPI)")
print("=" * 50)

api_code = '''
# Save this as app.py and run with: uvicorn app:host 0.0.0.0 --port 8000

import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ML Pipeline API")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class InputData(BaseModel):
    Age: float
    Income: float
    CreditScore: float
    AccountAgeMonths: float
    NumTransactions: int
    AvgTransactionAmount: float
    CustomerTier: str
    Region: str
    Feedback: str
    SignupDate: str

class Prediction(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=Prediction)
def predict(data: InputData):
    df = pd.DataFrame([data.model_dump()])
    df["SignupDate"] = pd.to_datetime(df["SignupDate"])
    prob = model.predict_proba(df)[0, 1]
    pred = int(prob > 0.5)
    return Prediction(prediction=pred, probability=round(prob, 4))

@app.get("/health")
def health():
    return {"status": "ok"}
'''

print("API code ready. See starter code for how to run.")
print(api_code[:200] + "\n...")

# ========================
# Summary
# ========================
print("\n" + "=" * 50)
print("FINAL SUMMARY")
print("=" * 50)
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC:  {auc:.4f}")
print(f"Model size: {model_size_mb:.2f} MB (<50MB: {'PASS' if model_size_mb < 50 else 'FAIL'})")
print(f"Inference: {avg_time:.2f} ms (<100ms: {'PASS' if avg_time < 100 else 'FAIL'})")
print(f"Drift detection: Drift identified in {drift_count}/{len(numeric_features)} numerical features")
print(f"Reproducible: {'YES' if identical else 'NO'}")
