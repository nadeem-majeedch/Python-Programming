#!/usr/bin/env python3
"""Reference solution: Autonomous Vehicle Sensor Fusion
Implements early fusion, late fusion, and handles missing data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ========================
# Load and prepare data
# ========================
df = pd.read_csv("data/sensor_data.csv")

camera_cols = [c for c in df.columns if c.startswith("Camera_")]
lidar_cols = [c for c in df.columns if c.startswith("LiDAR_")]
radar_cols = [c for c in df.columns if c.startswith("Radar_")]
all_sensor_cols = camera_cols + lidar_cols + radar_cols

y = df["ObstacleType"]
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Create missing-sensor indicators
for sensor in ["Camera", "LiDAR", "Radar"]:
    cols = [c for c in df.columns if c.startswith(f"{sensor}_")]
    df[f"{sensor}_missing"] = df[cols].isna().all(axis=1).astype(int)

indicator_cols = [c for c in df.columns if c.endswith("_missing")]

# Split
X_all = df[all_sensor_cols].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Categories: {le.classes_}")

# ================================
# Strategy 1: Early Fusion
# ================================
print("\n" + "=" * 50)
print("STRATEGY 1: EARLY FUSION (concatenate all features)")
print("=" * 50)

early_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=150, random_state=42))
])

early_pipeline.fit(X_train, y_train)
y_pred_early = early_pipeline.predict(X_test)
acc_early = accuracy_score(y_test, y_pred_early)
print(f"Accuracy: {acc_early:.4f}")
print(classification_report(y_test, y_pred_early))

# ================================
# Strategy 2: Late Fusion (meta-classifier)
# ================================
print("\n" + "=" * 50)
print("STRATEGY 2: LATE FUSION (separate models + meta-classifier)")
print("=" * 50)

def train_sensor_model(sensor_cols, name):
    """Train a model for a single sensor."""
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train[sensor_cols], y_train)
    y_pred = pipe.predict(X_test[sensor_cols])
    acc = accuracy_score(y_test, y_pred)
    print(f"  {name} accuracy: {acc:.4f}")
    return pipe

cam_model = train_sensor_model(camera_cols, "Camera")
lidar_model = train_sensor_model(lidar_cols, "LiDAR")
radar_model = train_sensor_model(radar_cols, "Radar")

# Meta-features: concatenate predicted probabilities from each sensor model
def get_meta_features(models, sensor_cols_list, X):
    """Generate meta-features from base models."""
    meta_feats = []
    for model, cols in zip(models, sensor_cols_list):
        probs = model.predict_proba(X[cols])
        meta_feats.append(probs)
    return np.hstack(meta_feats)

sensor_models = [cam_model, lidar_model, radar_model]
sensor_cols_list = [camera_cols, lidar_cols, radar_cols]

train_meta = get_meta_features(sensor_models, sensor_cols_list, X_train)
test_meta = get_meta_features(sensor_models, sensor_cols_list, X_test)

meta_clf = RandomForestClassifier(n_estimators=100, random_state=42)
meta_clf.fit(train_meta, y_train)
y_pred_late = meta_clf.predict(test_meta)
acc_late = accuracy_score(y_test, y_pred_late)
print(f"\nLate fusion accuracy: {acc_late:.4f}")
print(classification_report(y_test, y_pred_late))

# ================================
# Sensor Failure Simulation
# ================================
print("\n" + "=" * 50)
print("SENSOR FAILURE SIMULATION")
print("=" * 50)

def simulate_sensor_failure(X, cols_to_drop, strategy="mean"):
    """Simulate sensor failure by masking features."""
    X_fail = X.copy()
    for col in cols_to_drop:
        X_fail[col] = np.nan
    return X_fail

failure_scenarios = {
    "All sensors": all_sensor_cols,
    "No Camera": camera_cols,
    "No LiDAR": lidar_cols,
    "No Radar": radar_cols,
    "Camera + LiDAR": camera_cols + lidar_cols
}

for scenario, dropped_cols in failure_scenarios.items():
    remaining_cols = [c for c in all_sensor_cols if c not in dropped_cols]
    if not remaining_cols:
        continue
    X_test_fail = simulate_sensor_failure(X_test, dropped_cols)
    
    # Early fusion on remaining
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train[remaining_cols], y_train)
    y_pred_fail = pipe.predict(X_test_fail[remaining_cols])
    acc_fail = accuracy_score(y_test, y_pred_fail)
    print(f"  {scenario}: accuracy = {acc_fail:.4f}")

# ================================
# Uncertainty Estimation
# ================================
print("\n" + "=" * 50)
print("UNCERTAINTY QUANTIFICATION")
print("=" * 50)

# Use prediction confidence as uncertainty measure
y_prob = early_pipeline.predict_proba(X_test)
max_probs = y_prob.max(axis=1)
mean_conf = max_probs.mean()
print(f"Mean prediction confidence: {mean_conf:.4f}")

# Confidence correlates with correctness?
correct = y_pred_early == y_test
correct_conf = max_probs[correct].mean()
incorrect_conf = max_probs[~correct].mean()
print(f"Average confidence when correct: {correct_conf:.4f}")
print(f"Average confidence when incorrect: {incorrect_conf:.4f}")
print(f"Confidence gap: {correct_conf - incorrect_conf:.4f}")

# ================================
# Final Summary
# ================================
print("\n" + "=" * 50)
print("FINAL RESULTS")
print("=" * 50)
print(f"Early fusion accuracy:   {acc_early:.4f}")
print(f"Late fusion accuracy:    {acc_late:.4f}")
print(f"All-sensors threshold:   > 90% = {'PASS' if acc_early > 0.90 else 'NEEDS IMPROVEMENT'}")
