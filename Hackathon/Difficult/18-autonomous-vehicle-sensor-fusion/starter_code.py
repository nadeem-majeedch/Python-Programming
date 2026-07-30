#!/usr/bin/env python3
"""Starter code: Autonomous Vehicle Sensor Fusion"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("data/sensor_data.csv")
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Obstacle types:\n{df['ObstacleType'].value_counts()}")

# Simple voting ensemble: each sensor trains its own classifier
# This starter just uses camera features
camera_cols = [c for c in df.columns if c.startswith("Camera_")]
lidar_cols = [c for c in df.columns if c.startswith("LiDAR_")]
radar_cols = [c for c in df.columns if c.startswith("Radar_")]

y = df["ObstacleType"]
X_camera = df[camera_cols].fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X_camera, y, test_size=0.3, random_state=42, stratify=y
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"\nCamera-only accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

# TODO:
# 1. Implement late fusion: train separate models per sensor, then meta-classifier
# 2. Implement early fusion: concatenate all sensor features
# 3. Handle missing data (NaN) from sensor failures
# 4. Test performance when one sensor completely fails
# 5. Add uncertainty quantification
