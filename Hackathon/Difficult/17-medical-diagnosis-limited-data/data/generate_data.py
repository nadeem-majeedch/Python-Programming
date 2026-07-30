#!/usr/bin/env python3
"""Generate synthetic medical diagnosis data with limited samples."""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

n_samples = 300
n_features = 30
prevalence = 0.10

# Create latent structure: only 8 features are truly informative
n_informative = 8
n_noise = n_features - n_informative

# Generate labels
y = np.random.binomial(1, prevalence, n_samples)

# Informative features with multicollinearity
X_informative = np.zeros((n_samples, n_informative))
for i in range(n_informative):
    base = np.random.randn(n_samples) * 0.5
    # Add disease signal
    effect_size = np.random.uniform(1.0, 2.5)
    base += y * effect_size
    X_informative[:, i] = base

# Create multicollinearity: feature 3 is correlated with feature 1+2
X_informative[:, 2] = 0.7 * X_informative[:, 0] + 0.3 * X_informative[:, 1] + np.random.randn(n_samples) * 0.3
# Feature 7 is correlated with feature 5
X_informative[:, 6] = 0.8 * X_informative[:, 4] + np.random.randn(n_samples) * 0.4

# Noise features (pure random)
X_noise = np.random.randn(n_samples, n_noise) * 0.8

# Combine
X = np.hstack([X_informative, X_noise])

# Add some label noise (flip 5% of labels)
n_flip = int(n_samples * 0.05)
flip_idx = np.random.choice(n_samples, n_flip, replace=False)
y[flip_idx] = 1 - y[flip_idx]

# Create DataFrame
feature_names = [f"Biomarker_{i+1:02d}" for i in range(n_features)]
df = pd.DataFrame(X, columns=feature_names)
df["Diagnosis"] = y

out_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "medical_diagnosis.csv")
df.to_csv(out_path, index=False)
print(f"Generated {len(df)} samples (positive: {df['Diagnosis'].sum()}, {df['Diagnosis'].mean()*100:.1f}%) -> {out_path}")
