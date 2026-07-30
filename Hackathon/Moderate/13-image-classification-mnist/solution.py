"""Reference solution for Image Classification with MNIST.

Approach:
1. Load digits dataset and visualize samples
2. Train RandomForest with hyperparameter tuning
3. Train MLPClassifier (sklearn neural network)
4. PCA for dimensionality reduction and 2D visualization
5. Confusion matrix and per-class metrics
6. Analyze misclassified examples
7. Demonstrate simple data augmentation (shift)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
from scipy.ndimage import shift

df = pd.read_csv('data/digits.csv')

print("=" * 60)
print("IMAGE CLASSIFICATION (MNIST) - SOLUTION")
print("=" * 60)

X = df.drop('label', axis=1).values
y = df['label'].values

print(f"\nDataset shape: {X.shape}")
print(f"Classes: {np.unique(y)}")

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    img = X[i].reshape(8, 8)
    ax.imshow(img, cmap='gray')
    ax.set_title(f'Label: {y[i]}')
    ax.axis('off')
plt.suptitle('Sample Digits')
plt.tight_layout()
print("\nVisualized sample digits")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("\n=== RandomForest Classifier ===")
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_params = {'n_estimators': [100, 200], 'max_depth': [8, 12, None], 'min_samples_split': [2, 5]}
rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train, y_train)
rf_best = rf_grid.best_estimator_
y_pred_rf = rf_best.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Best params: {rf_grid.best_params_}")
print(f"Accuracy: {acc_rf:.4f}")
print(f"Per-class metrics:\n{classification_report(y_test, y_pred_rf)}")

print("\n=== MLPClassifier (Neural Network) ===")
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp)
print(f"Accuracy: {acc_mlp:.4f}")
print(f"Per-class metrics:\n{classification_report(y_test, y_pred_mlp)}")

print("\n=== Data Augmentation (Shift) ===")

def augment_images(images, labels, shift_range=1):
    augmented_images, augmented_labels = [], []
    for img, lbl in zip(images, labels):
        img_2d = img.reshape(8, 8)
        for dx, dy in [(0, 0), (shift_range, 0), (-shift_range, 0), (0, shift_range), (0, -shift_range)]:
            shifted = shift(img_2d, (dy, dx), mode='constant', cval=0)
            augmented_images.append(shifted.flatten())
            augmented_labels.append(lbl)
    return np.array(augmented_images), np.array(augmented_labels)

X_aug, y_aug = augment_images(X_train, y_train, shift_range=1)
print(f"Augmented training set: {X_aug.shape} (from {X_train.shape[0]})")

mlp_aug = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True)
mlp_aug.fit(X_aug, y_aug)
y_pred_aug = mlp_aug.predict(X_test)
acc_aug = accuracy_score(y_test, y_pred_aug)
print(f"MLP with augmentation - Accuracy: {acc_aug:.4f}")

print("\n=== PCA Visualization ===")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
ax.set_title(f'PCA 2D Projection (explained variance: {pca.explained_variance_ratio_.sum():.2%})')
plt.colorbar(scatter, ax=ax)
print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

print("\n=== Misclassification Analysis ===")
misclassified_idx = np.where(y_pred_rf != y_test)[0]
print(f"Total misclassified: {len(misclassified_idx)} out of {len(y_test)} ({len(misclassified_idx)/len(y_test)*100:.1f}%)")

if len(misclassified_idx) > 0:
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        if i < len(misclassified_idx):
            idx = misclassified_idx[i]
            img = X_test[idx].reshape(8, 8)
            ax.imshow(img, cmap='gray')
            ax.set_title(f'True: {y_test[idx]}, Pred: {y_pred_rf[idx]}', fontsize=8)
            ax.axis('off')
    plt.suptitle('Misclassified Examples')
    plt.tight_layout()
    print(f"Displayed {min(10, len(misclassified_idx))} misclassified examples")

print(f"\n=== Model Comparison ===")
print(f"{'Model':<30} {'Accuracy':<10}")
print(f"{'RandomForest':<30} {acc_rf:<10.4f}")
print(f"{'MLPClassifier':<30} {acc_mlp:<10.4f}")
print(f"{'MLP + Augmentation':<30} {acc_aug:<10.4f}")
