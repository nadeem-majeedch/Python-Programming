# Image Classification with MNIST

## Problem Statement
Build a classifier to recognize handwritten digits (0-9) from images.

## Dataset
Uses the scikit-learn digits dataset (8x8 grayscale images, 500 samples):
- 64 pixel features (flattened 8x8 images)
- Target: digit label (0-9)

## Objectives
1. Load and visualize sample digit images
2. Train a RandomForest classifier on flattened pixels
3. Build a simple MLP neural network
4. Apply PCA for dimensionality reduction and visualization
5. Analyze confusion matrix and per-class metrics
6. Visualize misclassified examples

## Success Criteria
- Accuracy > 0.85
- Per-class F1 > 0.75 for all digits
- Demonstrate confusion matrix analysis

## Evaluation Metrics
- Accuracy, Per-class Precision/Recall/F1
- Confusion Matrix
- Misclassification visualization
- PCA explained variance ratio
