import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             mean_squared_error, r2_score)

df = pd.read_csv("data/wine_quality.csv")

X = df.drop(columns=["quality", "quality_label"])
y_clf = (df["quality_label"] == "good").astype(int)
y_reg = df["quality"]

X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

# ── Classification ──
print("=== CLASSIFICATION ===")
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X_train_clf, y_train_clf)
y_pred_clf = rf_clf.predict(X_test_clf)

print(f"Accuracy:  {accuracy_score(y_test_clf, y_pred_clf):.3f}")
print(f"Precision: {precision_score(y_test_clf, y_pred_clf):.3f}")
print(f"Recall:    {recall_score(y_test_clf, y_pred_clf):.3f}")
print(f"F1 Score:  {f1_score(y_test_clf, y_pred_clf):.3f}")
print(confusion_matrix(y_test_clf, y_pred_clf))

# Threshold tuning
probs = rf_clf.predict_proba(X_test_clf)[:, 1]
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    preds_th = (probs >= thresh).astype(int)
    print(f"  Threshold {thresh:.1f} -> "
          f"Prec: {precision_score(y_test_clf, preds_th):.3f}, "
          f"Rec: {recall_score(y_test_clf, preds_th):.3f}")

# Hyperparameter tuning
print("\n--- Grid Search ---")
param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy")
gs.fit(X_train_clf, y_train_clf)
print(f"Best params: {gs.best_params_}")
print(f"Best CV accuracy: {gs.best_score_:.4f}")

# ── Regression ──
print("\n=== REGRESSION ===")
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = rf_reg.predict(X_test_reg)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.3f}")
print(f"R2:   {r2_score(y_test_reg, y_pred_reg):.3f}")

# Feature importance
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("\n=== Feature Importance ===")
for i in range(len(X.columns)):
    print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), X.columns[indices], rotation=45)
plt.title("Random Forest Feature Importance (Classification)")
plt.tight_layout()
plt.savefig("wine_feature_importance.png")
print("\nSaved wine_feature_importance.png")
