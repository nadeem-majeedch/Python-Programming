#!/usr/bin/env python3
"""
Master generator for all 6 Easy-level hackathon problems.
Creates README.md, data/generate_data.py, starter_code.py, solution.py, requirements.txt
and runs data generation to produce CSV files.
"""

import os
import sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
EASY = os.path.join(BASE, "Easy")

problems = []


def write(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# ─────────────────────────────────────────────
# Problem 01 – Customer Churn Prediction
# ─────────────────────────────────────────────
problems.append("01-customer-churn-prediction")

write(os.path.join(EASY, "01-customer-churn-prediction", "README.md"), """# Problem 01 – Customer Churn Prediction

**Domain**: Classification  
**Goal**: Predict whether a customer will churn (Yes/No) based on account and demographic information.

## Objectives
- Build a classification model to predict customer churn.
- Compare Logistic Regression vs. Random Forest performance.
- Evaluate models using accuracy, precision, recall, and F1-score.

## Success Criteria
- Achieve accuracy > 75% on the test set.
- Identify top 3 most important features influencing churn.
- Generate a confusion matrix and classification report.

## Dataset: `data/customer_churn.csv`
~2000 rows with columns:
- CustomerID, Tenure, MonthlyCharges, TotalCharges
- ContractType (Month-to-month / One year / Two year)
- PaymentMethod (Electronic check / Credit card / Bank transfer)
- TechSupport (Yes / No / No internet service)
- SeniorCitizen (0/1)
- Churn (Yes/No – target)
""")

write(os.path.join(EASY, "01-customer-churn-prediction", "data", "generate_data.py"), """#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 2000

data = {
    "CustomerID": [f"CUST_{i:05d}" for i in range(1, n + 1)],
    "Tenure": np.random.randint(1, 73, n),
    "MonthlyCharges": np.round(np.random.uniform(20, 120, n), 2),
    "TotalCharges": np.round(np.random.uniform(100, 9000, n), 2),
    "ContractType": np.random.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]
    ),
    "PaymentMethod": np.random.choice(
        ["Electronic check", "Credit card", "Bank transfer"], n, p=[0.40, 0.30, 0.30]
    ),
    "TechSupport": np.random.choice(
        ["Yes", "No", "No internet service"], n, p=[0.30, 0.40, 0.30]
    ),
    "SeniorCitizen": np.random.choice([0, 1], n, p=[0.80, 0.20]),
}

df = pd.DataFrame(data)

contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
payment_map = {"Electronic check": 0, "Credit card": 1, "Bank transfer": 2}
tech_map = {"No": 0, "Yes": 1, "No internet service": 2}

score = (
    -0.05 * df["Tenure"]
    + 0.02 * df["MonthlyCharges"]
    - 0.5 * df["SeniorCitizen"]
    + np.where(df["ContractType"].map(contract_map) == 0, 1.0, -0.5)
    + np.where(df["PaymentMethod"].map(payment_map) == 0, 0.3, -0.2)
    + np.where(df["TechSupport"].map(tech_map) == 1, -0.8, 0.2)
    + np.random.normal(0, 0.5, n)
)

df["Churn"] = np.where(score > 0, "Yes", "No")

out = os.path.join(os.getcwd(), "customer_churn.csv")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
""")

write(os.path.join(EASY, "01-customer-churn-prediction", "starter_code.py"), """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/customer_churn.csv")

# Encode categorical features
label_encoders = {}
for col in ["ContractType", "PaymentMethod", "TechSupport"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df["Churn"] = (df["Churn"] == "Yes").astype(int)

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
""")

write(os.path.join(EASY, "01-customer-churn-prediction", "solution.py"), """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("data/customer_churn.csv")

label_encoders = {}
for col in ["ContractType", "PaymentMethod", "TechSupport"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

df["Churn"] = (df["Churn"] == "Yes").astype(int)

X = df.drop(columns=["CustomerID", "Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("=== Logistic Regression ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_lr):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_lr):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_lr):.3f}")
print(confusion_matrix(y_test, y_pred_lr))

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\\n=== Random Forest ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred_rf):.3f}")
print(f"Precision: {precision_score(y_test, y_pred_rf):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred_rf):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred_rf):.3f}")
print(confusion_matrix(y_test, y_pred_rf))
print("\\nClassification Report:\\n", classification_report(y_test, y_pred_rf))

# Feature importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
print("\\n=== Feature Importance ===")
for i in range(len(X.columns)):
    print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

# Plot
plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), X.columns[indices], rotation=45)
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("\\nSaved feature_importance.png")
""")

write(os.path.join(EASY, "01-customer-churn-prediction", "requirements.txt"), """pandas
numpy
scikit-learn
matplotlib
""")


# ─────────────────────────────────────────────
# Problem 02 – House Price Prediction
# ─────────────────────────────────────────────
problems.append("02-house-price-prediction")

write(os.path.join(EASY, "02-house-price-prediction", "README.md"), """# Problem 02 – House Price Prediction

**Domain**: Regression  
**Goal**: Predict house sale prices based on structural and lot features.

## Objectives
- Build a regression model to predict house prices.
- Engineer new features (Age, PricePerSqFt).
- Compare LinearRegression vs. RidgeCV.

## Success Criteria
- Achieve RMSE < 50,000 on the test set.
- R² score > 0.80.
- Produce a residual plot to evaluate model fit.

## Dataset: `data/house_prices.csv`
~1500 rows with columns:
- SqFt, Bedrooms, Bathrooms, YearBuilt, LotSize, Garage
- Price (target)
""")

write(os.path.join(EASY, "02-house-price-prediction", "data", "generate_data.py"), """#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 1500

sqft = np.random.uniform(800, 5000, n)
bedrooms = np.random.randint(1, 7, n)
bathrooms = np.random.randint(1, 6, n)
year_built = np.random.randint(1950, 2024, n)
lot_size = np.random.uniform(2000, 20000, n)
garage = np.random.randint(0, 4, n)

price = (
    100 * sqft
    + 15000 * bedrooms
    + 10000 * bathrooms
    + 500 * (2023 - year_built)
    + 20 * lot_size
    + 8000 * garage
    + np.random.normal(0, 20000, n)
)

df = pd.DataFrame({
    "SqFt": np.round(sqft, 0).astype(int),
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "YearBuilt": year_built,
    "LotSize": np.round(lot_size, 0).astype(int),
    "Garage": garage,
    "Price": np.round(price, 2),
})

out = os.path.join(os.getcwd(), "house_prices.csv")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
""")

write(os.path.join(EASY, "02-house-price-prediction", "starter_code.py"), """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/house_prices.csv")

X = df.drop(columns=["Price"])
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.2f}")
""")

write(os.path.join(EASY, "02-house-price-prediction", "solution.py"), """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/house_prices.csv")

# Feature engineering
df["Age"] = 2023 - df["YearBuilt"]
df["PricePerSqFt"] = df["Price"] / df["SqFt"]
df["Rooms"] = df["Bedrooms"] + df["Bathrooms"]

feature_cols = ["SqFt", "Bedrooms", "Bathrooms", "LotSize", "Garage", "Age", "Rooms"]

X = df[feature_cols]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("=== Linear Regression ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
print(f"R2:   {r2_score(y_test, y_pred_lr):.4f}")

# RidgeCV
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

print("\\n=== Ridge Regression ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.2f}")
print(f"R2:   {r2_score(y_test, y_pred_ridge):.4f}")
print(f"Best alpha: {ridge.alpha_}")

# Coefficients
print("\\n=== Coefficients ===")
for col, coef in zip(feature_cols, ridge.coef_):
    print(f"{col}: {coef:.4f}")

# Residual plot
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_ridge, y_test - y_pred_ridge, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("residual_plot.png")
print("\\nSaved residual_plot.png")
""")

write(os.path.join(EASY, "02-house-price-prediction", "requirements.txt"), """pandas
numpy
scikit-learn
matplotlib
""")


# ─────────────────────────────────────────────
# Problem 03 – Iris Species Classifier
# ─────────────────────────────────────────────
problems.append("03-iris-species-classifier")

write(os.path.join(EASY, "03-iris-species-classifier", "README.md"), """# Problem 03 – Iris Species Classifier

**Domain**: Classification  
**Goal**: Classify iris flowers into setosa, versicolor, or virginica based on sepal/petal measurements.

## Objectives
- Build a K-Nearest Neighbors classifier.
- Use GridSearchCV to find the optimal k value.
- Compare with a Decision Tree model.
- Visualize decision boundaries.

## Success Criteria
- Achieve accuracy > 90% on the test set.
- Identify the optimal k hyperparameter.
- Generate a confusion matrix.

## Dataset: `data/iris.csv`
150 rows, 3 classes:
- sepal_length, sepal_width, petal_length, petal_width
- species (setosa, versicolor, virginica)
""")

write(os.path.join(EASY, "03-iris-species-classifier", "data", "generate_data.py"), """#!/usr/bin/env python3
import pandas as pd
import os
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = iris.target_names[iris.target]

out = os.path.join(os.getcwd(), "iris.csv")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
""")

write(os.path.join(EASY, "03-iris-species-classifier", "starter_code.py"), """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/iris.csv")

X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
""")

write(os.path.join(EASY, "03-iris-species-classifier", "solution.py"), """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA

df = pd.read_csv("data/iris.csv")

X = df.drop(columns=["species"])
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN with GridSearchCV
param_grid = {"n_neighbors": range(1, 21)}
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring="accuracy")
grid.fit(X_train, y_train)

print(f"Best k: {grid.best_params_['n_neighbors']}")
print(f"Best CV accuracy: {grid.best_score_:.4f}")

y_pred_knn = grid.predict(X_test)
print("\\n=== KNN Test Set ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42, max_depth=3)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\\n=== Decision Tree ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(confusion_matrix(y_test, y_pred_dt))

# Decision tree visualization
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=X.columns, class_names=np.unique(y), filled=True)
plt.savefig("decision_tree.png")
print("\\nSaved decision_tree.png")

# Decision boundary (using first 2 PCA components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
knn_pca = KNeighborsClassifier(n_neighbors=grid.best_params_["n_neighbors"])
knn_pca.fit(X_pca, y)

x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = knn_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pd.Categorical(y).codes, edgecolors="k", cmap="viridis")
plt.legend(handles=scatter.legend_elements()[0], labels=np.unique(y), title="Species")
plt.title("KNN Decision Boundary (PCA-reduced)")
plt.tight_layout()
plt.savefig("decision_boundary.png")
print("Saved decision_boundary.png")
""")

write(os.path.join(EASY, "03-iris-species-classifier", "requirements.txt"), """pandas
numpy
scikit-learn
matplotlib
""")


# ─────────────────────────────────────────────
# Problem 04 – Wine Quality Predictor
# ─────────────────────────────────────────────
problems.append("04-wine-quality-predictor")

write(os.path.join(EASY, "04-wine-quality-predictor", "README.md"), """# Problem 04 – Wine Quality Predictor

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
""")

write(os.path.join(EASY, "04-wine-quality-predictor", "data", "generate_data.py"), """#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    "acidity": np.round(np.random.uniform(3, 8, n), 2),
    "sugar": np.round(np.random.uniform(1, 20, n), 2),
    "alcohol": np.round(np.random.uniform(8, 15, n), 2),
    "pH": np.round(np.random.uniform(2.8, 4.2, n), 2),
    "sulphates": np.round(np.random.uniform(0.3, 0.8, n), 3),
})

# Quality formula
raw = (3
       + 0.5 * df["alcohol"]
       + 0.3 * df["sugar"]
       - 0.4 * df["acidity"]
       + 0.2 * df["pH"]
       + np.random.normal(0, 1.0, n))

df["quality"] = np.clip(np.round(raw).astype(int), 3, 8)
df["quality_label"] = np.where(df["quality"] >= 7, "good", "bad")

out = os.path.join(os.getcwd(), "wine_quality.csv")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
""")

write(os.path.join(EASY, "04-wine-quality-predictor", "starter_code.py"), """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/wine_quality.csv")

X = df.drop(columns=["quality", "quality_label"])
y = (df["quality_label"] == "good").astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
""")

write(os.path.join(EASY, "04-wine-quality-predictor", "solution.py"), """import pandas as pd
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
print("\\n--- Grid Search ---")
param_grid = {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
gs = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring="accuracy")
gs.fit(X_train_clf, y_train_clf)
print(f"Best params: {gs.best_params_}")
print(f"Best CV accuracy: {gs.best_score_:.4f}")

# ── Regression ──
print("\\n=== REGRESSION ===")
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = rf_reg.predict(X_test_reg)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)):.3f}")
print(f"R2:   {r2_score(y_test_reg, y_pred_reg):.3f}")

# Feature importance
importances = rf_clf.feature_importances_
indices = np.argsort(importances)[::-1]
print("\\n=== Feature Importance ===")
for i in range(len(X.columns)):
    print(f"{X.columns[indices[i]]}: {importances[indices[i]]:.4f}")

plt.figure(figsize=(10, 5))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), X.columns[indices], rotation=45)
plt.title("Random Forest Feature Importance (Classification)")
plt.tight_layout()
plt.savefig("wine_feature_importance.png")
print("\\nSaved wine_feature_importance.png")
""")

write(os.path.join(EASY, "04-wine-quality-predictor", "requirements.txt"), """pandas
numpy
scikit-learn
matplotlib
""")


# ─────────────────────────────────────────────
# Problem 05 – Mall Customer Segmentation
# ─────────────────────────────────────────────
problems.append("05-mall-customer-segmentation")

write(os.path.join(EASY, "05-mall-customer-segmentation", "README.md"), """# Problem 05 – Mall Customer Segmentation

**Domain**: Clustering (Unsupervised Learning)  
**Goal**: Segment mall customers into distinct groups based on age, income, and spending behavior.

## Objectives
- Use the Elbow Method to find the optimal number of clusters.
- Apply K-Means clustering with the optimal k.
- Visualize clusters using PCA.
- Profile each cluster's characteristics.

## Success Criteria
- Identify the optimal k from the elbow curve.
- Achieve silhouette score > 0.35.
- Provide meaningful cluster descriptions.

## Dataset: `data/mall_customers.csv`
~300 rows with columns:
- CustomerID, Age, AnnualIncome (k$), SpendingScore (1–100)
""")

write(os.path.join(EASY, "05-mall-customer-segmentation", "data", "generate_data.py"), """#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 300

# Generate 5 distinct cluster centers for realism
centers = [
    (20, 20, 80),   # young, low income, high spend
    (25, 80, 80),   # young, high income, high spend
    (45, 60, 50),   # middle-aged, medium
    (55, 40, 20),   # older, low-medium income, low spend
    (35, 90, 30),   # middle-aged, high income, low spend
]
pts_per_center = n // 5

ages = []
incomes = []
scores = []

for age_c, inc_c, score_c in centers:
    ages.extend(np.random.normal(age_c, 5, pts_per_center))
    incomes.extend(np.random.normal(inc_c, 8, pts_per_center))
    scores.extend(np.random.normal(score_c, 7, pts_per_center))

df = pd.DataFrame({
    "CustomerID": [f"CUST_{i:04d}" for i in range(1, n + 1)],
    "Age": np.clip(np.round(ages).astype(int), 18, 70),
    "AnnualIncome": np.clip(np.round(incomes, 1), 15, 120),
    "SpendingScore": np.clip(np.round(scores, 1), 1, 100),
})

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

out = os.path.join(os.getcwd(), "mall_customers.csv")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
""")

write(os.path.join(EASY, "05-mall-customer-segmentation", "starter_code.py"), """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv("data/mall_customers.csv")

X = df[["AnnualIncome", "SpendingScore"]]

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

plt.scatter(X["AnnualIncome"], X["SpendingScore"], c=df["Cluster"], cmap="viridis")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.title("Customer Segments (k=5)")
plt.savefig("clusters.png")
print("Saved clusters.png")
""")

write(os.path.join(EASY, "05-mall-customer-segmentation", "solution.py"), """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/mall_customers.csv")

X = df[["Age", "AnnualIncome", "SpendingScore"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
inertias = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

# Plot elbow
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, "bo-")
ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Method")
ax2.plot(K_range, sil_scores, "ro-")
ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette Score"); ax2.set_title("Silhouette Analysis")
plt.tight_layout()
plt.savefig("elbow_analysis.png")
print("Saved elbow_analysis.png")

# Optimal k
optimal_k = K_range[np.argmax(sil_scores)]
print(f"\\nOptimal k: {optimal_k} (silhouette score: {max(sil_scores):.4f})")

# K-Means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"], cmap="viridis", edgecolors="k")
plt.legend(*scatter.legend_elements(), title="Cluster")
plt.title(f"PCA Visualization (k={optimal_k})")
plt.tight_layout()
plt.savefig("pca_clusters.png")
print("Saved pca_clusters.png")

# Cluster profiling
print(f"\\n=== Cluster Profiles (k={optimal_k}) ===")
for cluster in sorted(df["Cluster"].unique()):
    subset = df[df["Cluster"] == cluster]
    print(f"\\nCluster {cluster}:")
    print(f"  Size: {len(subset)}")
    print(f"  Avg Age: {subset['Age'].mean():.1f}")
    print(f"  Avg Income: {subset['AnnualIncome'].mean():.1f}")
    print(f"  Avg Spending: {subset['SpendingScore'].mean():.1f}")

# Silhouette score on original scaled data
print(f"\\nSilhouette Score for k={optimal_k}: {silhouette_score(X_scaled, kmeans.labels_):.4f}")
""")

write(os.path.join(EASY, "05-mall-customer-segmentation", "requirements.txt"), """pandas
numpy
scikit-learn
matplotlib
""")


# ─────────────────────────────────────────────
# Problem 06 – Movie Rating Predictor
# ─────────────────────────────────────────────
problems.append("06-movie-rating-predictor")

write(os.path.join(EASY, "06-movie-rating-predictor", "README.md"), """# Problem 06 – Movie Rating Predictor

**Domain**: Regression  
**Goal**: Predict IMDb ratings based on movie attributes like genre, runtime, budget, and release year.

## Objectives
- Build a regression model using one-hot encoded genre.
- Apply Ridge regression to reduce overfitting.
- Create interaction features and evaluate their impact.
- Analyze residuals and feature importance.

## Success Criteria
- Achieve RMSE < 1.0 on the test set.
- R² score > 0.40 (ratings are noisy!).
- Identify which genre has the highest impact on rating.

## Dataset: `data/movie_ratings.csv`
~1000 rows with columns:
- MovieID, Genre, Runtime (min), Budget (M$), ReleaseYear
- IMDbRating (target)
""")

write(os.path.join(EASY, "06-movie-rating-predictor", "data", "generate_data.py"), """#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

np.random.seed(42)
n = 1000

genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror"]
genre_bonus = {"Action": 0.0, "Comedy": 0.5, "Drama": 1.5, "Sci-Fi": 0.3, "Horror": -0.5}

genre_choices = np.random.choice(genres, n, p=[0.25, 0.20, 0.25, 0.15, 0.15])
runtime = np.random.uniform(80, 200, n)
budget = np.random.uniform(1, 200, n)
release_year = np.random.randint(1990, 2024, n)

rating = (
    3.0
    + 0.3 * np.log(budget)
    - 0.01 * runtime
    + np.array([genre_bonus[g] for g in genre_choices])
    + np.random.normal(0, 0.5, n)
)

df = pd.DataFrame({
    "MovieID": [f"MOV_{i:04d}" for i in range(1, n + 1)],
    "Genre": genre_choices,
    "Runtime": np.round(runtime, 0).astype(int),
    "Budget": np.round(budget, 1),
    "ReleaseYear": release_year,
    "IMDbRating": np.clip(np.round(rating, 1), 1.0, 10.0),
})

out = os.path.join(os.getcwd(), "movie_ratings.csv")
df.to_csv(out, index=False)
print(f"Created {out} with {len(df)} rows")
""")

write(os.path.join(EASY, "06-movie-rating-predictor", "starter_code.py"), """import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("data/movie_ratings.csv")

# One-hot encode Genre
df = pd.get_dummies(df, columns=["Genre"], prefix="", prefix_sep="")

X = df.drop(columns=["MovieID", "IMDbRating"])
y = df["IMDbRating"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.3f}")
""")

write(os.path.join(EASY, "06-movie-rating-predictor", "solution.py"), """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/movie_ratings.csv")

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=["Genre"], prefix="", prefix_sep="")
feature_cols = ["Runtime", "Budget", "ReleaseYear"] + [g for g in ["Action", "Comedy", "Drama", "Sci-Fi", "Horror"] if g in df_encoded.columns]

X = df_encoded[feature_cols]
y = df_encoded["IMDbRating"]

# Interaction feature
X["Budget_per_Minute"] = X["Budget"] / X["Runtime"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("=== Linear Regression ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.3f}")
print(f"R2:   {r2_score(y_test, y_pred_lr):.3f}")

# Ridge Regression
ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

print("\\n=== Ridge Regression ===")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.3f}")
print(f"R2:   {r2_score(y_test, y_pred_ridge):.3f}")
print(f"Best alpha: {ridge.alpha_:.2f}")

# Feature importance
print("\\n=== Feature Coefficients ===")
for col, coef in zip(X.columns, ridge.coef_):
    print(f"{col}: {coef:.4f}")

# Residual plot
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_ridge, y_test - y_pred_ridge, alpha=0.5)
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Predicted Rating")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("movie_residual_plot.png")
print("\\nSaved movie_residual_plot.png")
""")

write(os.path.join(EASY, "06-movie-rating-predictor", "requirements.txt"), """pandas
numpy
scikit-learn
matplotlib
""")


# ─────────────────────────────────────────────
# Run all data generators
# ─────────────────────────────────────────────
print("=" * 60)
print("Data-Science-AI-Hackathon / Easy Problem Generator")
print("=" * 60)

for prob in problems:
    gen_path = os.path.join(EASY, prob, "data", "generate_data.py")
    gen_dir = os.path.join(EASY, prob, "data")
    print(f"\n>>> Running generator for {prob} ...")
    old_cwd = os.getcwd()
    os.chdir(gen_dir)
    try:
        exec(open(gen_path).read())
    finally:
        os.chdir(old_cwd)

print("\\n" + "=" * 60)
print("All files generated successfully!")
print("=" * 60)
