#!/usr/bin/env python3
"""Generate all 8 Phase 4 Jupyter notebooks using nbformat."""

import nbformat as nbf
from pathlib import Path
import json

OUT_DIR = Path("/workspace/notebooks/phase-4-advanced-ml")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def new_md(source):
    return nbf.v4.new_markdown_cell(source)


def new_code(source):
    return nbf.v4.new_code_cell(source)


def title_cell(n, title):
    return new_md(f"# Lecture {n} - {title}")


def objectives_cell(*items):
    bullets = "\n".join(f"- {item}" for item in items)
    return new_md("## Learning Objectives\n\n" + bullets)


def topics_cell(*items):
    bullets = "\n".join(f"- {item}" for item in items)
    return new_md("## Key Topics\n\n" + bullets)


def ds_connection(text):
    return new_md("## Data Science Connection\n\n" + text)


def save_notebook(n, title, cells):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {"name": "python", "version": "3.10.0"}
    }
    nb.cells = cells
    fname = f"Lecture {n} - {title}.ipynb"
    path = OUT_DIR / fname
    with open(path, "w") as f:
        nbf.write(nb, f)
    print(f"  Created: {fname}")
    return path


# ─────────────────────────────────────────────────────────────
# Lecture 25 - Introduction to Machine Learning with Scikit-Learn
# ─────────────────────────────────────────────────────────────
def build_lecture_25():
    n, t = 25, "Introduction to Machine Learning with Scikit-Learn"
    cells = [
        title_cell(n, t),
        objectives_cell(
            "Distinguish between supervised and unsupervised learning",
            "Understand the difference between classification and regression",
            "Use train_test_split() to create train and test sets",
            "Fit a model with .fit() and predict with .predict()",
            "Build LinearRegression, KNeighborsClassifier, and LogisticRegression models"
        ),
        topics_cell(
            "Supervised vs unsupervised learning",
            "Classification vs regression",
            "train_test_split()",
            "Feature matrices (X) and target vectors (y)",
            "model.fit(X, y) and model.predict(X_test)",
            "LinearRegression, KNeighborsClassifier, LogisticRegression"
        ),
        # --- ML taxonomy ---
        new_md("""\
## Machine Learning Taxonomy

Machine learning algorithms are broadly divided into two camps: **supervised** and **unsupervised** learning.

In **supervised learning**, we have a dataset with input features and known target labels. The model learns to map from inputs to outputs. Think of it as learning with a teacher — the correct answers are provided during training. Common tasks include **classification** (predicting a category — spam vs not spam) and **regression** (predicting a continuous value — house price).

In **unsupervised learning**, we have only input features and no target labels. The model must find hidden structure in the data on its own — clustering customers into segments or reducing dimensionality for visualisation are typical examples.

Scikit-learn provides a consistent API for all these tasks. Every estimator follows the same pattern: import the class, instantiate it, call `.fit()` to learn from data, and then call `.predict()` or `.transform()` to apply the model to new data."""),

        new_code("""\
# Toy example: create synthetic classification data
from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                           n_redundant=0, random_state=42)
df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(4)])
df["target"] = y
print(df.shape)
print(df["target"].value_counts())"""),

        new_code("""\
# Train / test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")
print(f"Class proportions in train:\\n{pd.Series(y_train).value_counts(normalize=True)}")"""),

        # --- Feature matrices and target vectors ---
        new_md("""\
## Feature Matrices (X) and Target Vectors (y)

Scikit-learn expects data in a very specific shape: **X** must be a 2D array-like (matrix) where each row is an observation and each column is a feature. **y** must be a 1D array-like (vector) containing the target value for each observation.

This design is deliberate — it forces you to separate your predictors from your target before modelling, which clarifies the modelling task. Pandas DataFrames (for X) and Series (for y) work natively with scikit-learn, making the transition from data wrangling to modelling seamless."""),

        new_code("""\
# Explicitly separate features and target using Pandas
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

print("Feature matrix shape:", X.shape)
print("Target vector shape:", y.shape)
print(X.head())"""),

        # --- Fitting and predicting ---
        new_md("""\
## Fitting a Model and Making Predictions

The scikit-learn API follows a two-step workflow:

1. **Fit**: `model.fit(X_train, y_train)` — the model learns parameters from the training data.
2. **Predict**: `model.predict(X_test)` — the model produces predictions on unseen test data.

For classification, you can also use `.predict_proba()` to get class probabilities instead of hard labels. This is especially useful when you want to calibrate confidence thresholds or compute ROC curves.

Never evaluate a model on data it was trained on; that would give an overly optimistic estimate of performance. Always hold out a test set."""),

        new_code("""\
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("First 10 predictions:", y_pred[:10])
print("First 10 true labels:", y_test[:10])
print(f"Accuracy: {(y_pred == y_test).mean():.3f}")"""),

        new_code("""\
# predict_proba gives probability estimates
probs = knn.predict_proba(X_test)
print("Class probabilities for first 5 test samples:")
print(probs[:5].round(3))"""),

        # --- First models ---
        new_md("""\
## First Models: LinearRegression, KNeighborsClassifier, LogisticRegression

Three workhorse models every data scientist should know:

- **LinearRegression** — fits a straight line (or hyperplane) through the data. Simple, interpretable, and fast. Best for relationships that are roughly linear.
- **KNeighborsClassifier** — predicts the class of a point by majority vote among its k nearest neighbours. Non-parametric and flexible, but sensitive to the scale of features and the choice of k.
- **LogisticRegression** — despite its name, a classification model. It models the log-odds of class membership as a linear combination of features. Provides well-calibrated probabilities and is a strong baseline for binary and multiclass problems.

Each of these models is a great starting point for any new dataset."""),

        new_code("""\
# LinearRegression on synthetic regression data
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np

X_reg, y_reg = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
lr = LinearRegression()
lr.fit(X_reg, y_reg)
y_pred_reg = lr.predict(X_reg)

print(f"Coefficient: {lr.coef_[0]:.3f}")
print(f"Intercept: {lr.intercept_:.3f}")"""),

        new_code("""\
# LogisticRegression on the Iris dataset
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X_iris, y_iris, test_size=0.3, random_state=42)

logreg = LogisticRegression(max_iter=200)
logreg.fit(X_tr, y_tr)
print(f"Logistic Regression accuracy: {logreg.score(X_te, y_te):.3f}")
print("Coefficient shape:", logreg.coef_.shape)"""),

        new_code("""\
# KNN classification on Iris dataset
knn_iris = KNeighborsClassifier(n_neighbors=3)
knn_iris.fit(X_tr, y_tr)
y_pred_iris = knn_iris.predict(X_te)
print(f"KNN (k=3) accuracy: {(y_pred_iris == y_te).mean():.3f}")

# Try different k values
for k in [1, 3, 5, 10, 20]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_tr, y_tr)
    acc = knn.score(X_te, y_te)
    print(f"  k={k:2d}  accuracy={acc:.3f}")"""),

        ds_connection("""\
Understanding the train/test split and scikit-learn's .fit() / .predict() API is the foundation of every machine learning pipeline you will build. From predicting house prices to classifying medical images, this same pattern applies. The models introduced here — LinearRegression, KNeighborsClassifier, and LogisticRegression — are the building blocks you will pit against more complex models in later lectures to establish baselines and gauge improvement."""),
    ]
    return save_notebook(n, t, cells)


# ─────────────────────────────────────────────────────────────
# Lecture 26 - Model Evaluation and Cross-Validation
# ─────────────────────────────────────────────────────────────
def build_lecture_26():
    n, t = 26, "Model Evaluation and Cross-Validation"
    cells = [
        title_cell(n, t),
        objectives_cell(
            "Interpret a confusion matrix and basic classification metrics",
            "Use classification_report() and confusion_matrix()",
            "Evaluate regression models with MSE, MAE, and R-squared",
            "Perform k-fold cross-validation with cross_val_score()",
            "Understand learning curves and the bias-variance trade-off"
        ),
        topics_cell(
            "Confusion matrix, accuracy, precision, recall, F1-score",
            "classification_report() and confusion_matrix()",
            "Regression metrics: MSE, MAE, R-squared",
            "cross_val_score() and cross_validate()",
            "Learning curves and validation curves",
            "Bias-variance trade-off"
        ),
        # --- Confusion matrix ---
        new_md("""\
## Confusion Matrix and Classification Metrics

A **confusion matrix** is a table that compares predicted labels against true labels. For binary classification it has four entries:

- **True Positives (TP)** — correctly predicted positives
- **True Negatives (TN)** — correctly predicted negatives
- **False Positives (FP)** — incorrectly predicted positives (Type I error)
- **False Negatives (FN)** — incorrectly predicted negatives (Type II error)

From these four numbers we derive the key classification metrics:
- **Accuracy**: (TP + TN) / (TP + TN + FP + FN) — overall correctness
- **Precision**: TP / (TP + FP) — how many selected items are relevant
- **Recall**: TP / (TP + FN) — how many relevant items are selected
- **F1-score**: harmonic mean of precision and recall — a balanced metric when classes are imbalanced

Scikit-learn's `classification_report()` prints all these metrics at once, making it easy to assess model performance at a glance."""),

        new_code("""\
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
y_pred = knn.predict(X_te)

cm = confusion_matrix(y_te, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix - Iris KNN")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()"""),

        new_code("""\
print(classification_report(y_te, y_pred, target_names=iris.target_names))"""),

        # --- Regression metrics ---
        new_md("""\
## Regression Metrics: MSE, MAE, R-squared

Regression problems require different evaluation metrics:

- **Mean Absolute Error (MAE)**: average absolute difference between predictions and true values. Interpretable in the original units.
- **Mean Squared Error (MSE)**: average squared difference. Penalises large errors more heavily.
- **R-squared (R²)**: proportion of variance in the target explained by the model. Ranges from -∞ to 1, with 1 being perfect.

MAE is more intuitive, while MSE is mathematically convenient (differentiable everywhere). R-squared gives you a scale-free measure of fit quality. Always look at all three to get a complete picture."""),

        new_code("""\
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X_r, y_r = make_regression(n_samples=200, n_features=1, noise=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=0.3, random_state=42)

lr = LinearRegression().fit(X_tr, y_tr)
y_pred_r = lr.predict(X_te)

print(f"MAE:  {mean_absolute_error(y_te, y_pred_r):.2f}")
print(f"MSE:  {mean_squared_error(y_te, y_pred_r):.2f}")
print(f"RMSE: {mean_squared_error(y_te, y_pred_r, squared=False):.2f}")
print(f"R²:   {r2_score(y_te, y_pred_r):.3f}")"""),

        # --- Cross-validation ---
        new_md("""\
## Cross-Validation with cross_val_score()

A single train/test split can be noisy — your estimate of performance depends heavily on which points end up in the test set. **K-fold cross-validation** solves this by splitting the data into k folds, training on k-1 folds, and evaluating on the held-out fold. This process repeats k times, and the scores are averaged.

Scikit-learn's `cross_val_score()` makes this trivially easy. For more detail (fit times, training scores), use `cross_validate()` which returns a dictionary of metrics.

Cross-validation gives a more reliable estimate of how your model will generalise to unseen data and helps detect overfitting when training scores far exceed validation scores."""),

        new_code("""\
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits

digits = load_digits()
X_d, y_d = digits.data, digits.target

rf = RandomForestClassifier(n_estimators=50, random_state=42)
scores = cross_val_score(rf, X_d, y_d, cv=5)
print(f"CV scores: {scores}")
print(f"Mean accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")"""),

        new_code("""\
# Detailed cross-validation with training scores
cv_results = cross_validate(rf, X_d, y_d, cv=5,
                            return_train_score=True)
print("Test scores:", cv_results["test_score"])
print("Train scores:", cv_results["train_score"])
print(f"Mean train: {cv_results['train_score'].mean():.3f}  "
      f"Mean test: {cv_results['test_score'].mean():.3f}")"""),

        # --- Learning curves ---
        new_md("""\
## Learning Curves and the Bias-Variance Trade-Off

A **learning curve** plots model performance on both the training and validation sets as a function of the number of training examples.

- If training score stays high but validation score stays low, the model is **overfitting** (high variance) — it memorises the training data but fails to generalise. Adding more data or reducing model complexity usually helps.
- If both scores converge at a low level, the model is **underfitting** (high bias) — it is too simple to capture the underlying pattern. A more complex model or better features is needed.

This is the **bias-variance trade-off**: simple models have high bias but low variance; complex models have low bias but high variance. The goal is to find the sweet spot that minimises total error on unseen data."""),

        new_code("""\
from sklearn.model_selection import learning_curve
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=50, random_state=42),
    X_d, y_d, cv=5, train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy"
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(8, 4))
plt.plot(train_sizes, train_mean, "o-", label="Training score")
plt.plot(train_sizes, test_mean, "o-", label="Validation score")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.title("Learning Curve - Random Forest on Digits")
plt.legend()
plt.grid(True)
plt.show()"""),

        ds_connection("""\
Evaluation is where machine learning meets scientific rigour. A model is only as good as its measured performance on unseen data. Cross-validation gives you honest estimates, confusion matrices diagnose where your classifier goes wrong, and learning curves guide you toward collecting more data or adjusting model complexity. Every Kaggle competition winner and every production ML system relies on these evaluation tools to build trustworthy models."""),
    ]
    return save_notebook(n, t, cells)


# ─────────────────────────────────────────────────────────────
# Lecture 27 - Feature Engineering and Preprocessing
# ─────────────────────────────────────────────────────────────
def build_lecture_27():
    n, t = 27, "Feature Engineering and Preprocessing"
    cells = [
        title_cell(n, t),
        objectives_cell(
            "Apply one-hot encoding with OneHotEncoder and pd.get_dummies()",
            "Distinguish label encoding from ordinal encoding",
            "Standardise features with StandardScaler",
            "Normalise features with MinMaxScaler",
            "Use ColumnTransformer to mix column types",
            "Chain preprocessing + modelling with Pipeline",
            "Generate polynomial features for non-linear relationships"
        ),
        topics_cell(
            "OneHotEncoder vs pd.get_dummies()",
            "Label encoding vs ordinal encoding",
            "StandardScaler",
            "MinMaxScaler",
            "ColumnTransformer",
            "Pipeline",
            "PolynomialFeatures"
        ),
        # --- One-hot encoding ---
        new_md("""\
## One-Hot Encoding with OneHotEncoder and pd.get_dummies()

Most machine learning algorithms require numerical input. Categorical variables (like "red", "blue", "green") must be converted into numbers. **One-hot encoding** creates a binary column for each category: for a feature with k categories, it produces k binary columns, exactly one of which is 1 for each row.

Pandas provides `pd.get_dummies()` for quick one-hot encoding on DataFrames. Scikit-learn's `OneHotEncoder` integrates seamlessly with Pipelines and `ColumnTransformer`, making it the preferred choice for production workflows.

Be aware of the **dummy variable trap**: when a categorical feature has k categories, you only need k-1 dummy columns to avoid perfect multicollinearity. Use `drop="first"` in OneHotEncoder or `drop_first=True` in pd.get_dummies() to handle this."""),

        new_code("""\
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Sample data with mixed types
df = pd.DataFrame({
    "color": ["red", "blue", "green", "blue", "red"],
    "size": ["S", "M", "L", "M", "XL"],
    "price": [10, 15, 20, 12, 18]
})
print("Original DataFrame:")
print(df)

# One-hot encoding with pandas
df_dummies = pd.get_dummies(df, columns=["color", "size"], drop_first=False)
print("\\nAfter pd.get_dummies():")
print(df_dummies)"""),

        new_code("""\
# One-hot encoding with sklearn OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop="first")
encoded = ohe.fit_transform(df[["color", "size"]])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out())
print("After OneHotEncoder:")
print(encoded_df)"""),

        # --- Label vs Ordinal ---
        new_md("""\
## Label Encoding vs Ordinal Encoding

**Label encoding** assigns each category a unique integer (0, 1, 2, ...). This is simple but can mislead models into thinking categories have an ordinal relationship (e.g., 0 < 1 < 2). For nominal categories like colours, this is dangerous.

**Ordinal encoding** is for variables where the categories do have a natural order (e.g., "small" < "medium" < "large"). You assign integers that respect this ordering. Scikit-learn provides `OrdinalEncoder` with an explicit `categories` parameter to define the order.

Rule of thumb: use one-hot encoding for nominal categories (no order) and ordinal encoding for categories with a clear ordering."""),

        new_code("""\
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# OrdinalEncoder for features with a natural order
sizes = pd.DataFrame({"size": ["S", "M", "L", "XL", "M", "S"]})
ord_enc = OrdinalEncoder(categories=[["S", "M", "L", "XL"]])
sizes_encoded = ord_enc.fit_transform(sizes)
print("Ordinal encoding of sizes:")
print(pd.DataFrame({"size_original": sizes["size"],
                     "size_encoded": sizes_encoded.flatten().astype(int)}))

# LabelEncoder is for the target variable (y), not features
le = LabelEncoder()
y_labels = le.fit_transform(["cat", "dog", "bird", "dog", "cat"])
print("\\nLabel encoding of target:", y_labels)"""),

        # --- StandardScaler & MinMaxScaler ---
        new_md("""\
## Standardization and Normalization

Many ML algorithms (SVM, KNN, LogisticRegression with regularisation, PCA) assume features have similar scales. If one feature ranges from 0-1 and another from 0-100000, the latter dominates the distance calculations.

- **StandardScaler** standardises features by removing the mean and scaling to unit variance: z = (x - μ) / σ. The result has mean 0 and standard deviation 1. Best for algorithms that assume normally distributed data.
- **MinMaxScaler** scales features to a fixed range (usually [0, 1]): x_scaled = (x - min) / (max - min). Preserves the shape of the original distribution. Best for neural networks and algorithms that don't assume any distribution.

Always **fit** the scaler on the training set only, then **transform** both train and test sets to avoid data leakage."""),

        new_code("""\
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

np.random.seed(42)
data = np.random.exponential(scale=2, size=(100, 3))

std_scaler = StandardScaler()
data_std = std_scaler.fit_transform(data)
print(f"StandardScaler - mean: {data_std.mean(axis=0).round(3)}")
print(f"StandardScaler - std:  {data_std.std(axis=0).round(3)}")

mm_scaler = MinMaxScaler()
data_mm = mm_scaler.fit_transform(data)
print(f"\\nMinMaxScaler - min: {data_mm.min(axis=0).round(3)}")
print(f"MinMaxScaler - max: {data_mm.max(axis=0).round(3)}")"""),

        # --- ColumnTransformer ---
        new_md("""\
## ColumnTransformer for Mixed Column Types

Real-world datasets contain a mix of numeric, categorical, and sometimes text features. Applying different preprocessing steps to different columns by hand is error-prone and leads to messy code.

`ColumnTransformer` applies different transformers to different columns in a single object. You specify a list of (name, transformer, columns) tuples. It handles the bookkeeping of joining the transformed columns back together.

This is the professional way to preprocess heterogeneous data in scikit-learn."""),

        new_code("""\
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd

df_full = pd.DataFrame({
    "age": [25, 45, 30, 50, 35],
    "income": [50000, 80000, 60000, 120000, 75000],
    "education": ["HS", "College", "Graduate", "College", "Graduate"],
    "city": ["NYC", "LA", "SF", "NYC", "LA"]
})

ct = ColumnTransformer([
    ("num", StandardScaler(), ["age", "income"]),
    ("cat", OneHotEncoder(drop="first"), ["education", "city"])
])

transformed = ct.fit_transform(df_full)
print("Transformed shape:", transformed.shape)
print(pd.DataFrame(transformed, columns=ct.get_feature_names_out()))"""),

        # --- Pipeline ---
        new_md("""\
## Pipeline: Chaining Preprocessing + Model

A `Pipeline` chains multiple transformers and a final estimator into a single object. When you call `.fit()`, each step transforms the data sequentially before passing it to the final model. During `.predict()`, the same transformations are applied automatically.

Pipelines ensure that preprocessing is applied consistently to train and test sets, prevent data leakage, and make your code cleaner and more reproducible. They are essential for proper cross-validation: `cross_val_score(pipeline, X, y)` will correctly fit the scaler on each training fold separately, avoiding any contamination from the validation fold."""),

        new_code("""\
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=50, random_state=42))
])

# Using the same df_full: create a target
y_full = pd.Series([0, 1, 0, 1, 1], name="high_income")
X_full = df_full

pipe.fit(X_full, y_full)
print("Pipeline score:", pipe.score(X_full, y_full))"""),

        new_code("""\
# Cross-validation with pipeline (no data leakage!)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipe, X_full, y_full, cv=3)
print(f"CV scores: {scores}")
print(f"Mean CV accuracy: {scores.mean():.3f}")"""),

        # --- PolynomialFeatures ---
        new_md("""\
## PolynomialFeatures and Interaction Terms

Linear models can only capture linear relationships. When the relationship between features and target is curved, you need **polynomial features**: adding squared or cubed versions of existing features (x → x, x², x³) lets a linear model fit non-linear patterns.

`PolynomialFeatures` also generates **interaction terms** (x₁ × x₂). These capture multiplicative effects — for example, the combined effect of age and exercise on health.

Be cautious: the number of features grows quickly. A dataset with 10 features and degree 3 produces hundreds of features, risking overfitting and slower computation."""),

        new_code("""\
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X_simple = np.arange(1, 11).reshape(-1, 1)
poly = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly.fit_transform(X_simple)

print("Original -> Polynomial (deg=3):")
print(pd.DataFrame(X_poly, columns=poly.get_feature_names_out(["x"])))"""),

        new_code("""\
# Pipeline with polynomial features + regression
from sklearn.linear_model import LinearRegression

poly_pipe = Pipeline([
    ("poly", PolynomialFeatures(degree=3, include_bias=False)),
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])

np.random.seed(42)
x = np.linspace(-3, 3, 100).reshape(-1, 1)
y = x.ravel()**3 - 2 * x.ravel()**2 + np.random.normal(0, 2, 100)

poly_pipe.fit(x, y)
print("Polynomial regression R²:", poly_pipe.score(x, y))"""),

        ds_connection("""\
Feature engineering is where domain expertise meets data science. Raw data rarely comes in a form ready for modelling — you must encode categories, scale numbers, create interaction terms, and build automated preprocessing pipelines. These skills separate a good data scientist from a great one. The ColumnTransformer and Pipeline tools you learned here are used daily at companies like Netflix (recommendation features), Uber (pricing features), and in virtually every Kaggle-winning solution."""),
    ]
    return save_notebook(n, t, cells)


# ─────────────────────────────────────────────────────────────
# Lecture 28 - Decision Trees, Random Forests, and Ensemble
# ─────────────────────────────────────────────────────────────
def build_lecture_28():
    n, t = 28, "Decision Trees, Random Forests, and Ensemble Methods"
    cells = [
        title_cell(n, t),
        objectives_cell(
            "Train and visualise decision trees",
            "Interpret feature importance from trees",
            "Build random forests and tune key hyperparameters",
            "Understand bagging vs boosting conceptually",
            "Use GradientBoostingClassifier for improved performance"
        ),
        topics_cell(
            "DecisionTreeClassifier / DecisionTreeRegressor",
            "Visualising trees and feature importance",
            "RandomForestClassifier / RandomForestRegressor",
            "Hyperparameters: n_estimators, max_depth, min_samples_split",
            "Bagging vs Boosting",
            "GradientBoostingClassifier"
        ),
        # --- Decision trees ---
        new_md("""\
## Decision Trees

A **decision tree** splits the data recursively based on feature values, creating a flowchart-like structure. Each internal node tests a feature, each branch represents the outcome of the test, and each leaf holds a prediction.

Trees are highly interpretable — you can visualise them and literally read the decision rules. They handle both numeric and categorical data and capture non-linear relationships automatically. However, they tend to **overfit** unless pruned or constrained.

Key hyperparameters to control overfitting:
- `max_depth`: maximum depth of the tree
- `min_samples_split`: minimum samples required to split a node
- `min_samples_leaf`: minimum samples required in a leaf node

Scikit-learn provides `DecisionTreeClassifier` and `DecisionTreeRegressor` with identical APIs."""),

        new_code("""\
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X, y)

plt.figure(figsize=(14, 8))
plot_tree(dt, filled=True, feature_names=iris.feature_names,
          class_names=iris.target_names, rounded=True)
plt.title("Decision Tree (max_depth=3) on Iris Dataset")
plt.show()"""),

        new_code("""\
# Feature importance from a decision tree
importance = pd.DataFrame({
    "feature": iris.feature_names,
    "importance": dt.feature_importances_
}).sort_values("importance", ascending=False)

print("Feature Importance:")
print(importance)"""),

        # --- Random Forest ---
        new_md("""\
## Random Forest: Bagging + Random Feature Selection

A **Random Forest** builds many decision trees on bootstrapped samples of the data and averages their predictions. It also randomly selects a subset of features at each split, which decorrelates the trees.

The result is a model that:
- Retains most of the interpretability benefits (via feature importance)
- Dramatically reduces overfitting compared to a single tree
- Handles high-dimensional data well
- Requires minimal preprocessing (no scaling needed)

Key hyperparameters:
- `n_estimators`: number of trees (more is better, with diminishing returns)
- `max_depth`: constrain tree depth to control overfitting
- `min_samples_split`: prevent splits on very small subsets
- `max_features`: fraction of features to consider at each split"""),

        new_code("""\
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import numpy as np

wine = load_wine()
X_w, y_w = wine.data, wine.target
X_tr, X_te, y_tr, y_te = train_test_split(X_w, y_w, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_tr, y_tr)
print(f"Random Forest accuracy: {rf.score(X_te, y_te):.3f}")"""),

        new_code("""\
# Hyperparameter tuning: n_estimators and max_depth
for n in [10, 50, 100, 200]:
    for d in [None, 5, 10]:
        rf_tune = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42)
        rf_tune.fit(X_tr, y_tr)
        acc = rf_tune.score(X_te, y_te)
        print(f"n_estimators={n:3d}, max_depth={str(d):4s}  accuracy={acc:.3f}")"""),

        new_code("""\
# Feature importance analysis
import pandas as pd
import matplotlib.pyplot as plt

feat_imp = pd.DataFrame({
    "feature": wine.feature_names,
    "importance": rf.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 5))
plt.barh(feat_imp["feature"], feat_imp["importance"])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance - Wine Dataset")
plt.gca().invert_yaxis()
plt.show()"""),

        # --- Bagging vs Boosting ---
        new_md("""\
## Bagging vs Boosting

**Bagging** (Bootstrap Aggregating) trains many models in parallel on different bootstrap samples and averages their predictions. Random Forest is the most famous example. Bagging reduces variance without increasing bias significantly.

**Boosting** trains models sequentially, where each new model focuses on the mistakes made by the previous one. Models are added to correct the errors of the ensemble. Boosting reduces both bias and variance, but can overfit if not carefully regularised.

Common boosting algorithms include AdaBoost, Gradient Boosting, and XGBoost. In practice, gradient boosting often produces state-of-the-art results on tabular data, but requires more hyperparameter tuning than random forests."""),

        new_code("""\
# Basic comparison: RandomForest (bagging) vs GradientBoosting (boosting)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

for name, model in [("RandomForest", rf_model), ("GradientBoosting", gb_model)]:
    scores = cross_val_score(model, X_w, y_w, cv=5)
    print(f"{name:20s}  CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")"""),

        new_code("""\
# GradientBoosting with different learning rates
for lr in [0.01, 0.1, 0.5, 1.0]:
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=lr, random_state=42)
    scores = cross_val_score(gb, X_w, y_w, cv=5)
    print(f"learning_rate={lr:4.2f}  CV accuracy: {scores.mean():.3f}")"""),

        ds_connection("""\
Ensemble methods — particularly random forests and gradient boosting — dominate tabular data competitions on Kaggle and are widely used in industry. Feature importance from tree-based models is one of the most powerful tools for understanding your data: it tells you which features actually drive predictions. When you need a strong out-of-the-box model with minimal preprocessing, reach for a random forest. When you need to squeeze out every last percentage point of accuracy, gradient boosting is your friend."""),
    ]
    return save_notebook(n, t, cells)


# ─────────────────────────────────────────────────────────────
# Lecture 29 - Dimensionality Reduction and Clustering
# ─────────────────────────────────────────────────────────────
def build_lecture_29():
    n, t = 29, "Dimensionality Reduction and Clustering"
    cells = [
        title_cell(n, t),
        objectives_cell(
            "Apply PCA for dimensionality reduction and visualisation",
            "Interpret explained variance ratio and scree plots",
            "Use K-Means clustering and the elbow method",
            "Evaluate clusters with silhouette score",
            "Scale features before PCA and K-Means",
            "Use DBSCAN for density-based clustering"
        ),
        topics_cell(
            "PCA and explained variance ratio",
            "Scree plots",
            "2D visualisation after PCA",
            "K-Means and inertia",
            "Elbow method and silhouette score",
            "StandardScaler before PCA / K-Means",
            "DBSCAN for density-based clustering"
        ),
        # --- PCA ---
        new_md("""\
## PCA: Principal Component Analysis

**PCA** is an unsupervised technique that reduces the dimensionality of data while preserving as much variance as possible. It finds new axes (principal components) that are linear combinations of the original features, ordered by how much variance they capture.

Why reduce dimensions?
- **Visualisation**: project high-dimensional data into 2D or 3D for plotting
- **Noise reduction**: discard low-variance components that may be noise
- **Feature compression**: reduce memory and computation for downstream models
- **Multicollinearity**: decorrelate features before linear models

The `explained_variance_ratio_` tells you the proportion of total variance captured by each component. A **scree plot** visualises this — look for the "elbow" where adding more components yields diminishing returns."""),

        new_code("""\
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X_d, y_d = digits.data, digits.target
print(f"Original shape: {X_d.shape} (64 features)")

# Scale before PCA
X_scaled = StandardScaler().fit_transform(X_d)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"After PCA (2 components): {X_pca.shape}")"""),

        new_code("""\
# 2D visualisation after PCA
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_d, cmap="tab10",
                      alpha=0.7, s=40)
plt.colorbar(scatter, label="Digit")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.title("PCA of Digits Dataset (64D -> 2D)")
plt.grid(True, alpha=0.3)
plt.show()"""),

        new_code("""\
# Scree plot: explained variance ratio
pca_full = PCA().fit(X_scaled)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(pca_full.explained_variance_ratio_) + 1),
         np.cumsum(pca_full.explained_variance_ratio_), "bo-")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Cumulative Explained Variance")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(range(1, 11), pca_full.explained_variance_ratio_[:10])
plt.xlabel("Principal component")
plt.ylabel("Explained variance ratio")
plt.title("Scree Plot (first 10 components)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""),

        # --- K-Means ---
        new_md("""\
## K-Means Clustering

**K-Means** is one of the most popular clustering algorithms. It partitions data into k clusters, where each point belongs to the cluster with the nearest mean (centroid).

The algorithm works iteratively: (1) assign each point to the nearest centroid, (2) recompute centroids as the mean of assigned points, (3) repeat until convergence.

Key concepts:
- **Inertia**: sum of squared distances of samples to their closest centroid. Lower is better, but it decreases monotonically with k, making it a poor standalone criterion for choosing k.
- **Elbow method**: plot inertia vs k and look for the "elbow" point where adding more clusters gives diminishing returns.
- **Silhouette score**: measures how similar a point is to its own cluster vs neighbouring clusters. Ranges from -1 to 1 — higher is better.

Always scale features before K-Means! Otherwise, features with larger scales dominate distance calculations."""),

        new_code("""\
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# K-Means on the PCA-reduced digits data
kmeans = KMeans(n_clusters=10, random_state=42, n_init="auto")
labels = kmeans.fit_predict(X_scaled)

print(f"Silhouette score (10 clusters): "
      f"{silhouette_score(X_scaled, labels):.3f}")

# Visualise clustering in 2D PCA space
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10",
            alpha=0.7, s=40)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c="red", marker="X", s=200, edgecolors="black", label="Centroids")
plt.title("K-Means Clustering on Digits (PCA-reduced)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""),

        new_code("""\
# Elbow method and silhouette score for optimal k
inertias = []
silhouettes = []
k_range = range(2, 15)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(k_range, inertias, "bo-")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(k_range, silhouettes, "ro-")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

best_k = k_range[np.argmax(silhouettes)]
print(f"Optimal k by silhouette score: {best_k}")"""),

        # --- DBSCAN ---
        new_md("""\
## DBSCAN: Density-Based Clustering

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) groups points that are closely packed together, marking points in low-density regions as outliers.

Unlike K-Means:
- You don't need to specify the number of clusters
- It can find arbitrarily shaped clusters
- It handles noise/outliers naturally

The algorithm has two parameters:
- `eps`: maximum distance between two points to be considered neighbours
- `min_samples`: minimum neighbours to form a dense region

DBSCAN is excellent for anomaly detection and datasets with clusters of irregular shapes, but struggles when clusters have vastly different densities."""),

        new_code("""\
from sklearn.cluster import DBSCAN

# Generate data with clusters and outliers
from sklearn.datasets import make_blobs

X_blob, y_blob = make_blobs(n_samples=300, centers=3,
                            cluster_std=1.0, random_state=42)
# Add some outliers
rng = np.random.RandomState(42)
outliers = rng.uniform(low=-10, high=10, size=(30, 2))
X_blob = np.vstack([X_blob, outliers])

dbscan = DBSCAN(eps=1.5, min_samples=5)
db_labels = dbscan.fit_predict(X_blob)

n_clusters = len(set(db_labels) - {-1})
n_noise = list(db_labels).count(-1)
print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")"""),

        new_code("""\
# Visualise DBSCAN results
plt.figure(figsize=(8, 6))
unique_labels = set(db_labels)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

for label, color in zip(unique_labels, colors):
    mask = db_labels == label
    label_name = f"Noise (label={label})" if label == -1 else f"Cluster {label}"
    plt.scatter(X_blob[mask, 0], X_blob[mask, 1],
                c=[color], label=label_name, alpha=0.7, s=30, edgecolors="black")

plt.title(f"DBSCAN Clustering ({n_clusters} clusters, {n_noise} outliers)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""),

        ds_connection("""\
Dimensionality reduction and clustering are the workhorses of unsupervised learning. PCA lets you visualise high-dimensional data and compress features before modelling — it is used in everything from genetics (thousands of genes down to 2D) to finance (factors from hundreds of stock returns). K-Means segments customers, DBSCAN detects credit card fraud. Together, these methods reveal the hidden structure in your data that you cannot see with raw numbers alone."""),
    ]
    return save_notebook(n, t, cells)


# ─────────────────────────────────────────────────────────────
# Lecture 30 - Working with Larger Datasets
# ─────────────────────────────────────────────────────────────
def build_lecture_30():
    n, t = 30, "Working with Larger Datasets: Performance and Optimisation"
    cells = [
        title_cell(n, t),
        objectives_cell(
            "Profile code with %timeit, cProfile, and .memory_usage()",
            "Replace iterative operations with vectorised Pandas code",
            "Use df.eval() and df.query() for faster expressions",
            "Process large datasets in chunks with pd.read_csv(chunksize=...)",
            "Parallelise tasks with concurrent.futures",
            "Identify when to move to databases or Spark"
        ),
        topics_cell(
            "%timeit, cProfile, .memory_usage()",
            "Vectorisation vs iteration",
            "df.eval() and df.query()",
            "Chunked reading with pd.read_csv(chunksize=...)",
            "concurrent.futures",
            "Dask introduction",
            "When to move to database or Spark"
        ),
        # --- Profiling ---
        new_md("""\
## Profiling Code: Time and Memory

Before optimising, you must measure. **Profiling** identifies where your code actually spends time and memory, so you focus your efforts where they matter most.

- **%timeit**: IPython magic command that runs a statement many times and reports the best execution time. Use it to compare small code snippets.
- **cProfile**: built-in module that profiles an entire script and produces a detailed report of function calls and timings.
- **.memory_usage()**: Pandas method that returns memory usage per column in bytes. Use `.info(memory_usage="deep")` for an entire DataFrame.

Rule of thumb: profile first, optimise second. Don't guess where the bottleneck is — measure it."""),

        new_code("""\
# %timeit example (run in a notebook cell, shown for reference)
# %%timeit
# result = sum(range(1000000))

# Memory profiling with Pandas
import pandas as pd
df_mem = pd.DataFrame({
    "int_col": range(100000),
    "float_col": [float(i) for i in range(100000)],
    "str_col": [f"string_{i}" for i in range(100000)]
})
print("Memory per column (bytes):")
print(df_mem.memory_usage(deep=True))
print(f"\\nTotal memory: {df_mem.memory_usage(deep=True).sum() / 1024:.2f} KB")"""),

        new_code("""\
# cProfile usage example (conceptual — run in script)
import cProfile
import pstats

def slow_function():
    total = 0
    for i in range(100000):
        total += i ** 2
    return total

# Uncomment to profile:
# profiler = cProfile.Profile()
# profiler.enable()
# result = slow_function()
# profiler.disable()
# pstats.Stats(profiler).sort_stats("cumulative").print_stats(10)
print("cProfile example ready — uncomment lines above to run.")"""),

        # --- Vectorisation ---
        new_md("""\
## Vectorisation vs Iteration in Pandas

Pandas is built on NumPy, which is written in C. **Vectorised operations** (applying operations to entire columns at once) run at C speed. Iterating row-by-row with `for` loops or `df.iterrows()` forces Python-level overhead on every row, making it 100-1000x slower.

The Golden Rule of Pandas: **never iterate when you can vectorise**.

- Use `df["col"] * 2` instead of a loop over rows
- Use `df["col"].apply(my_function)` when you need element-wise custom logic
- Only fall back to `df.iterrows()` or `df.itertuples()` when vectorisation is truly impossible

If you find yourself writing a for loop over rows in Pandas, stop and ask if there is a vectorised alternative."""),

        new_code("""\
import numpy as np
import pandas as pd
import time

df_vec = pd.DataFrame({
    "a": np.random.rand(100000),
    "b": np.random.rand(100000)
})

# Vectorised operation (fast)
start = time.time()
result_vec = df_vec["a"] * df_vec["b"] + df_vec["a"] ** 2
vec_time = time.time() - start

# Loop operation (slow)
start = time.time()
result_loop = []
for i in range(len(df_vec)):
    result_loop.append(df_vec["a"].iloc[i] * df_vec["b"].iloc[i] + df_vec["a"].iloc[i] ** 2)
loop_time = time.time() - start

print(f"Vectorised: {vec_time:.4f}s")
print(f"Loop:       {loop_time:.4f}s")
print(f"Speed-up:   {loop_time / vec_time:.0f}x")"""),

        new_code("""\
# Using .apply() as a middle ground
def my_func(row):
    return row["a"] * row["b"] + row["a"] ** 2

start = time.time()
result_apply = df_vec.apply(my_func, axis=1)
apply_time = time.time() - start
print(f".apply(): {apply_time:.4f}s  ({vec_time:.4f}s for vectorised)")"""),

        # --- eval() and query() ---
        new_md("""\
## df.eval() and df.query() for Fast Expressions

Pandas' `eval()` and `query()` use **numexpr** under the hood to evaluate expressions without creating intermediate DataFrames. This saves memory and computation.

- `df.eval("new_col = a + b * c")` — compute column expressions efficiently
- `df.query("a > 0.5 & b < 0.8")` — filter rows without building boolean masks

These methods shine on large DataFrames where intermediate arrays would consume significant memory. For small DataFrames (under 10,000 rows), the overhead is often larger than the benefit, so use them when your data is big."""),

        new_code("""\
df_eval = pd.DataFrame(np.random.rand(1000000, 5), columns=list("abcde"))

# Traditional method
start = time.time()
df_eval["f"] = df_eval["a"] + df_eval["b"] * df_eval["c"] - df_eval["d"]
trad_time = time.time() - start

# Using eval
start = time.time()
df_eval.eval("g = a + b * c - d", inplace=True)
eval_time = time.time() - start

print(f"Traditional: {trad_time:.4f}s")
print(f"eval():      {eval_time:.4f}s")
print(f"Speed-up:    {trad_time / eval_time:.2f}x")"""),

        new_code("""\
# Using query for filtering
filtered_trad = df_eval[(df_eval["a"] > 0.5) & (df_eval["b"] < 0.3)]
filtered_query = df_eval.query("a > 0.5 & b < 0.3")

print(f"Traditional filter rows: {len(filtered_trad)}")
print(f"Query filter rows:       {len(filtered_query)}")
print("Results match:", filtered_trad.equals(filtered_query))"""),

        # --- Chunked reading ---
        new_md("""\
## Chunked Reading with pd.read_csv(chunksize=...)

When a dataset is too large to fit in memory, you can process it in **chunks**. `pd.read_csv(chunksize=N)` returns an iterator that yields DataFrames of N rows each. You process each chunk, aggregate the results, and combine them at the end.

This approach works for:
- Computing summary statistics across the entire dataset
- Filtering rows and writing to a smaller output file
- Feature extraction that is row-independent

When even chunked processing is too slow or memory-intensive, it is time to consider out-of-core tools like Dask, a database, or Spark."""),

        new_code("""\
# Simulate chunked processing on a large dataset
import tempfile, os

# Create a large CSV
large_df = pd.DataFrame(np.random.rand(100000, 5), columns=list("abcde"))
large_df.to_csv("/tmp/large_data.csv", index=False)
total_rows = 0
col_sums = np.zeros(5)

for chunk in pd.read_csv("/tmp/large_data.csv", chunksize=10000):
    total_rows += len(chunk)
    col_sums += chunk.sum(numeric_only=True).values

print(f"Processed {total_rows} rows in chunks")
print(f"Column means: {col_sums / total_rows}")

os.remove("/tmp/large_data.csv")"""),

        # --- concurrent.futures ---
        new_md("""\
## Parallel Processing with concurrent.futures

Modern CPUs have multiple cores, but standard Python code only uses one due to the Global Interpreter Lock (GIL). For **CPU-bound** tasks, you can bypass the GIL using `multiprocessing`. For **I/O-bound** tasks (file reads, API calls), threading works well.

The `concurrent.futures` module provides a clean, high-level interface:
- `ThreadPoolExecutor` for I/O-bound tasks
- `ProcessPoolExecutor` for CPU-bound tasks

Both use a `.map()` or `.submit()` pattern to distribute work across workers. The overhead of process/thread creation means parallelism only helps when each unit of work is substantial (at least 0.1s)."""),

        new_code("""\
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

def compute_square(x):
    return sum(i ** 2 for i in range(x))

numbers = [20000, 30000, 25000, 40000, 35000, 15000, 50000, 45000]

# Sequential
start = time.time()
seq_results = [compute_square(n) for n in numbers]
seq_time = time.time() - start
print(f"Sequential: {seq_time:.3f}s")

# Parallel with ProcessPoolExecutor
start = time.time()
with ProcessPoolExecutor(max_workers=4) as executor:
    par_results = list(executor.map(compute_square, numbers))
par_time = time.time() - start
print(f"Parallel (4 workers): {par_time:.3f}s")
print(f"Speed-up: {seq_time / par_time:.1f}x")"""),

        # --- When to move to DB/Spark ---
        new_md("""\
## When to Move to a Database or Spark

Pandas is fantastic for datasets that fit in memory (up to ~10-20 GB on a typical laptop). Beyond that, consider:

1. **Dask**: a drop-in replacement for Pandas that works on larger-than-memory data and distributes across cores or clusters. It uses lazy evaluation and a task scheduler.
2. **SQL database**: for structured, relational data, a database (PostgreSQL, DuckDB) can filter and aggregate far more efficiently than Pandas, using indexes and query optimisation.
3. **Apache Spark**: for truly large-scale data (terabytes+), Spark distributes computation across a cluster. PySpark provides a Pandas-like API.

Rule of thumb: if your data fits in RAM, use Pandas. If it doesn't, try chunks. If chunks are too slow, try Dask or a database. If the data is huge (multi-TB), use Spark."""),

        new_code("""\
# Dask example (conceptual — requires dask installed)
# import dask.dataframe as dd
# ddf = dd.read_csv("huge_dataset.csv")
# result = ddf.groupby("category").mean().compute()
# print(result)
print("Dask usage pattern shown (uncomment to run with dask installed).")

# SQL alternative example
import sqlite3
conn = sqlite3.connect("/tmp/example.db")
df_sql = pd.DataFrame(np.random.rand(1000, 3), columns=list("xyz"))
df_sql.to_sql("mydata", conn, if_exists="replace", index=False)

query_result = pd.read_sql_query("SELECT AVG(x), AVG(y), AVG(z) FROM mydata", conn)
print("\\nSQL aggregation result:")
print(query_result)
conn.close()
os.remove("/tmp/example.db")"""),

        ds_connection("""\
Performance and scalability are what separate hobbyist scripts from production data pipelines. As datasets grow from megabytes to gigabytes to terabytes, the techniques in this lecture — profiling, vectorisation, chunking, and parallelism — become essential. Every major tech company deals with data at scale; knowing when and how to optimise is a core skill for senior data scientists and data engineers. Start with Pandas, scale up with Dask, and reach for Spark when you must."""),
    ]
    return save_notebook(n, t, cells)


# ─────────────────────────────────────────────────────────────
# Lecture 31 - Reproducible Workflows and Best Practices
# ─────────────────────────────────────────────────────────────
def build_lecture_31():
    n, t = 31, "Reproducible Workflows and Best Practices"
    cells = [
        title_cell(n, t),
        objectives_cell(
            "Structure data science projects with a standard directory layout",
            "Use Git for version control in data science projects",
            "Manage dependencies with virtual environments and requirements files",
            "Write unit tests for data pipelines with pytest",
            "Implement logging for data processing steps",
            "Apply notebook best practices for reproducible research"
        ),
        topics_cell(
            "Project directory structure (Cookiecutter Data Science)",
            "Git basics: init, add, commit, push, branching",
            "Virtual environments: conda env and venv",
            "requirements.txt and environment.yml",
            "Unit testing with pytest for data pipelines",
            "Logging with the logging module",
            "Notebook best practices"
        ),
        # --- Project structure ---
        new_md("""\
## Project Directory Structure

A consistent project structure makes your work understandable to collaborators (and your future self). The **Cookiecutter Data Science** template is the de-facto standard:

```
project/
├── data/
│   ├── raw/          # immutable original data
│   ├── processed/    # cleaned, transformed data
│   └── interim/      # intermediate results
├── notebooks/        # exploratory notebooks
├── src/              # reusable Python modules
├── tests/            # unit tests
├── models/           # serialised model files
├── reports/          # generated reports and figures
├── requirements.txt  # project dependencies
├── environment.yml   # conda environment file
├── README.md         # project overview
└── .gitignore        # files git should ignore
```

This separation of raw, processed, and interim data ensures you always know which data is original and which is derived. Notebooks go in their own folder, reusable code goes in `src/`, and models are stored separately so they can be loaded for inference."""),

        new_code("""\
# Example: programmatically create the directory structure
import os
from pathlib import Path

base = Path("/tmp/my_project")
dirs = [
    "data/raw", "data/processed", "data/interim",
    "notebooks", "src", "tests", "models", "reports/figures"
]
for d in dirs:
    (base / d).mkdir(parents=True, exist_ok=True)

print("Project structure created at", base)
for p in sorted(base.rglob("*")):
    if p.is_dir():
        print(f"  {p.relative_to(base)}/")"""),

        # --- Git basics ---
        new_md("""\
## Git Basics for Data Scientists

Version control is not just for software engineers. As a data scientist, you need Git to:

- Track changes to code, notebooks, and configuration files
- Experiment with different approaches on branches
- Collaborate with teammates without overwriting work
- Revert to a previous version when something breaks

Essential commands:
- `git init` — initialise a repository
- `git add <file>` — stage changes
- `git commit -m "message"` — commit staged changes
- `git push` — upload to remote (GitHub/GitLab)
- `git branch <name>` — create a branch for experimentation
- `git checkout <branch>` — switch branches

**Never** commit large data files or credentials. Use `.gitignore` to exclude data files and `.env` files from version control."""),

        new_code("""\
# Git commands shown as strings (run in terminal, not here)
commands = """
# cd /tmp/my_project
# git init
# git add src/ tests/ requirements.txt README.md
# git commit -m "Initial project setup"
# git branch feature/feature-engineering
# git checkout feature/feature-engineering
"""
print("Git workflow:")
for line in commands.strip().split("\\n"):
    if line.strip():
        print(f"  $ {line.strip()}")"""),

        new_code("""\
# Example .gitignore
ignore_rules = [
    "# Data files",
    "*.csv", "*.xlsx", "*.parquet",
    "data/raw/*", "data/processed/*",
    "",
    "# Environment and secrets",
    ".env", "*.env.local",
    "",
    "# Notebook outputs",
    ".ipynb_checkpoints/", "*.nbconvert.*",
    "",
    "# Python cache",
    "__pycache__/", "*.pyc", "*.eggs",
    "",
    "# Models",
    "*.pkl", "*.h5", "models/",
]
print("Recommended .gitignore:")
print("\\n".join(ignore_rules))"""),

        # --- Virtual environments ---
        new_md("""\
## Virtual Environments and Dependency Management

A **virtual environment** isolates your project's dependencies so different projects can use different library versions without conflicts.

```bash
# Using venv (built-in)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Using conda
conda env create -f environment.yml
conda activate my_project
```

Two files capture dependencies:
- **requirements.txt**: flat list of `pip` packages pinning exact versions
- **environment.yml**: conda's richer format supporting channels and conda packages

Always pin exact versions (e.g., `pandas==2.0.3`) rather than loose versions (`pandas>=2.0`) so that anyone can recreate your exact environment."""),

        new_code("""\
# Example requirements.txt as a list
req_lines = [
    "pandas==2.0.3", "numpy==1.25.2", "scikit-learn==1.3.0",
    "matplotlib==3.7.2", "jupyter==1.0.0", "pytest==7.4.0",
    "black==23.7.0", "flake8==6.1.0",
]
print("Example requirements.txt:")
print("\\n".join(req_lines))"""),

        new_code("""\
# Example environment.yml as a multi-line string
env_lines = [
    "name: ds_project",
    "channels:",
    "  - conda-forge",
    "  - defaults",
    "dependencies:",
    "  - python=3.10",
    "  - pandas=2.0.3",
    "  - numpy=1.25.2",
    "  - scikit-learn=1.3.0",
    "  - matplotlib=3.7.2",
    "  - jupyter=1.0.0",
    "  - pip",
    "  - pip:",
    "    - pytest==7.4.0",
    "    - black==23.7.0",
    "    - flake8==6.1.0",
]
print("Example environment.yml:")
print("\\n".join(env_lines))"""),

        # --- Unit tests ---
        new_md("""\
## Writing Unit Tests with pytest

Data pipelines are software, and software should be tested. **Unit tests** verify that individual components (functions, transformations) produce correct outputs.

pytest is the most popular testing framework for Python. Test files go in a `tests/` directory, and test functions start with `test_`. A simple assertion is all you need:

```python
def test_clean_column_names():
    df = pd.DataFrame({"My Column": [1, 2]})
    result = clean_column_names(df)
    assert "my_column" in result.columns
```

Testing data pipelines catches subtle bugs: off-by-one errors, wrong column types, incorrect filtering logic. It may feel slow at first, but it saves hours of debugging later."""),

        new_code("""\
# Define a simple data processing function
def clean_column_names(df):
    import pandas as pd
    df_clean = df.copy()
    df_clean.columns = [
        col.strip().lower().replace(" ", "_") for col in df_clean.columns
    ]
    return df_clean

def clip_outliers(series, lower=0.01, upper=0.99):
    q_low = series.quantile(lower)
    q_high = series.quantile(upper)
    return series.clip(q_low, q_high)

# Test the functions
import pandas as pd
import numpy as np

test_df = pd.DataFrame({"A Col": [10, 20], "B Col": [30, 40]})
cleaned = clean_column_names(test_df)
print("Cleaned columns:", list(cleaned.columns))
assert list(cleaned.columns) == ["a_col", "b_col"]

series = pd.Series([1, 2, 3, 100, 200, 5])
clipped = clip_outliers(series, 0.1, 0.9)
print(f"Clipped series: min={clipped.min()}, max={clipped.max()}")
assert clipped.max() <= series.quantile(0.9)
assert clipped.min() >= series.quantile(0.1)
print("All tests passed!")"""),

        new_code("""\
# Show how tests would be structured in a file
test_script = '''
# tests/test_data_utils.py
import pandas as pd
import numpy as np
from src.data_utils import clean_column_names, clip_outliers

def test_clean_column_names():
    df = pd.DataFrame({"A Col": [1, 2], "B Col": [3, 4]})
    result = clean_column_names(df)
    assert list(result.columns) == ["a_col", "b_col"]

def test_clip_outliers_bounds():
    s = pd.Series([1, 2, 3, 100, 200])
    result = clip_outliers(s, 0.1, 0.9)
    assert result.min() >= s.quantile(0.1)
    assert result.max() <= s.quantile(0.9)

def test_clip_outliers_preserves_middle():
    s = pd.Series([1, 2, 3, 4, 5, 100])
    result = clip_outliers(s, 0.1, 0.9)
    assert result.iloc[2] == 3
'''
print("Example test file (tests/test_data_utils.py):")
print(test_script)"""),

        # --- Logging ---
        new_md("""\
## Logging with the logging Module

Print statements work for quick debugging, but they are not suitable for production pipelines. The **logging** module provides:

- **Log levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Output control**: route logs to console, files, or both
- **Timestamps**: every log entry gets a timestamp automatically
- **Granularity**: turn debug logs on/off without editing code

A typical data pipeline logs each step: "Loading data...", "Cleaning data...found 45 missing values", "Feature engineering...created 12 features", "Training model...CV score=0.87".

This creates an audit trail that makes debugging and monitoring much easier."""),

        new_code("""\
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/tmp/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Simulate a data pipeline with logging
logger.info("Starting data pipeline")

logger.info("Loading data from CSV...")
df_sim = pd.DataFrame(np.random.rand(100, 3), columns=list("abc"))

logger.info(f"Loaded {len(df_sim)} rows with {len(df_sim.columns)} columns")

missing_count = df_sim.isnull().sum().sum()
if missing_count > 0:
    logger.warning(f"Found {missing_count} missing values")
else:
    logger.info("No missing values found")

logger.info("Feature engineering step complete")
logger.info("Pipeline finished successfully")

# Show log file content
with open("/tmp/pipeline.log") as f:
    print("\\nLog file contents:")
    print(f.read())"""),

        # --- Notebook best practices ---
        new_md("""\
## Notebook Best Practices

Jupyter notebooks are the most popular tool for exploratory data analysis, but they can easily become messy. Follow these best practices:

1. **Run top-to-bottom**: always execute cells in order before sharing. Restart the kernel and "Run All" to verify.
2. **Keep cells small**: each cell should do one thing — load data, clean data, visualise, etc.
3. **Use functions**: encapsulate reusable logic in functions, defined at the top of the notebook or in a separate `.py` module.
4. **Clear outputs**: strip cell outputs before committing to Git (use `jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebook.ipynb`).
5. **Use markdown cells**: document your thinking, explain your decisions, and present results in markdown, not code comments.
6. **Version control notebooks**: Jupyter diffs are messy but tools like `nbdime` and ReviewNB make code review possible.
7. **Avoid state-dependence**: don't rely on executing cells in a non-linear order. Use `%load_ext autoreload` to pick up changes from `.py` modules."""),

        new_code("""\
# Demonstration of notebook best practices
# 1. Imports at the top
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2. Reusable functions defined early
def compute_summary_stats(df):
    return pd.DataFrame({
        "mean": df.mean(),
        "std": df.std(),
        "min": df.min(),
        "max": df.max()
    })

# 3. Then data loading, cleaning, analysis
np.random.seed(42)
df_demo = pd.DataFrame(
    np.random.randn(100, 3),
    columns=["feature_a", "feature_b", "feature_c"]
)

# 4. Analysis and visualisation
summary = compute_summary_stats(df_demo)
print("Summary statistics:")
print(summary)

df_demo.hist(bins=20, figsize=(10, 4))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()"""),

        ds_connection("""\
Reproducibility is the cornerstone of scientific computing. A project that cannot be reproduced — by you, six months from now, on a different machine — is not trustworthy. The practices in this lecture — project structure, version control, dependency management, testing, logging, and notebook discipline — transform ad-hoc analyses into professional, auditable data science. Every top data science team follows these practices, and adopting them early will save you countless hours of frustration."""),
    ]
    return save_notebook(n, t, cells)


# ─────────────────────────────────────────────────────────────
# Lecture 32 - Capstone Project: End-to-End Data Science Pipeline
# ─────────────────────────────────────────────────────────────
def build_lecture_32():
    n, t = 32, "Capstone Project: End-to-End Data Science Pipeline"
    cells = [
        title_cell(n, t),
        objectives_cell(
            "Load a dataset and perform initial cleaning",
            "Conduct univariate and multivariate EDA",
            "Engineer features through encoding, scaling, and creation",
            "Train and compare multiple models",
            "Tune hyperparameters and select the best model",
            "Communicate findings with narrative, plots, and model summary"
        ),
        topics_cell(
            "Load and clean: handle missing values, outliers, type fixes",
            "EDA: univariate and multivariate analysis, visualisations",
            "Feature engineering: encoding, scaling, new feature creation",
            "Modelling: train multiple models, tune hyperparameters",
            "Evaluation: compare models, select best, interpret results",
            "Communication: final notebook with narrative, plots, summary"
        ),
        # --- Load and clean ---
        new_md("""\
## Capstone: End-to-End Data Science Pipeline

This capstone brings together everything you have learned in Phase 4. We will work through a complete data science pipeline on the **Wine Quality** dataset from scikit-learn, progressing from raw data to a final model evaluation.

The pipeline will cover:
1. **Load & Clean**: inspect the data, handle missing values and outliers
2. **EDA**: explore distributions, correlations, and relationships
3. **Feature Engineering**: scale, encode, and create new features
4. **Modelling**: train RandomForest and LogisticRegression with cross-validation
5. **Hyperparameter Tuning**: find the best configuration for each model
6. **Evaluation & Comparison**: compare models and select the winner
7. **Communication**: present results with clear narrative and visuals

This is the workflow used in real-world data science projects every day."""),

        new_code("""\
# Step 1: Load and explore the Wine Quality dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target
target_names = wine.target_names

print(f"Dataset shape: {df.shape}")
print(f"Feature names: {list(wine.feature_names)}")
print(f"Target classes: {target_names}")
print(f"\\nFirst 5 rows:")
print(df.head())
print(f"\\nData types:")
print(df.dtypes)"""),

        new_code("""\
# Step 2: Initial cleaning — check missing values and outliers
print("Missing values:")
print(df.isnull().sum())

print("\\nBasic statistics:")
print(df.describe())

# Check for outliers using IQR
def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    return ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()

outlier_counts = df.drop("target", axis=1).apply(count_outliers)
print("\\nOutlier counts per feature:")
print(outlier_counts[outlier_counts > 0])"""),

        # --- EDA ---
        new_md("""\
## Exploratory Data Analysis

EDA is the most important step in any data science project. We will:

1. **Univariate analysis**: examine the distribution of each feature and the target
2. **Multivariate analysis**: explore correlations between features and relationships with the target

These visualisations help us understand which features might be predictive, whether there are data quality issues, and what transformations might be needed.

The Wine Quality dataset has 13 chemical features (alcohol, malic acid, ash, etc.) and a target with 3 classes of wine cultivar. This is a multiclass classification problem."""),

        new_code("""\
# Step 3: Univariate analysis — target distribution and feature histograms
plt.figure(figsize=(14, 6))

# Target distribution
plt.subplot(1, 2, 1)
target_counts = df["target"].value_counts().sort_index()
plt.bar(target_names, target_counts.values)
plt.title("Target Class Distribution")
plt.ylabel("Count")

# Feature distributions
plt.subplot(1, 2, 2)
df.drop("target", axis=1).hist(bins=20, figsize=(12, 10))
plt.suptitle("Feature Distributions", y=1.02)
plt.tight_layout()
plt.show()"""),

        new_code("""\
# Step 4: Multivariate analysis — correlations and pairplots
plt.figure(figsize=(12, 10))
corr = df.drop("target", axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# Pairplot of selected features (to avoid overcrowding)
selected_features = ["alcohol", "malic_acid", "ash", "color_intensity", "proline"]
sns.pairplot(df, vars=selected_features, hue="target", palette="Set1")
plt.show()"""),

        # --- Feature engineering ---
        new_md("""\
## Feature Engineering

Feature engineering transforms raw features into better inputs for our models. For this dataset:

- **Scaling**: many ML algorithms (LogisticRegression with regularisation, KNN) require features on the same scale. We use StandardScaler.
- **New features**: we can create interaction terms (e.g., alcohol × color_intensity) and polynomial features to capture non-linear relationships.
- **Feature selection**: based on our correlation analysis and feature importance from tree-based models, we may drop or keep features.

We will use a `ColumnTransformer` and `Pipeline` to keep our preprocessing clean and prevent data leakage."""),

        new_code("""\
# Step 5: Feature engineering — scaling and new features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

# Create preprocessing pipeline
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features scaled successfully")
print(f"Mean after scaling: {X_train_scaled.mean(axis=0).round(3)[:5]}...")
print(f"Std after scaling:  {X_train_scaled.std(axis=0).round(3)[:5]}...")"""),

        new_code("""\
# Create interaction features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print(f"Original features: {X_train_scaled.shape[1]}")
print(f"After interactions: {X_train_poly.shape[1]}")
print(f"Interaction feature names:")
print(poly.get_feature_names_out()[:15], "...")"""),

        # --- Modelling ---
        new_md("""\
## Modelling: Train and Compare Multiple Models

A good data scientist never settles on one model. We train multiple families of models and compare them using cross-validation:

- **LogisticRegression**: strong linear baseline, well-calibrated probabilities
- **RandomForestClassifier**: handles non-linear relationships, robust to outliers
- **KNeighborsClassifier**: simple, non-parametric baseline

We then tune the most promising model using `GridSearchCV` to find optimal hyperparameters.

All training happens inside a `Pipeline` that includes scaling, so we never leak information from the test set during training."""),

        new_code("""\
# Step 6: Train and compare multiple models with cross-validation
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings("ignore")

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

results = []
for name, model in models.items():
    # Create pipeline with scaling for each model
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])
    scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
    results.append({
        "Model": name,
        "Mean CV Accuracy": scores.mean(),
        "Std CV Accuracy": scores.std()
    })
    print(f"{name:25s}  CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

results_df = pd.DataFrame(results).sort_values("Mean CV Accuracy", ascending=False)"""),

        new_code("""\
# Step 7: Hyperparameter tuning with GridSearchCV
from sklearn.model_selection import GridSearchCV

pipe_rf = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(random_state=42))
])

param_grid = {
    "rf__n_estimators": [50, 100, 200],
    "rf__max_depth": [None, 5, 10, 15],
    "rf__min_samples_split": [2, 5, 10]
}

grid_search = GridSearchCV(
    pipe_rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=0
)
grid_search.fit(X, y)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.3f}")

# Store best model
best_rf = grid_search.best_estimator_"""),

        new_code("""\
# Also tune LogisticRegression
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000, random_state=42))
])

lr_param_grid = {
    "lr__C": [0.01, 0.1, 1, 10, 100],
    "lr__solver": ["lbfgs", "liblinear"]
}

grid_lr = GridSearchCV(
    pipe_lr, lr_param_grid, cv=5, scoring="accuracy", n_jobs=-1
)
grid_lr.fit(X, y)

print(f"Best LR parameters: {grid_lr.best_params_}")
print(f"Best LR CV accuracy: {grid_lr.best_score_:.3f}")
best_lr = grid_lr.best_estimator_"""),

        # --- Evaluation and comparison ---
        new_md("""\
## Model Evaluation and Selection

After tuning, we evaluate the best models on the held-out test set. We use:

- **Accuracy**: overall correctness
- **Confusion matrix**: where does the model get confused between classes?
- **Classification report**: precision, recall, F1 per class
- **Feature importance** (for RandomForest): which features drive predictions?

The final model is chosen based on test set performance, simplicity, and interpretability. Sometimes the slightly less accurate model is preferred because it is simpler to explain and deploy."""),

        new_code("""\
# Step 8: Final evaluation on test set
from sklearn.metrics import confusion_matrix, classification_report

# Evaluate best Random Forest
y_pred_rf = best_rf.predict(X_test)

print("=== Random Forest (Tuned) ===")
print(f"Test accuracy: {best_rf.score(X_test, y_test):.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=target_names))

# Confusion matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix - Random Forest (Test Set)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()"""),

        new_code("""\
# Compare with tuned LogisticRegression
y_pred_lr = best_lr.predict(X_test)

print("=== Logistic Regression (Tuned) ===")
print(f"Test accuracy: {best_lr.score(X_test, y_test):.3f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_pred_lr, target_names=target_names))

cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Greens",
            xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix - Logistic Regression (Test Set)")
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()"""),

        new_code("""\
# Feature importance from the best Random Forest
rf_model = best_rf.named_steps["rf"]
feat_imp = pd.DataFrame({
    "feature": wine.feature_names,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

plt.figure(figsize=(10, 5))
plt.barh(feat_imp["feature"], feat_imp["importance"])
plt.xlabel("Feature Importance")
plt.title("Top Predictive Features - Wine Quality Dataset")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

print("\\nTop 5 features:")
print(feat_imp.head().to_string(index=False))"""),

        # --- Final summary ---
        new_md("""\
## Final Summary and Communication

This capstone demonstrated a complete end-to-end data science pipeline:

1. **Data Loading & Cleaning**: inspected the Wine Quality dataset, checked for missing values and outliers
2. **EDA**: visualised feature distributions, class balance, and correlations; identified key relationships
3. **Feature Engineering**: standardised features and created interaction terms
4. **Modelling**: compared LogisticRegression, RandomForest, and KNN with cross-validation
5. **Hyperparameter Tuning**: used GridSearchCV to optimise RandomForest and LogisticRegression
6. **Evaluation**: compared tuned models on a held-out test set with accuracy, confusion matrices, and classification reports

Both tuned Random Forest and Logistic Regression performed well on this dataset. The final choice depends on whether you prioritise interpretability (LogisticRegression provides clear coefficients) or predictive power (RandomForest captures non-linear interactions automatically). Always complement model performance with domain knowledge when making the final selection."""),

        new_code("""\
# Final comparison table
comparison = pd.DataFrame({
    "Model": ["RandomForest (tuned)", "LogisticRegression (tuned)"],
    "Test Accuracy": [
        best_rf.score(X_test, y_test),
        best_lr.score(X_test, y_test)
    ],
    "Best CV Accuracy": [
        grid_search.best_score_,
        grid_lr.best_score_
    ]
})
print("=== Final Model Comparison ===")
print(comparison.to_string(index=False))

# Determine winner
winner = comparison.loc[comparison["Test Accuracy"].idxmax(), "Model"]
print(f"\\nWinner: {winner}")"""),

        new_code("""\
# Summary visualisation
models_compared = ["LR (tuned)", "RF (tuned)"]
accuracies = [
    best_lr.score(X_test, y_test),
    best_rf.score(X_test, y_test)
]
cv_scores = [
    grid_lr.best_score_,
    grid_search.best_score_
]

x = np.arange(len(models_compared))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, accuracies, width, label="Test Accuracy", color="steelblue")
plt.bar(x + width/2, cv_scores, width, label="Best CV Accuracy", color="coral")
plt.ylabel("Accuracy")
plt.title("Model Performance Comparison")
plt.xticks(x, models_compared)
plt.ylim(0.8, 1.0)
plt.legend()
plt.grid(axis="y", alpha=0.3)

# Add value labels
for i, (acc, cv) in enumerate(zip(accuracies, cv_scores)):
    plt.text(i - width/2, acc + 0.005, f"{acc:.3f}", ha="center", fontsize=10)
    plt.text(i + width/2, cv + 0.005, f"{cv:.3f}", ha="center", fontsize=10)

plt.tight_layout()
plt.show()"""),

        ds_connection("""\
You have now completed the full data science lifecycle — from raw data to a tuned, evaluated model. This capstone pipeline is the same process used by data scientists at companies of all sizes: understand the data, clean it, explore it, engineer features, model, evaluate, and communicate. The Wine Quality dataset is a classic benchmark; the skills you applied here transfer directly to real-world problems. Congratulations on completing Phase 4 — you are ready to apply these techniques to your own datasets and projects."""),
    ]
    return save_notebook(n, t, cells)


# ─────────────────────────────────────────────────────────────
# Main: generate all notebooks
# ─────────────────────────────────────────────────────────────
def main():
    print("Generating Phase 4 notebooks...\n")
    build_lecture_25()
    build_lecture_26()
    build_lecture_27()
    build_lecture_28()
    build_lecture_29()
    build_lecture_30()
    build_lecture_31()
    build_lecture_32()
    print("\nAll notebooks generated successfully!")


if __name__ == "__main__":
    main()
