import pandas as pd
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
print("\n=== KNN Test Set ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(confusion_matrix(y_test, y_pred_knn))
print(classification_report(y_test, y_pred_knn))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42, max_depth=3)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\n=== Decision Tree ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(confusion_matrix(y_test, y_pred_dt))

# Decision tree visualization
plt.figure(figsize=(12, 8))
plot_tree(dt, feature_names=X.columns, class_names=np.unique(y), filled=True)
plt.savefig("decision_tree.png")
print("\nSaved decision_tree.png")

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
