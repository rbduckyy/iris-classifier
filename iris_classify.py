# iris_classify.py
# Beginner-friendly end-to-end workflow:
# - Load iris.csv
# - Basic EDA (exploratory data analysis)
# - Train/test split
# - Two models: Logistic Regression & k-NN
# - Metrics + confusion matrix plot

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------------
# 1) Load data
# ---------------------------
csv_path = Path("iris.csv")
if not csv_path.exists():
    raise FileNotFoundError("iris.csv not found. Place it in the same folder as this script.")

df = pd.read_csv(csv_path)

print("\n=== First 5 rows ===")
print(df.head())

print("\n=== Info ===")
print(df.info())

print("\n=== Descriptive stats ===")
num_df = df.select_dtypes(include=["number"])
print(num_df.describe())

# Try to standardize expected columns
# Common iris formats:
# sepal_length,sepal_width,petal_length,petal_width,species
expected_cols = {"sepal_length","sepal_width","petal_length","petal_width","species"}
if expected_cols.issubset(set(df.columns)):
    feature_cols = ["sepal_length","sepal_width","petal_length","petal_width"]
    target_col = "species"
else:
    # Fallback: try common alternatives (e.g., from scikit-learn CSVs)
    # Adjust here if your columns differ.
    # Print columns to help user map them.
    print("\nColumns found:", list(df.columns))
    raise ValueError(
        "Column names differ from the expected Iris format. "
        "Rename columns to sepal_length,sepal_width,petal_length,petal_width,species."
    )

# ---------------------------
# 2) Basic cleaning checks
# ---------------------------
print("\n=== Missing values per column ===")
print(df[feature_cols + [target_col]].isna().sum())

# Drop rows with any missing values (Iris typically has none, but this is safe)
df = df.dropna(subset=feature_cols + [target_col])

# ---------------------------
# 3) Quick visualization (scatter)
# ---------------------------
sns.set(style="whitegrid", context="notebook")

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df, x="sepal_length", y="sepal_width", hue=target_col,
    palette="Set2", alpha=0.9
)
plt.title("Iris: Sepal Length vs. Sepal Width")
plt.tight_layout()

# Save FIRST, then show or close
plt.savefig("scatter_sepal.png", dpi=150)
plt.show()  # or: plt.close()

# Pairplot for more relationships (can be slower)
g = sns.pairplot(df, hue=target_col, diag_kind="hist", corner=True, palette="Set2")
# suptitle must be set on the Figure attached to the PairGrid:
g.fig.suptitle("Iris Pairwise Feature Relationships", y=1.02)

# Save the pairplot via the grid object
g.savefig("pairplot.png", dpi=150)

plt.show()  # or: plt.close(g.fig)

# ---------------------------
# 4) Train/test split
# ---------------------------
X = df[feature_cols].values
y = df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------
# 5) Standardize features
# ---------------------------
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# ---------------------------
# 6) Models
# ---------------------------
# (A) Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_std, y_train)
y_pred_lr = log_reg.predict(X_test_std)
acc_lr = accuracy_score(y_test, y_pred_lr)

# (B) k-Nearest Neighbors (k=5 by default)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_std, y_train)
y_pred_knn = knn.predict(X_test_std)
acc_knn = accuracy_score(y_test, y_pred_knn)

print("\n=== Logistic Regression ===")
print(f"Accuracy: {acc_lr:.3f}")
print(classification_report(y_test, y_pred_lr))

print("\n=== k-NN (k=5) ===")
print(f"Accuracy: {acc_knn:.3f}")
print(classification_report(y_test, y_pred_knn))

# ---------------------------
# 7) Confusion matrix plot (for k-NN)
# ---------------------------
cm = confusion_matrix(y_test, y_pred_knn, labels=np.unique(y))
cm_df = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))

plt.figure(figsize=(6, 5))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – k-NN")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()

# Save FIRST, then show or close
plt.savefig("confusion_matrix_knn.png", dpi=150)
plt.show()  # or: plt.close()

print("\nDone. You can screenshot the plots and add them to your README for GitHub.")