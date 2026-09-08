"""
Project 2: Data Classification Using AI
DecodeLabs Industrial Training Kit — Batch 2026

Goal: Build a basic classification model using a small dataset
(the classic Iris dataset), following the full supervised
learning pipeline: Input -> Process -> Output.
"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# PHASE 1: INPUT — Load and understand the dataset
# The Iris dataset: 150 samples, 3 classes (setosa, versicolor, virginica),
# 4 features (sepal length/width, petal length/width) — see "Raw Material:
# The Iris Benchmark" slide.
# ---------------------------------------------------------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)   # the 4 features
y = pd.Series(iris.target, name="species")                # the 3 classes (0, 1, 2)

print("=" * 60)
print("PHASE 1: DATASET OVERVIEW")
print("=" * 60)
print(f"Samples: {X.shape[0]}")
print(f"Features: {X.shape[1]} -> {list(X.columns)}")
print(f"Classes: {list(iris.target_names)}")
print("\nFirst 5 rows:")
print(X.head())
print("\nClass distribution:")
print(y.value_counts().rename(index=dict(enumerate(iris.target_names))))


# ---------------------------------------------------------------------------
# PHASE 2: PROCESS — Split into training/testing sets
# We shuffle before splitting to remove any order bias in the raw data
# (see "Structural Integrity: The Split" slide). 80% train / 20% test,
# stratified so each class is proportionally represented in both sets.
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,   # fixed seed -> reproducible results
    stratify=y         # keeps class balance consistent across the split
)

print("\n" + "=" * 60)
print("PHASE 2: TRAIN/TEST SPLIT")
print("=" * 60)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ---------------------------------------------------------------------------
# PHASE 2b: The Gatekeeper Rule — Feature Scaling
# KNN is distance-based, so features on different scales (e.g. cm vs cm here,
# but in general very different ranges) can bias the "closeness" calculation.
# StandardScaler transforms each feature to mean=0, variance=1.
# Important: fit the scaler on TRAINING data only, then apply it to test data
# — this avoids leaking information from the test set into training.
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ---------------------------------------------------------------------------
# PHASE 2c: The Algorithm — K-Nearest Neighbors (KNN)
# "The Proximity Principle: similar things exist in close proximity."
# For a new data point, KNN looks at its 'k' closest neighbors in the
# training data and assigns the majority class among them.
# ---------------------------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=5)   # k=5, a common default
model.fit(X_train_scaled, y_train)            # FIT: memorize the training data
predictions = model.predict(X_test_scaled)    # PREDICT: classify unseen test data

print("\n" + "=" * 60)
print("PHASE 2: MODEL TRAINED (KNN, k=5)")
print("=" * 60)


# ---------------------------------------------------------------------------
# PHASE 3: OUTPUT — Evaluate the model
# Accuracy alone can be misleading ("Accuracy Mirage" slide) — especially
# on imbalanced datasets. We use a confusion matrix plus precision/recall/F1
# for a fuller picture. The Iris dataset is balanced, but it's good practice
# to always check beyond plain accuracy.
# ---------------------------------------------------------------------------
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print("\n" + "=" * 60)
print("PHASE 3: OUTPUT VALIDATION")
print("=" * 60)
print(f"Accuracy: {accuracy:.2%}")
print(f"F1 Score (macro avg): {f1:.2%}")

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

print("\nFull Classification Report:")
print(classification_report(y_test, predictions, target_names=iris.target_names))


# ---------------------------------------------------------------------------
# BONUS: Visualize the confusion matrix
# ---------------------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=iris.target_names, yticklabels=iris.target_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix — Iris Classification (KNN)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix chart saved as 'confusion_matrix.png'")


# ---------------------------------------------------------------------------
# BONUS: Tuning K — find the best value of k (see "Tuning the Engine" slide)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("BONUS: FINDING THE OPTIMAL K")
print("=" * 60)
error_rates = []
k_range = range(1, 21)
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    pred_k = knn.predict(X_test_scaled)
    error_rates.append(1 - accuracy_score(y_test, pred_k))

best_k = k_range[error_rates.index(min(error_rates))]
print(f"Best k found: {best_k} (lowest error rate: {min(error_rates):.2%})")

plt.figure(figsize=(8, 5))
plt.plot(k_range, error_rates, marker="o", linestyle="--")
plt.xlabel("K Value")
plt.ylabel("Error Rate")
plt.title("Error Rate vs. K Value")
plt.axvline(best_k, color="red", linestyle=":", label=f"Best k = {best_k}")
plt.legend()
plt.tight_layout()
plt.savefig("k_tuning.png")
print("K-tuning chart saved as 'k_tuning.png'")
