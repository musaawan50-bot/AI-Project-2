# Data Classification Using AI (KNN + Iris Dataset)

A supervised learning classification model built in Python as Project 2 of the DecodeLabs Industrial Training Kit (AI Track, Batch 2026).

## Overview

This project builds a classification model using the classic Iris dataset, following the full supervised learning pipeline: loading and understanding data, splitting it into training/testing sets, scaling features, training a K-Nearest Neighbors (KNN) model, and evaluating its performance with proper metrics beyond plain accuracy.

## Features

- **Dataset overview** — loads the Iris dataset (150 samples, 3 classes, 4 features) and prints a summary
- **Train/test split** — 80/20 stratified split with shuffling to remove order bias
- **Feature scaling** — StandardScaler applied (fit on training data only, to avoid data leakage)
- **KNN classification** — trains a K-Nearest Neighbors model (k=5) using scikit-learn
- **Full evaluation** — accuracy, F1 score, confusion matrix, and a full precision/recall/F1 classification report (not just raw accuracy)
- **Confusion matrix visualization** — saved as a heatmap image
- **K-tuning experiment** — tests k=1 through 20 to find the optimal K value, with a saved error-rate chart

## How to Run

Requires Python 3 with the following packages:

```bash
pip3 install pandas scikit-learn matplotlib seaborn
```

Then run:

```bash
python3 classification.py
```

## Example Output

```
Accuracy: 93.33%
F1 Score (macro avg): 93.27%

Confusion Matrix:
[[10  0  0]
 [ 0 10  0]
 [ 0  2  8]]
```

Two chart images are also generated:
- `confusion_matrix.png` — visual breakdown of correct vs incorrect predictions per class
- `k_tuning.png` — error rate across different K values, with the optimal K marked

## Project Structure

```
.
├── classification.py     # Main script — full ML pipeline
├── confusion_matrix.png  # Generated after running
├── k_tuning.png           # Generated after running
└── README.md              # This file
```

## Concepts Demonstrated

- Supervised learning fundamentals (train/test split, fit/predict workflow)
- Feature scaling and why it matters for distance-based algorithms like KNN
- The K-Nearest Neighbors algorithm and the "proximity principle"
- Why accuracy alone can be misleading, and how confusion matrices and F1 score give a fuller picture
- Hyperparameter tuning (finding the optimal K)

## Author

Musa — DecodeLabs AI Engineering Track, Batch 2026
