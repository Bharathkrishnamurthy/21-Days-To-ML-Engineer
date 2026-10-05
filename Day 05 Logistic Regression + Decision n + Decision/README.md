# 🚀 BHARATH BUILDS
# 21 DAYS TO ML ENGINEER — DAY 5

## Logistic Regression + Decision Threshold

> **From Probability → Prediction → Decision**

---

## 📌 Day 5 Objective

- Understand **Classification vs Regression**
- Understand **Binary Classification**
- Learn how **Logistic Regression** works
- Understand the **Sigmoid Function**
- Generate probability predictions using `predict_proba()`
- Understand **Decision Threshold**
- Compare different thresholds: `0.3`, `0.5`, `0.7`
- Understand **Confusion Matrix**
- Analyze **TP, TN, FP, FN**
- Evaluate **Precision, Recall, and Accuracy**
- Perform practical **Threshold Experimentation**
- Understand how threshold selection affects model performance

---

# 🧠 1. Classification vs Regression

| Regression | Classification |
|---|---|
| Predicts continuous values | Predicts classes |
| Output is a number | Output is a category |
| Example: House Price | Example: Disease / No Disease |
| Metrics: MAE, MSE, RMSE, R² | Metrics: Accuracy, Precision, Recall |

### Examples

**Regression**
```text
House Price → ₹65,00,000
Temperature → 28.5°C
Salary → ₹8.5 LPA
```

**Classification**
```text
Disease → 1
No Disease → 0
```

---

# 🎯 2. Binary Classification

Binary classification contains **two possible classes**.

```text
Class 0 → Negative
Class 1 → Positive
```

### Example

```text
0 → No Disease
1 → Disease
```

Other examples:

- Spam / Not Spam
- Fraud / Not Fraud
- Pass / Fail
- Customer Churn / No Churn

---

# 📈 3. Logistic Regression

Logistic Regression is a classification algorithm used mainly for predicting the probability of a class.

### Basic Flow

```text
Input Features
      ↓
Linear Combination
      ↓
Sigmoid Function
      ↓
Probability
      ↓
Decision Threshold
      ↓
Final Class
```

### Linear Equation

```text
z = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ
```

---

# 🔄 4. Sigmoid Function

The sigmoid function converts any real-valued number into a value between `0` and `1`.

### Formula

```text
σ(z) = 1 / (1 + e⁻ᶻ)
```

### Interpretation

```text
z → -∞     → Probability ≈ 0
z = 0      → Probability = 0.5
z → +∞     → Probability ≈ 1
```

### Simple Flow

```text
Linear Score
     ↓
 Sigmoid
     ↓
Probability
  0 → 1
```

---

# 🎲 5. Probability Output

Logistic Regression does not directly start with `0` or `1`.

It first produces a probability.

Example:

```text
Prediction Probability = 0.82
```

Meaning:

```text
82% probability of Class 1
```

This probability is then converted into a class using a threshold.

---

# ⚖️ 6. Decision Threshold

The threshold determines when a probability becomes Class 1.

### Default Threshold

```text
If probability ≥ 0.5
    → Class 1

If probability < 0.5
    → Class 0
```

### Example

```text
Probability = 0.82
Threshold  = 0.50

0.82 ≥ 0.50
→ Class 1
```

---

# 🎚️ 7. Threshold Experiment

Instead of using only `0.5`, we tested:

```text
Threshold = 0.3
Threshold = 0.5
Threshold = 0.7
```

### Threshold Effect

```text
Lower Threshold
      ↓
More Positive Predictions
      ↓
Higher Recall
      ↓
Potentially More False Positives
```

```text
Higher Threshold
      ↓
Fewer Positive Predictions
      ↓
Potentially Higher Precision
      ↓
Potentially More False Negatives
```

---

# ❌ 8. Confusion Matrix

A confusion matrix shows how the classification model makes mistakes.

```text
                  Predicted
                0          1

Actual 0       TN         FP
Actual 1       FN         TP
```

### True Positive — TP

```text
Actual = 1
Predicted = 1
```

Correctly predicted positive case.

### True Negative — TN

```text
Actual = 0
Predicted = 0
```

Correctly predicted negative case.

### False Positive — FP

```text
Actual = 0
Predicted = 1
```

Model predicted positive when the actual class was negative.

### False Negative — FN

```text
Actual = 1
Predicted = 0
```

Model predicted negative when the actual class was positive.

---

# 📊 9. Classification Metrics

### Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Measures overall correct predictions.

### Precision

```text
Precision = TP / (TP + FP)
```

Among predicted positives, how many were actually positive?

### Recall

```text
Recall = TP / (TP + FN)
```

Among actual positives, how many did the model correctly identify?

---

# 🔍 10. `predict()` vs `predict_proba()`

### `predict()`

Returns the final class.

```python
model.predict(X_test)
```

Example:

```text
[0, 1, 1, 0, 1]
```

### `predict_proba()`

Returns probability for each class.

```python
model.predict_proba(X_test)
```

Example:

```text
[[0.92, 0.08],
 [0.15, 0.85],
 [0.20, 0.80]]
```

For Class 1:

```python
model.predict_proba(X_test)[:, 1]
```

---

# 💻 11. Practical Implementation

## Dataset

**Breast Cancer Wisconsin Dataset**

Source:

```python
from sklearn.datasets import load_breast_cancer
```

### Dataset Goal

```text
Predict whether the tumor is:

0 → Malignant
1 → Benign
```

---

## Practical Workflow

```text
Load Dataset
     ↓
Separate X and y
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Build Logistic Regression Pipeline
     ↓
Train Model
     ↓
Generate Predictions
     ↓
Generate Probabilities
     ↓
Test Different Thresholds
     ↓
Confusion Matrix
     ↓
Evaluate Metrics
     ↓
Analyze Errors
```

---

# 🧪 12. Implementation Steps

### Step 1 — Import Libraries

```python
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
```

### Step 2 — Load Dataset

```python
data = load_breast_cancer()

X = data.data
y = data.target
```

### Step 3 — Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Step 4 — Build Pipeline

```python
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])
```

### Step 5 — Train Model

```python
model.fit(X_train, y_train)
```

### Step 6 — Generate Predictions

```python
y_pred = model.predict(X_test)
```

### Step 7 — Generate Probabilities

```python
y_proba = model.predict_proba(X_test)[:, 1]
```

---

# 📊 13. Threshold Experiment Results

| Threshold | Accuracy | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|
| **0.3** | 98.25% | 97.30% | **100%** | 2 | **0** |
| **0.5** | **98.25%** | **98.61%** | **98.61%** | **1** | 1 |
| **0.7** | 94.74% | 98.53% | 93.06% | 1 | 5 |

---

# 🔎 14. Threshold = 0.3

### Results

```text
Accuracy  → 98.25%
Precision → 97.30%
Recall    → 100%
FP        → 2
FN        → 0
```

### Interpretation

- Lower threshold makes the model more willing to predict Class 1.
- Every actual positive case was detected.
- `FN = 0`
- Recall reached **100%**.
- However, False Positives increased to `2`.
- Useful when missing a positive case is very costly.

---

# ⚖️ 15. Threshold = 0.5

### Results

```text
Accuracy  → 98.25%
Precision → 98.61%
Recall    → 98.61%
FP        → 1
FN        → 1
```

### Interpretation

- Provides a strong balance between Precision and Recall.
- Only `1` False Positive and `1` False Negative.
- Precision and Recall are both **98.61%**.
- This produced the most balanced error pattern among the tested thresholds.

---

# 🔒 16. Threshold = 0.7

### Results

```text
Accuracy  → 94.74%
Precision → 98.53%
Recall    → 93.06%
FP        → 1
FN        → 5
```

### Interpretation

- Higher threshold makes the model more strict about predicting Class 1.
- False Positives remained low.
- False Negatives increased from `1` to `5`.
- Recall dropped to **93.06%**.
- Overall accuracy also dropped to **94.74%**.

---

# 🧠 17. Practical Interpretation

### What changed when the threshold changed?

```text
Threshold ↓
     ↓
More Positive Predictions
     ↓
FN ↓
Recall ↑
FP can ↑
```

```text
Threshold ↑
     ↓
Fewer Positive Predictions
     ↓
FP can ↓
FN ↑
Recall ↓
```

The important point:

> **Changing the threshold changes the model's decision behavior without retraining the model.**

---

# 🚨 18. Why Accuracy Alone Is Not Enough

Threshold `0.3`:

```text
Accuracy = 98.25%
```

Threshold `0.5`:

```text
Accuracy = 98.25%
```

Same accuracy.

But:

```text
Threshold 0.3 → FN = 0
Threshold 0.5 → FN = 1
```

So, the error pattern is different even though the accuracy is identical.

### Key Point

**Always inspect the type of errors, not just the overall accuracy.**

---

# 🏗️ 19. ML Engineering Perspective

A classification system is not only:

```text
Train Model
```

A more realistic workflow is:

```text
Train Model
     ↓
Generate Probabilities
     ↓
Choose Threshold
     ↓
Evaluate Metrics
     ↓
Analyze FP / FN
     ↓
Consider Business Cost
     ↓
Select Final Threshold
```

### Threshold Selection Depends On:

- Cost of False Positives
- Cost of False Negatives
- Business requirements
- Safety requirements
- Domain requirements
- Desired Precision / Recall

---

# 🎯 20. Key Learnings

- Logistic Regression is a **classification algorithm**.
- Sigmoid converts the linear output into a probability.
- `predict_proba()` provides probability scores.
- `predict()` converts probabilities into final classes.
- The default threshold is usually `0.5`.
- Thresholds can be changed without retraining the model.
- Lower thresholds generally favor **Recall**.
- Higher thresholds can favor **Precision**, but may increase False Negatives.
- Confusion Matrix helps identify the exact types of errors.
- Accuracy alone does not tell the complete story.
- Threshold selection is an important **ML engineering decision**.

---

# 🚀 Final Takeaway

> **The model produces the probability.  
> The threshold converts probability into a decision.  
> The best threshold depends on the real-world cost of False Positives and False Negatives.**

---



### Portfolio Deliverable

**A Logistic Regression classification notebook with probability prediction, threshold experimentation, confusion matrix analysis, and error interpretation.**

---

# 🔜 Next Day

## DAY 6 — Decision Trees

**From Probability-Based Decisions → Rule-Based Decisions**

Topics:

- Decision Tree fundamentals
- Gini Impurity
- Entropy
- Information Gain
- Tree splitting
- Overfitting in Decision Trees
- `max_depth`
- Practical model implementation
- Model interpretation
- Interview questions
- GitHub portfolio update
