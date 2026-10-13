# Day 9: Support Vector Machine (SVM)

**21 Days to ML Engineer | Bharath Builds**

## Overview
- Implemented Support Vector Machine (SVM) for binary classification.
- Dataset: Breast Cancer Wisconsin dataset from Scikit-learn.
- Objective: Compare Linear SVM, RBF SVM, and tuned RBF SVM.

## Key Concepts Covered
- **Hyperplane:** Boundary separating two classes.
- **Margin:** Distance between the decision boundary and nearest data points.
- **Support Vectors:** Data points that determine the decision boundary.
- **Soft Margin and C:** Controls the trade-off between margin size and classification errors.
- **Kernel Trick:** Helps SVM learn nonlinear decision boundaries.
- **Linear Kernel:** Suitable for approximately linearly separable data.
- **RBF Kernel:** Handles nonlinear patterns.
- **Gamma:** Controls the influence of individual data points in the RBF kernel.
- **Feature Scaling:** Standardization is important because SVM is sensitive to feature scales.
- **Limitations:** Computational cost can increase with larger datasets; parameter selection affects performance.

## Practical Pipeline
1. Load the Breast Cancer dataset.
2. Separate features and target labels.
3. Split data into training and testing sets using stratification.
4. Apply `StandardScaler` using an ML pipeline.
5. Train Linear SVM and RBF SVM models.
6. Compare different `C` values and tune RBF parameters using cross-validation.
7. Evaluate accuracy, malignant precision, malignant recall, F1-scores, and confusion matrix.
8. Inspect misclassified samples and interpret results.

## Model Comparison

| Model | Accuracy | F1 (Malignant) | F1 (Benign) |
|---|---:|---:|---:|
| Linear SVM (C=1) | 97.37% | 96.47% | 97.90% |
| RBF SVM (C=1) | 98.25% | 97.62% | 98.61% |
| Tuned RBF SVM | 98.25% | 97.62% | 98.61% |

## Results and Interpretation
- RBF SVM achieved the highest reported accuracy: **98.25%**.
- Both RBF models achieved **97.62% malignant recall**.
- RBF performed better than Linear SVM on the reported metrics.
- Hyperparameter tuning did not improve the reported test metrics.
- Check the confusion matrix, especially false negatives, when evaluating malignant cases.
- Results are from a test split and should not be treated as clinical validation.

## Tech Stack
- Python
- Google Colab / Jupyter Notebook
- NumPy and Pandas
- Matplotlib
- Scikit-learn

## Key Learning
**Understand the algorithm → Implement the model → Evaluate performance → Interpret errors → Document results.**

## Day 9 Deliverable
SVM comparison notebook covering Linear SVM, RBF SVM, feature scaling, hyperparameter tuning, and model evaluation.
