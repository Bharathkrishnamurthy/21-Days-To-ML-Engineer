
# BHARATH BUILDS
## 21 Days to ML Engineer

# Day 13: Gradient Boosting + XGBoost

A practical machine learning experiment focused on Gradient Boosting, XGBoost, hyperparameter tuning, model evaluation, and comparison against a Random Forest baseline.

## 1. Project Overview

Day 13 focuses on understanding how boosting algorithms build predictive models sequentially, how XGBoost incorporates regularization and optimization techniques, and how to select a model using validation evidence and held-out test performance.

The experiment compares three classification models on the same dataset:

- Random Forest
- Gradient Boosting
- XGBoost

The objective is not simply to achieve the highest accuracy. It is to understand model behavior, compare evaluation metrics, identify potential overfitting, and select a suitable model based on the requirements of the problem.

## 2. Objectives

- Understand sequential error correction in boosting.
- Explain the Gradient Boosting algorithm and negative gradients.
- Understand the relationship between learning rate and number of estimators.
- Study XGBoost's objective function and regularization.
- Understand important boosted-tree hyperparameters.
- Identify potential overfitting using training and validation performance.
- Apply stratified cross-validation and hyperparameter tuning.
- Compare Random Forest, Gradient Boosting, and XGBoost.
- Evaluate classification performance using appropriate metrics.
- Document and export reproducible experimental results.

## 3. Dataset

**Dataset:** Breast Cancer Wisconsin Dataset  
**Source:** Scikit-learn built-in dataset  
**Task:** Binary classification

The dataset contains numerical features computed from digitized images of breast mass samples.

### Target Classes

| Target Label | Class |
|---|---|
| 0 | Malignant |
| 1 | Benign |

The target encoding is important when interpreting precision, recall, F1-score, and class-specific errors.

For medical classification, malignant-class recall and precision deserve particular attention because different types of prediction errors have different implications.

## 4. Technologies and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Google Colab
- Git and GitHub

## 5. Theoretical Concepts Covered

### 5.1. Gradient Boosting

- Boosting intuition and sequential error correction
- Loss functions and negative gradients
- Residuals in squared-error regression
- Additive model updates
- Learning rate and number of boosting stages
- Learning rate versus number of estimators

### 5.2. XGBoost

- Overview of Extreme Gradient Boosting
- First-order gradient and second-order information
- Objective function: prediction loss plus regularization penalty
- Tree construction and split evaluation
- Native missing-value handling

### 5.3. Regularization and Overfitting Control

| Parameter | Purpose |
|---|---|
| `learning_rate` | Controls each tree's contribution |
| `n_estimators` | Specifies the number of boosting stages |
| `max_depth` | Controls maximum tree depth |
| `min_child_weight` | Controls the minimum summed Hessian required in a child |
| `gamma` | Specifies the minimum loss reduction required for a split |
| `reg_alpha` | Controls L1 regularization |
| `reg_lambda` | Controls L2 regularization |
| `subsample` | Controls the fraction of training rows sampled |
| `colsample_bytree` | Controls the fraction of features sampled per tree |

Other topics include early stopping, underfitting, overfitting, validation-based model selection, and test-set leakage.

## 6. Practical Implementation Workflow

The experiment follows a reproducible model-selection process.

1. Install and import the required libraries.
2. Load the Breast Cancer Wisconsin dataset.
3. Inspect the dataset shape and target distribution.
4. Split the dataset into training and held-out test sets.
5. Configure stratified five-fold cross-validation.
6. Train and tune a Random Forest baseline.
7. Train and tune a Gradient Boosting classifier.
8. Train and tune an XGBoost classifier.
9. Compare cross-validation training and validation ROC-AUC.
10. Visualize model comparison results.
11. Evaluate the selected configurations on the held-out test set.
12. Inspect class-specific classification metrics.
13. Plot and interpret the confusion matrix.
14. Export comparison tables as CSV files.
15. Document findings, limitations, and conclusions.

## 7. Model Selection Strategy

| Component | Approach |
|---|---|
| Baseline | Random Forest |
| Boosting model 1 | Gradient Boosting |
| Boosting model 2 | XGBoost |
| Validation strategy | Stratified 5-fold cross-validation |
| Selection metric | ROC-AUC |
| Final evaluation | Held-out test set |

Hyperparameter tuning is performed using training data and cross-validation. The held-out test set is reserved for final evaluation.

## 8. Day 13 — Results Interpretation

The following results are from the completed experiment.

### 8.1. Overall Model Performance

| Metric | Random Forest | Gradient Boosting | XGBoost |
|---|---:|---:|---:|
| Test Accuracy | 95.61% | 94.74% | 95.61% |
| Test ROC-AUC | See experiment output | See experiment output | 0.9660 |

The supplied results establish that Random Forest and XGBoost achieved the same accuracy, while Gradient Boosting achieved slightly lower accuracy.

XGBoost achieved the highest reported ROC-AUC of **0.9660**, indicating strong ranking performance in this experiment.

The exact ROC-AUC values for Random Forest and Gradient Boosting should be taken from the saved test comparison table rather than inferred from accuracy.

### 8.2. Class-Specific Performance

| Observation | Result |
|---|---|
| Benign-class recall | XGBoost achieved 98.6% |
| Malignant-class recall | Random Forest achieved 92.9% |
| Malignant-class recall for boosting models | Gradient Boosting and XGBoost achieved 90.5% |
| Malignant-class precision | XGBoost achieved 97.4% |

These results highlight that different models can perform better on different metrics.

Random Forest identified a slightly larger proportion of malignant cases than the two boosting models in this experiment.

XGBoost achieved the highest reported malignant precision, indicating strong precision for its malignant predictions.

### 8.3. Random Forest Interpretation

Random Forest achieved 95.61% accuracy and performed consistently.

Its malignant-class recall was 92.9%, slightly higher than the 90.5% achieved by both boosting models.

This makes Random Forest an important candidate when identifying malignant cases is a priority.

### 8.4. Gradient Boosting Interpretation

Gradient Boosting achieved 94.74% accuracy, slightly below the accuracy of Random Forest and XGBoost.

Its malignant-class recall was 90.5%.

Although its reported accuracy was lower, its overall suitability should be judged using the complete metric profile and the requirements of the prediction task.

### 8.5. XGBoost Interpretation

XGBoost achieved 95.61% accuracy and the highest reported ROC-AUC of 0.9660.

It also achieved 98.6% benign-class recall and 97.4% malignant-class precision.

These results make XGBoost a strong overall candidate in this experiment, particularly when ranking performance and malignant precision are important.

However, its malignant-class recall was lower than Random Forest's, so it is not the winner on every metric.

### 8.6. Final Comparison

The results support three conclusions:

1. Random Forest and XGBoost tied on test accuracy at 95.61%.
2. XGBoost achieved the highest reported ROC-AUC and malignant precision.
3. Random Forest achieved the highest reported malignant-class recall.

**Key takeaway:** No single model wins every metric. Model selection should reflect the problem's priorities rather than relying on accuracy alone.

These results are from a machine learning experiment and should not be interpreted as evidence of clinical readiness or suitability for real-world medical diagnosis.

## 9. Evaluation Metrics

### Accuracy

Measures the proportion of all predictions that are correct.

### Precision

Measures the proportion of predicted positive cases that are actually positive.

### Recall

Measures the proportion of actual positive cases correctly identified.

### F1-Score

The harmonic mean of precision and recall.

### ROC-AUC

Measures how effectively a model ranks positive examples above negative examples across classification thresholds.

For every class-specific metric, the positive label must be clearly defined. In this dataset, label `0` represents malignant and label `1` represents benign.

## 10. Engineering Principles Applied

The experiment follows these principles:

- Use the same training and test split for fair comparison.
- Use cross-validation for hyperparameter selection.
- Keep the test set separate from model-tuning decisions.
- Compare multiple metrics rather than accuracy alone.
- Examine class-specific errors and confusion matrices.
- Record best parameters and experimental results.
- Interpret model performance in the context of the problem.
- Avoid claiming that small metric differences are necessarily meaningful.

## 11. Expected Project Deliverables

- Google Colab notebook containing the experiment
- Random Forest, Gradient Boosting, and XGBoost comparison
- Cross-validation comparison table
- Training-versus-validation performance analysis
- Final test metrics
- Classification reports
- Confusion matrix visualization
- CSV files containing experimental results
- README documentation and conclusions

## 12. Repository Structure

A suggested repository structure is:

```text
day-13-gradient-boosting-xgboost/
├── README.md
├── Day13_Gradient_Boosting_XGBoost_Konglish.ipynb
├── day13_cv_comparison.csv
└── day13_test_comparison.csv
```

The notebook and CSV files should be included only if they are present in the repository.

## 13. Future Improvements

- Experiment with additional learning rates and estimator counts.
- Explore XGBoost regularization parameters.
- Compare validation performance across multiple parameter configurations.
- Analyze class-specific false positives and false negatives.
- Examine feature importance and model interpretation.
- Evaluate the stability of results across different splits or repeated cross-validation.
- Consider computational cost alongside predictive performance.

## 14. Conclusion

Day 13 demonstrates how Gradient Boosting and XGBoost can be implemented and evaluated alongside a Random Forest baseline.

The experiment reinforces the importance of sequential learning, learning-rate selection, regularization, cross-validation, class-specific evaluation, and final test-set assessment.

The main lesson is that selecting a machine learning model requires more than comparing accuracy. A reliable decision considers validation evidence, class-specific errors, ranking performance, generalization, and the practical requirements of the problem.

## 15. Learning Workflow

**Understand → Implement → Evaluate → Compare → Interpret → Document → Commit**

---

# BHARATH BUILDS
### 21 Days to ML Engineer

Day 13: Gradient Boosting + XGBoost

Practical Machine Learning | Model Evaluation | Reproducible Experiments

