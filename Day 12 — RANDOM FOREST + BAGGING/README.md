# Day 12 — Random Forest and Bagging

## 21 Days to ML Engineer | Bharath Builds

Welcome to Day 12 of the **21 Days to ML Engineer** series by Bharath Builds.

In this session, I explored how ensemble learning can improve machine learning predictions by combining multiple models instead of relying on a single decision tree.

The main focus was understanding Bagging, Bootstrap Sampling, and Random Forest, followed by implementing these concepts using Python and Scikit-learn.

The objective was to understand not only how to train a model, but also how to compare models, interpret predictions, and evaluate performance using an appropriate dataset.

## Learning Objectives

- Understand the fundamentals of ensemble learning.
- Learn how Bagging uses bootstrap samples to train multiple models.
- Understand how Random Forest combines bootstrap sampling with random feature selection.
- Compare a single Decision Tree with a Random Forest model.
- Explore Out-of-Bag (OOB) evaluation and feature importance.
- Tune important hyperparameters and evaluate model performance.
- Interpret evaluation metrics and understand model behavior.

## Theoretical Concepts Covered

### 1. Ensemble Learning
Understanding how combining multiple models can improve prediction performance, stability, and generalization compared with relying on a single model.

### 2. Bagging — Bootstrap Aggregation
Learning how multiple training samples are generated through random sampling with replacement, allowing different models to learn from different versions of the training data.

### 3. Bootstrap Sampling
Understanding how observations may appear multiple times in a bootstrap sample while others may be left out.

### 4. Random Forest
Exploring how Random Forest builds multiple decision trees and combines their predictions to produce a final result.

### 5. Random Feature Selection
Understanding how Random Forest considers a random subset of features when building trees, helping introduce diversity among the trees.

### 6. Out-of-Bag Evaluation
Learning how observations excluded from individual bootstrap samples can help estimate model performance.

### 7. Feature Importance
Understanding how feature importance helps identify influential features, along with its limitations and the need for careful interpretation.

## Practical Implementation

The concepts were implemented using Python, Pandas, Matplotlib, and Scikit-learn in Google Colab.

### Datasets Used

Two datasets were considered during the practical implementation.

**1. Wine Quality — White Wine Dataset**

The Wine Quality dataset was initially explored to demonstrate classification using physicochemical properties such as acidity, residual sugar, chlorides, density, pH, sulphates, and alcohol.

The original quality score was converted into a binary target, where scores greater than or equal to 6 represented good quality and scores below 6 represented lower quality.

However, the resulting target distribution in the dataset used for the experiment did not provide the intended representation of both classes. The observations fell into the below-6 category, leaving no samples in the above-or-equal-to-6 category. This made the dataset unsuitable for the intended binary classification demonstration in that experiment.

**2. Breast Cancer Dataset**

The Breast Cancer dataset was subsequently used for the main classification demonstration because it provided both target classes for model training and evaluation.

This made it more suitable for demonstrating the Decision Tree and Random Forest workflows, comparing classification metrics, and explaining model performance.

The dataset was used to demonstrate the complete workflow, including data splitting, model training, hyperparameter tuning, feature importance, and final evaluation.

### Practical Steps Followed

1. **Load the dataset:** Imported the selected dataset and separated input features from the target variable.
2. **Perform basic data checks:** Reviewed the dataset structure, missing values, and target-class distribution.
3. **Split the data:** Created training, validation, and test sets while preserving the class distribution.
4. **Build a Decision Tree baseline:** Trained a single Decision Tree to establish a baseline for comparison.
5. **Evaluate the baseline:** Measured accuracy, precision, recall, F1-score, and ROC-AUC where applicable.
6. **Demonstrate bootstrap sampling:** Explored how sampling with replacement creates different training samples.
7. **Train a Random Forest:** Built an ensemble of decision trees using bootstrap sampling and random feature selection.
8. **Compare model performance:** Evaluated the Decision Tree and Random Forest using validation metrics.
9. **Tune hyperparameters:** Explored parameters such as `n_estimators`, `max_depth`, and `max_features` using cross-validation.
10. **Inspect feature importance:** Compared built-in feature importance with permutation importance.
11. **Perform error analysis:** Examined the confusion matrix and prediction errors to understand model behavior.
12. **Evaluate the final model:** Assessed the selected model on the held-out test set.
13. **Save the results:** Exported evaluation tables and analysis results for documentation and portfolio use.

## Final Model Evaluation

The final evaluation metrics should correspond to the dataset and model used for the final test evaluation.

| Metric | Score |
|---|---:|
| Accuracy | 90.70% |
| Precision | 89.66% |
| Recall | 96.30% |
| F1-score | 92.86% |
| ROC-AUC | 98.21% |

### Interpretation

- **Accuracy — 90.70%:** The model correctly classified approximately 91% of the test samples.
- **Precision — 89.66%:** Of the samples predicted as the positive class, approximately 90% actually belonged to that class.
- **Recall — 96.30%:** The model identified a high proportion of the actual positive-class samples.
- **F1-score — 92.86%:** The model maintained a strong balance between precision and recall.
- **ROC-AUC — 98.21%:** The model demonstrated excellent ability to distinguish between the two classes.

These results indicate strong performance on the evaluated test set. They should not be interpreted as a guarantee of equivalent performance on other datasets.

## Tools and Technologies

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Wine Quality Dataset
- Breast Cancer Dataset
- Git and GitHub

## Suggested Repository Structure

```text
day-12-random-forest-bagging/
│
├── day12_random_forest_bagging.ipynb
├── README.md
├── results/
│   ├── model_comparison.csv
│   ├── validation_results.csv
│   ├── feature_importance.csv
│   └── test_results.csv
└── images/
    └── model_comparison.png
```

Include only the files and folders that are actually present in the repository.

## Key Takeaways

- A single Decision Tree provides a useful baseline but may be sensitive to changes in training data.
- Bagging trains multiple models on different bootstrap samples and combines their predictions.
- Random Forest adds random feature selection to introduce diversity among trees.
- Hyperparameter tuning helps identify a suitable model configuration.
- Feature importance can provide useful insights but does not establish causation.
- A held-out test set should be reserved for final evaluation after model selection and tuning are complete.
- Dataset selection and target-class distribution are important considerations when building a classification workflow.

## What I Learned

This session helped me move beyond training individual machine learning models and understand how ensemble methods work in practice.

I explored how to build a Random Forest workflow, compare it with a Decision Tree, tune model settings, investigate feature importance, and interpret evaluation metrics. I also learned why checking the target distribution is an important step before selecting a dataset for classification.

The emphasis was on understanding the reasoning behind each step rather than simply executing the code.

## About the Series

**21 Days to ML Engineer — Bharath Builds**

This series focuses on learning machine learning through practical implementation, clear explanations, interview preparation, and portfolio-oriented projects.

The learning approach is:

**EXPLAIN → IMPLEMENT → INTERPRET → DOCUMENT → COMMIT**

YouTube: https://youtube.com/@bharathk-0406
