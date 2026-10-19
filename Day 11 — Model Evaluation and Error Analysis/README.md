# Day 11 — Model Evaluation and Error Analysis

**21 Days to ML Engineer | Bharath Builds**

## Overview

Building a machine learning model is only one part of the process. We also need to understand how well it performs, where it makes mistakes, and whether its predictions are reliable for the problem we are trying to solve.

In Day 11 of the *21 Days to ML Engineer* series, I explored model evaluation and error analysis using a **Titanic Survival Prediction** dataset. The goal was to evaluate multiple classification models, compare their performance using different metrics, and understand how decision thresholds affect predictions.

## Learning Objectives

- Understand the confusion matrix and its four outcomes: TP, TN, FP, and FN.
- Evaluate classification models using accuracy, precision, recall, F1-score, and ROC-AUC.
- Understand the difference between false positives and false negatives.
- Compare multiple machine learning models using cross-validation and validation metrics.
- Explore how changing the decision threshold affects model predictions.
- Analyze incorrect predictions to identify potential weaknesses.
- Understand why accuracy alone is not enough to evaluate a classification model.

## Dataset

The project uses the Titanic dataset to predict whether a passenger survived.

Selected features include:

- Passenger class (`pclass`)
- Sex (`sex`)
- Age (`age`)
- Number of siblings or spouses aboard (`sibsp`)
- Number of parents or children aboard (`parch`)
- Ticket fare (`fare`)
- Port of embarkation (`embarked`)

**Target variable:** `survived`

The dataset is loaded using Scikit-learn's OpenML interface.

## Machine Learning Models

The following classification models were evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Support Vector Machine (SVM)

The workflow uses preprocessing pipelines to handle missing values, scale numerical features, and encode categorical features.

## Project Workflow

1. Load the Titanic dataset and select relevant features.
2. Separate the input features and target variable.
3. Split the data into training, validation, and test sets.
4. Build preprocessing pipelines for numerical and categorical data.
5. Train and compare multiple classification models.
6. Evaluate models using stratified cross-validation.
7. Select a model using validation recall.
8. Evaluate the selected model on the held-out test set.
9. Compare classification metrics and ROC-AUC.
10. Examine decision thresholds and perform error analysis.

## Final Evaluation Results

The K-Nearest Neighbors (KNN) model was selected based on its highest recall on the validation set.

| Metric | Test Result |
|---|---:|
| Accuracy | 80.2% |
| Precision | 75.0% |
| Recall | 72.0% |
| F1-score | 73.5% |
| ROC-AUC | 0.741 |

### Results Interpretation

- **Accuracy — 80.2%:** The model correctly classified approximately 80% of passengers in the test set.
- **Precision — 75.0%:** Of the passengers predicted to survive, 75% actually survived.
- **Recall — 72.0%:** The model identified 72% of passengers who actually survived.
- **F1-score — 73.5%:** This summarizes the balance between precision and recall.
- **ROC-AUC — 0.741:** The score indicates a moderate ability to distinguish between survivors and non-survivors.

## Decision Threshold Analysis

The default classification threshold is 0.5. In this experiment, a threshold of **0.429** was selected using validation data to maximize the F1-score.

Changing the threshold can affect the balance between precision and recall. The validation set is used for threshold selection so that the held-out test set remains available for final evaluation.

The reported count of 52 incorrect predictions was obtained at the default threshold of 0.5. It should not be interpreted as the error count at the selected threshold of 0.429.

## Error Analysis

Error analysis helps us understand where the model's predictions differ from the actual outcomes.

The incorrect predictions can be examined as:

- **False Positive:** The model predicts survival, but the passenger did not survive.
- **False Negative:** The model predicts non-survival, but the passenger survived.

Studying these errors helps us understand model limitations and identify areas for further improvement.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab
- Jupyter Notebook

## Key Learnings

This project reinforced an important machine learning principle: a model should not be judged by accuracy alone.

Different evaluation metrics reveal different aspects of performance. Cross-validation helps assess consistency, threshold tuning changes classification behavior, and error analysis provides a clearer understanding of model mistakes.

## Limitations

- Results may vary depending on the data split, selected features, and model configuration.
- The selected features may not capture every factor associated with survival.
- The historical Titanic dataset may contain biases.
- This project is intended for learning and experimentation, not real-world decision-making.

## Conclusion

Day 11 focused on moving beyond model training to understand model performance. By comparing classification models, evaluating multiple metrics, adjusting the decision threshold, and examining prediction errors, I developed a more practical understanding of how machine learning models should be evaluated.

**The main takeaway:** A good machine learning workflow does not stop when a model makes predictions. It continues with evaluation, error analysis, and evidence-based improvement.

---

**Created by:** Bharath Builds  
**Series:** 21 Days to ML Engineer  
**Day 11 Topic:** Model Evaluation and Error Analysis
