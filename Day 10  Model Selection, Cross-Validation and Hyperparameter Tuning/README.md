# Day 10 — Model Selection, Cross-Validation and Hyperparameter Tuning

**Channel:** Bharath Builds in AI  
**Series:** 21 Days to ML Engineer  
**Notebook:** `Day10_Model_Selection_Cross_Validation.ipynb`  
**Dataset:** Breast Cancer Wisconsin Dataset  
**Task:** Binary Classification

## Overview

This project focuses on comparing machine learning models, evaluating their performance using cross-validation, and selecting suitable hyperparameters.

The practical implementation uses scikit-learn to compare five classification models, tune an SVM using GridSearchCV, evaluate the selected model on an untouched test set, and save the experiment results.

## Objectives

- Understand training, validation and test sets.
- Learn K-Fold and Stratified K-Fold Cross-Validation.
- Differentiate model parameters and hyperparameters.
- Understand GridSearchCV and RandomizedSearchCV.
- Prevent common preprocessing data leakage.
- Compare classification models using a consistent evaluation strategy.
- Build a reproducible model-selection workflow.

## Theoretical Concepts Covered

- Model selection
- Training, validation and test sets
- K-Fold Cross-Validation
- Stratified K-Fold Cross-Validation
- Parameters vs hyperparameters
- Hyperparameter tuning
- GridSearchCV and RandomizedSearchCV
- Data leakage
- Pipeline-based preprocessing
- Fair model comparison
- ROC-AUC and evaluation metrics

## Practical Implementation

The Google Colab notebook follows these steps:

1. Import the required Python and scikit-learn libraries.
2. Load the Breast Cancer Wisconsin dataset.
3. Separate input features (`X`) and target (`y`).
4. Create an 80/20 stratified train/test split.
5. Build model Pipelines with appropriate preprocessing.
6. Compare five classification models using five-fold Stratified K-Fold Cross-Validation.
7. Tune SVM hyperparameters using GridSearchCV.
8. Evaluate the selected SVM on the untouched test set.
9. Save model comparison results, hyperparameters and evaluation metrics.
10. Interpret the results and document the conclusions.

## Models Compared

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest

## Machine Learning Pipeline

```text
Load Dataset
     |
     v
Train/Test Split
     |
     v
Define Model Pipelines
     |
     v
Stratified K-Fold Cross-Validation
     |
     v
Compare Five Models
     |
     v
SVM Hyperparameter Tuning
     |
     v
Select Best Configuration
     |
     v
Final Test Evaluation
     |
     v
Save Results
```

Pipelines keep preprocessing and model fitting together. For models that require scaling, the scaler is fitted within each training fold and applied to its corresponding validation fold.

## Evaluation Method

- Training/test split: 80/20
- Cross-validation: Five-fold Stratified K-Fold
- Model comparison metric: ROC-AUC
- Hyperparameter tuning: GridSearchCV
- Final evaluation: Untouched test set

Additional metrics include accuracy, precision, recall, F1-score and the confusion matrix.

## Results and Interpretation

The following results were obtained from the baseline model comparison.

| Model | Mean CV ROC-AUC | Standard Deviation |
|---|---:|---:|
| Logistic Regression | 0.9959 | 0.0050 |
| SVM | 0.9956 | 0.0048 |
| Random Forest | 0.9896 | 0.0083 |
| KNN | 0.9882 | 0.0081 |
| Decision Tree | 0.9179 | 0.0201 |

### Key Observations

- Logistic Regression achieved the highest mean ROC-AUC among the five baseline models.
- SVM performed almost as well as Logistic Regression.
- Random Forest and KNN also achieved strong validation results.
- Decision Tree had the lowest mean ROC-AUC and the greatest variation across folds.
- The difference between Logistic Regression and SVM was small.

### SVM Hyperparameter Tuning

GridSearchCV selected the following configuration:

- `C`: 0.1
- `kernel`: linear
- `gamma`: scale
- Best mean CV ROC-AUC: 0.9957
- Final test ROC-AUC: 0.9937

The tuned SVM achieved excellent ranking performance on the held-out test set. However, Logistic Regression remained the strongest baseline model in the comparison. The results do not establish that the tuned SVM is the overall best model.

Actual performance should be interpreted alongside the evaluation metrics, classification errors and requirements of the problem.

## Data Leakage Prevention

- The test set is not used during model comparison or hyperparameter tuning.
- Preprocessing is included inside Pipelines where required.
- Cross-validation is performed on the training data.
- The selected model is evaluated on the untouched test set after tuning.

## Technologies Used

- Python
- Google Colab
- Pandas
- scikit-learn
- GridSearchCV
- StratifiedKFold
- Pipeline
- Joblib

## Saved Outputs

The notebook saves the following files in `day10_results/`:

- `cv_results.csv` — baseline model comparison.
- `svm_grid_search_results.csv` — evaluated SVM configurations.
- `best_model_config.json` — selected hyperparameters and best CV score.
- `test_metrics.json` — final test evaluation metrics.
- `best_svm_pipeline.joblib` — fitted SVM Pipeline.

## How to Run

1. Open the Google Colab notebook.
2. Run the cells in order.
3. Review the baseline model comparison.
4. Inspect the selected SVM hyperparameters.
5. Review the final test metrics and confusion matrix.
6. Check the saved experiment results.

## Conclusion

This project demonstrates a structured approach to machine learning model selection. It combines cross-validation, systematic hyperparameter tuning, preprocessing Pipelines and final test evaluation in one reproducible workflow.

The main lesson is to compare models fairly, interpret actual results, prevent data leakage and document experiments so they can be reviewed and repeated.

**Bharath Builds in AI — Learn. Build. Explain.**
