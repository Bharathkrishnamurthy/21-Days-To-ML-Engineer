# BHARATH BUILDS | 21 DAYS TO ML ENGINEER | DAY 8

# Naive Bayes Classification: Text Classification Using Python

## 1. Project Overview

This project demonstrates how to build a text classification model using the Naive Bayes algorithm and Scikit-learn.

The model classifies text documents into two categories using the **20 Newsgroups dataset**:

- `rec.sport.hockey` — Hockey and sports-related discussions
- `sci.space` — Space and science-related discussions

The project covers the mathematical intuition behind Naive Bayes, text preprocessing, TF-IDF vectorization, model training, classification evaluation, confusion matrix interpretation, and error analysis.

**Objective:** Build a practical machine learning pipeline that learns from text data and predicts whether a document belongs to the hockey or space category.

## 2. Learning Objectives

By completing this project, I learned how to:

- Understand classification versus regression.
- Explain Bayes' Theorem and conditional probability.
- Understand prior, likelihood, evidence, and posterior probability.
- Understand the Naive Bayes independence assumption.
- Explore Gaussian, Multinomial, and Bernoulli Naive Bayes.
- Apply text vectorization using TF-IDF.
- Train a Multinomial Naive Bayes classifier.
- Build a machine learning pipeline using Scikit-learn.
- Evaluate a classifier using accuracy, precision, recall, and F1-score.
- Interpret a confusion matrix and classification report.
- Analyze incorrect predictions.
- Understand the role of the smoothing parameter `alpha`.
- Save a trained machine learning pipeline for reuse.

## 3. Dataset Description

**Dataset:** 20 Newsgroups

**Source:** `sklearn.datasets.fetch_20newsgroups`

The dataset contains newsgroup discussion documents belonging to different topics. For this project, only two categories are selected to create a binary text classification task.

| Property | Description |
|---|---|
| Dataset name | 20 Newsgroups |
| Data type | Text documents |
| Machine learning task | Binary text classification |
| Category 1 | `rec.sport.hockey` |
| Category 2 | `sci.space` |
| Training data | Loaded using the training subset |
| Testing data | Loaded using the testing subset |
| Feature representation | TF-IDF |
| Classification algorithm | Multinomial Naive Bayes |
| Evaluation | Accuracy, precision, recall, F1-score, confusion matrix |

The dataset is loaded through Scikit-learn. The required dataset files may be downloaded automatically on the first run.

### Why use this dataset?

- It provides real text documents for classification.
- The two categories have different subject matter.
- It helps demonstrate text vectorization and probabilistic classification.
- It provides a practical example of document classification.

**Important:** The 20 Newsgroups dataset is not the SMS Spam Collection dataset. This project classifies hockey and space discussions, not spam and legitimate messages.

## 4. Theoretical Concepts Covered

### 4.1 Classification vs Regression

- **Classification:** Predicts a discrete class or category.
- **Regression:** Predicts a continuous numerical value.

Example:
- Classification: Predict whether a document discusses hockey or space.
- Regression: Predict the price of a house.

### 4.2 Bayes' Theorem

Bayes' Theorem calculates the probability of a hypothesis given observed evidence.

Formula:

\[
P(C \mid X) = \frac{P(X \mid C)P(C)}{P(X)}
\]

Where:

- \(P(C \mid X)\): Posterior probability
- \(P(X \mid C)\): Likelihood
- \(P(C)\): Prior probability
- \(P(X)\): Evidence

In this project, the model estimates which category is most likely given the words in a document.

### 4.3 Conditional Probability

Conditional probability measures the probability of an event when another event is known.

For example, the probability that a document belongs to the space category given that it contains certain words.

### 4.4 Naive Independence Assumption

Naive Bayes assumes that features are conditionally independent given the class.

In text classification, this simplifies the calculation of the probability of a document belonging to a particular category.

Although words in real documents are not truly independent, this assumption makes the algorithm computationally efficient.

### 4.5 Types of Naive Bayes

- **Gaussian Naive Bayes:** Suitable for continuous features modeled using Gaussian distributions.
- **Multinomial Naive Bayes:** Commonly used for word counts and non-negative text features.
- **Bernoulli Naive Bayes:** Suitable for binary features representing whether a word or feature is present.

**Algorithm used in this project:** Multinomial Naive Bayes with TF-IDF features.

### 4.6 TF-IDF Vectorization

Machine learning algorithms cannot directly learn from raw text in the same way they learn from numerical feature matrices.

TF-IDF converts text into numerical features based on word importance.

- **TF (Term Frequency):** Measures how frequently a term appears in a document.
- **IDF (Inverse Document Frequency):** Reduces the importance of terms that occur across many documents.
- **TF-IDF:** Combines these ideas to represent text numerically.

### 4.7 Laplace Smoothing

Naive Bayes uses smoothing to avoid zero probabilities for unseen features.

The `alpha` parameter controls the amount of smoothing in Multinomial Naive Bayes.

### 4.8 Decision and Prediction

The classifier estimates the probability of each class and predicts the most likely category.

## 5. Machine Learning Pipeline

The following pipeline represents the workflow implemented in this project:

```text
20 Newsgroups Dataset
          |
          v
Select Two Categories
          |
          v
Load Training and Testing Data
          |
          v
Remove Headers, Footers and Quotes
          |
          v
Separate Text Features and Labels
          |
          v
TF-IDF Vectorization
          |
          v
Multinomial Naive Bayes
          |
          v
Train the Model
          |
          v
Predict Test Documents
          |
          v
Evaluate Classification Metrics
          |
          v
Confusion Matrix
          |
          v
Error Analysis
          |
          v
Experiment with Alpha
          |
          v
Save the Trained Pipeline
```

### Why use a Pipeline?

A Scikit-learn Pipeline combines text vectorization and classification into one reusable workflow.

The pipeline ensures that TF-IDF is fitted on training documents and then applied to testing documents using the learned vocabulary and inverse document frequencies.

This helps prevent data leakage during preprocessing.

## 6. Implementation Workflow

### Step 1: Import Libraries

Import the required libraries from NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

### Step 2: Load the Dataset

Load the training and testing subsets of the 20 Newsgroups dataset using `fetch_20newsgroups`.

Select only the hockey and space categories.

### Step 3: Prepare the Text Data

Remove headers, footers, and quoted material using the dataset loader options.

Separate the documents into text features (`X`) and target labels (`y`).

### Step 4: Understand the Dataset

Inspect the number of documents, category names, and class distribution.

The aim is to validate the dataset before model training without repeating unnecessary exploratory data analysis.

### Step 5: Understand Bayes' Theorem

Review prior probability, likelihood, evidence, and posterior probability.

Understand how the algorithm estimates the probability of a document belonging to each category.

### Step 6: Understand the Independence Assumption

Learn why Naive Bayes simplifies probability calculations by assuming conditional independence between features.

### Step 7: Convert Text into Numerical Features

Use `TfidfVectorizer` to transform the text documents into numerical feature vectors.

Use `fit_transform()` on training documents and `transform()` on test documents when vectorizing manually.

### Step 8: Build the Model

Create a Scikit-learn Pipeline containing:

1. `TfidfVectorizer`
2. `MultinomialNB`

### Step 9: Train the Model

Fit the pipeline on the training documents and their corresponding category labels.

### Step 10: Generate Predictions

Use the trained model to predict categories for unseen test documents.

### Step 11: Evaluate the Model

Calculate the following metrics:

- Accuracy
- Precision
- Recall
- F1-score

Generate a classification report for both categories.

### Step 12: Visualize the Confusion Matrix

Use a heatmap to understand correct predictions and misclassifications for both categories.

### Step 13: Perform Error Analysis

Inspect incorrectly classified documents to understand where the model makes mistakes.

### Step 14: Experiment with Smoothing

Experiment with different `alpha` values to understand how smoothing affects model performance.

For a rigorous model-selection workflow, choose hyperparameters using training data and cross-validation or a separate validation set, and reserve the test set for final evaluation.

### Step 15: Save the Model

Save the complete fitted pipeline using `joblib` so that text preprocessing and classification can be reused together.

## 7. Model Evaluation Results

The following results were obtained from the model evaluation shown in the notebook.

### Overall Performance

| Metric | Result |
|---|---:|
| Test documents | 793 |
| Correct predictions | 753 |
| Incorrect predictions | 40 |
| Accuracy | 94.96% |

### Classification Report

| Category | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `rec.sport.hockey` | 0.93 | 0.98 | 0.95 | 399 |
| `sci.space` | 0.98 | 0.92 | 0.95 | 394 |
| Macro average | 0.95 | 0.95 | 0.95 | 793 |
| Weighted average | 0.95 | 0.95 | 0.95 | 793 |

**Note:** Results can vary slightly with changes to the dataset version, preprocessing, vectorizer settings, model configuration, or train/test data.

## 8. Confusion Matrix Analysis

The confusion matrix compares the actual category with the predicted category.

| Actual Category | Predicted Hockey | Predicted Space |
|---|---:|---:|
| Actual Hockey | 390 | 9 |
| Actual Space | 31 | 363 |

### Key Observations

1. **Correct predictions:** 390 hockey + 363 space = 753.
2. **Incorrect predictions:** 9 hockey documents were predicted as space, and 31 space documents were predicted as hockey.
3. **Accuracy:** Approximately 94.96%.
4. **Hockey:** High recall of 0.98, meaning the model correctly identifies most actual hockey documents.
5. **Space:** High precision of 0.98, but lower recall of 0.92.
6. **Conclusion:** The model performs well overall, but it misclassifies more actual space documents as hockey than actual hockey documents as space.

### Interpretation

The model identifies hockey documents particularly well in terms of recall. Space predictions have high precision, meaning documents predicted as space are usually correct.

However, some actual space documents are classified as hockey. This suggests an opportunity to investigate ambiguous vocabulary, shared words, and documents that contain limited topic-specific information.

## 9. Important Evaluation Metrics

### Accuracy

Accuracy measures the proportion of all predictions that are correct.

\[
\text{Accuracy} =
\frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]

For this experiment:

\[
\frac{753}{793} \approx 94.96\%
\]

### Precision

Precision measures how many documents predicted as a particular class actually belong to that class.

High precision means fewer false-positive predictions for that class.

### Recall

Recall measures how many documents belonging to a particular class were correctly identified.

High recall means fewer false negatives for that class.

### F1-score

F1-score is the harmonic mean of precision and recall.

\[
F1 = 2 \times
\frac{\text{Precision} \times \text{Recall}}
{\text{Precision} + \text{Recall}}
\]

It is useful when both precision and recall matter.

## 10. Key Learnings

- Naive Bayes is a probabilistic classification algorithm based on Bayes' Theorem.
- The independence assumption simplifies probability calculations.
- Multinomial Naive Bayes is a useful baseline for text classification.
- TF-IDF transforms raw text into numerical features.
- A Pipeline combines preprocessing and classification into a reusable workflow.
- Training-only fitting of the vectorizer prevents test-data leakage.
- Accuracy alone does not explain all classification errors.
- Precision and recall provide class-specific insights.
- A confusion matrix reveals the types and numbers of misclassifications.
- Error analysis helps identify where a model may need improvement.
- Smoothing can influence Naive Bayes predictions.
- A saved pipeline can be reused to classify new documents.

## 11. Limitations

- The independence assumption is often unrealistic for natural language.
- TF-IDF does not fully capture word order, context, or semantic meaning.
- Ambiguous documents can contain words associated with both categories.
- The experiment uses only two categories from the full dataset.
- The reported results apply to this dataset and configuration; they do not guarantee similar performance on other text-classification tasks.
- The model may require further validation and tuning before use in a production system.

## 12. Interview Questions

1. What is Naive Bayes?
2. Explain Bayes' Theorem.
3. What are prior, likelihood, evidence, and posterior probability?
4. Why is Naive Bayes called a naive classifier?
5. What is the conditional independence assumption?
6. What is the difference between Gaussian, Multinomial, and Bernoulli Naive Bayes?
7. Why is Multinomial Naive Bayes used for text classification?
8. What is TF-IDF, and why is it used?
9. What is Laplace smoothing?
10. What does the `alpha` parameter control?
11. Why should the vectorizer be fitted only on training data?
12. What is the purpose of a Scikit-learn Pipeline?
13. What is the difference between precision and recall?
14. How do you interpret a confusion matrix?
15. Why is accuracy alone insufficient for evaluating a classification model?
16. What is the purpose of error analysis?
17. What are the limitations of Naive Bayes?
18. How can you improve a text-classification model?

## 13. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Joblib
- Google Colab
- Git and GitHub

## 14. Project Structure

```text
day-08-naive-bayes/
│
├── naive_bayes_classification.ipynb
├── README.md
└── requirements.txt
```

The notebook contains the implementation, evaluation, visualizations, and experiments.

If the trained model is saved in the repository, consider storing it in a separate `models/` directory and avoiding unnecessary large generated files in Git.

## 15. How to Run the Project

### Option 1: Google Colab

1. Open Google Colab.
2. Create a new notebook.
3. Copy the notebook code into the appropriate cells.
4. Run the cells in sequence.
5. Allow Scikit-learn to download the dataset if necessary.
6. Review the evaluation metrics and confusion matrix.

### Option 2: Run Locally

Clone the repository:

```bash
git clone <your-repository-url>
cd day-08-naive-bayes
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `naive_bayes_classification.ipynb` and run the cells.

### Example `requirements.txt`

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

## 16. GitHub Portfolio Deliverable

The final deliverable is a reproducible notebook demonstrating:

- Dataset loading and preparation
- Naive Bayes theory
- TF-IDF vectorization
- Model training and prediction
- Classification metrics
- Confusion matrix visualization
- Error analysis
- Smoothing experiments
- Model serialization

This project demonstrates the practical application of a machine learning classification algorithm from data preparation to evaluation.

## 17. Day 8 Summary

**Topic:** Naive Bayes Classification

**Dataset:** 20 Newsgroups — Hockey vs Space

**Algorithm:** Multinomial Naive Bayes

**Feature Engineering:** TF-IDF

**Accuracy:** Approximately 94.96%

**Main takeaway:** A probabilistic model can perform effective text classification when raw text is transformed into useful numerical features and the results are evaluated using multiple metrics.

## 18. 21 DAYS TO ML ENGINEER — Learning Journey

This project is part of the **BHARATH BUILDS | 21 DAYS TO ML ENGINEER** series.

The learning journey progresses from machine learning foundations to practical algorithms, evaluation, and model-building workflows.

Previous topics covered:

- Day 1: Machine Learning Foundations and ML Workflow
- Day 2: Data Cleaning, EDA, and Preprocessing
- Day 3: Linear Regression
- Day 4: Multiple Regression and Regularization
- Day 5: Logistic Regression and Decision Threshold
- Day 6: K-Nearest Neighbors (KNN)
- Day 7: Decision Trees
- Day 8: Naive Bayes Classification

Follow the series to continue learning machine learning through theory, practical implementation, interview preparation, assignments, and GitHub portfolio projects.

---

**Created as part of the BHARATH BUILDS learning series.**

Explain. Implement. Interpret. Document. Commit.
