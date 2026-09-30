# 🚀 BHARATH BUILDS — 21 DAYS TO ML ENGINEER

## DAY 4 — REGULARIZATION + MULTIPLE REGRESSION

> **From Multiple Features → Regularization → Model Comparison → Generalization**

Day 4 focuses on understanding how machine learning models behave when we work with **multiple features** and how **regularization** can help control model complexity.

Instead of only training a regression model, today's practical goes one step further — we train **Linear Regression, Ridge Regression, and Lasso Regression**, compare their performance, analyze their coefficients, and use **Cross-Validation** to understand generalization.

---

## 🎯 Objective

By the end of Day 4, I learned how to:

* Understand Multiple Linear Regression.
* Understand why multiple features can increase model complexity.
* Identify Underfitting, Good Fit, and Overfitting.
* Understand Multicollinearity.
* Understand the need for Regularization.
* Understand the Regularization parameter `alpha`.
* Implement Ridge Regression using L2 regularization.
* Implement Lasso Regression using L1 regularization.
* Understand how feature scaling affects regularized models.
* Compare model coefficients.
* Use Cross-Validation to evaluate generalization.
* Understand how excessive regularization can cause underfitting.

---

# 📚 Topics Covered

### Theory

* Multiple Linear Regression
* Why Multiple Features?
* Underfitting vs Good Fit vs Overfitting
* Why More Features Can Cause Overfitting
* Multicollinearity
* Regularization
* Regularization Parameter — Alpha
* Ridge Regression — L2
* Lasso Regression — L1
* Ridge vs Lasso
* Feature Scaling
* Bias-Variance Trade-off
* Generalization
* Training Performance vs Generalization
* Cross-Validation

### Practical

* Load California Housing Dataset
* Perform minimum dataset checks
* Train/Test Split
* Train Linear Regression
* Train Ridge Regression
* Train Lasso Regression
* Compare R² and MAE
* Compare model coefficients
* Perform 5-Fold Cross-Validation
* Analyze generalization
* Create a final model comparison table

---

# 🗂️ Dataset

For this practical, I used the **California Housing Dataset** available through Scikit-learn.

The dataset contains multiple numerical features that are used to predict a continuous target — house value.

### Features Used

* MedInc
* HouseAge
* AveRooms
* AveBedrms
* Population
* AveOccup
* Latitude
* Longitude

### Target

**House Value**

---

# 🔄 Machine Learning Workflow

```text
                DATASET
                   ↓
           MULTIPLE FEATURES
                   ↓
          TRAIN / TEST SPLIT
                   ↓
          LINEAR REGRESSION
                   ↓
        ┌──────────┴──────────┐
        ↓                     ↓
      RIDGE                  LASSO
       L2                     L1
        ↓                     ↓
        └──────────┬──────────┘
                   ↓
           COMPARE METRICS
                   ↓
        COMPARE COEFFICIENTS
                   ↓
          CROSS-VALIDATION
                   ↓
          CHECK GENERALIZATION
```

---

# 💻 Models Implemented

## 1. Linear Regression

Used as the **baseline model**.

It gives us a reference point to understand how the regularized models behave.

---

## 2. Ridge Regression

Ridge uses **L2 regularization**.

The goal is to control large coefficients and reduce model complexity.

In this practical, Ridge was implemented with:

```python
Ridge(alpha=1.0)
```

Feature scaling was applied before Ridge using `StandardScaler`.

---

## 3. Lasso Regression

Lasso uses **L1 regularization**.

One important property of Lasso is that it can push some coefficients exactly to zero, which can help with feature selection.

Initial configuration:

```python
Lasso(alpha=0.01, max_iter=10000)
```

---

# 📊 Model Results

| Model             | Train R² |   Test R² | Test MAE | Mean CV R² | Std CV R² |
| ----------------- | -------: | --------: | -------: | ---------: | --------: |
| Linear Regression | 0.612551 |  0.575788 | 0.533200 |   0.611484 |  0.006467 |
| Ridge             | 0.612551 |  0.575816 | 0.533193 |   0.611484 |  0.006460 |
| Lasso             | 0.000000 | -0.000219 | 0.906069 |   0.607711 |  0.004625 |

---

# 🔍 Results Analysis

## Linear Regression

Linear Regression achieved:

* Train R² → **0.6126**
* Test R² → **0.5758**
* Test MAE → **0.5332**

This gave us a reasonable baseline for comparison.

The difference between training and testing performance was not extremely large, so there was no strong indication of severe overfitting from these metrics.

---

## Ridge Regression

Ridge achieved:

* Train R² → **0.6126**
* Test R² → **0.5758**
* Test MAE → **0.5332**

The results were almost identical to Linear Regression.

This shows that with the selected `alpha`, Ridge had only a small effect on predictive performance.

However, Ridge still provides coefficient shrinkage, which is useful for controlling model complexity.

---

## Lasso Regression

Lasso produced:

* Train R² → **0.0000**
* Test R² → **-0.0002**
* Test MAE → **0.9061**

The Lasso coefficients were effectively pushed to zero.

This indicates that the selected:

```python
alpha = 0.01
```

was too strong for this particular experiment.

As a result, the model became too simple and **underfit the data**.

A negative R² means that the model performed slightly worse than simply predicting the average target value.

---

# 📌 Coefficient Analysis

The coefficients were compared across:

```text
Linear Regression
       ↓
Ridge Regression
       ↓
Lasso Regression
```

The important observation was that Lasso pushed the coefficients toward zero.

For the selected Lasso configuration, the coefficients became effectively:

```text
0
-0
```

Here, `-0.0` is numerically the same as zero.

This does **not** mean those features are universally useless.

It means that **for this dataset and the selected regularization strength**, Lasso removed their contribution from the model.

---

# 🔄 Cross-Validation

I also performed **5-Fold Cross-Validation** to check how consistently the models performed across different validation splits.

### Mean CV R²

```text
Linear Regression → 0.611484
Ridge Regression  → 0.611484
Lasso Regression  → 0.607711
```

Linear Regression and Ridge produced almost identical mean CV R² values.

Lasso produced a slightly lower mean CV R² with the selected alpha.

This supported the observation that the current Lasso configuration was too strongly regularized.

---

# 🧠 Key Learning

One of the biggest lessons from today's practical was:

> **Regularization is not automatically better just because we use it. The regularization strength matters.**

A very high regularization strength can remove too much information from the model and cause underfitting.

So instead of choosing `alpha` randomly, we should consider **tuning the regularization parameter**.

---

# ⚖️ Ridge vs Lasso

| Ridge                                           | Lasso                                    |
| ----------------------------------------------- | ---------------------------------------- |
| L2 Regularization                               | L1 Regularization                        |
| Shrinks coefficients                            | Can shrink coefficients to zero          |
| Usually keeps all features                      | Can perform feature selection            |
| Useful for controlling large coefficients       | Useful when sparse solutions are desired |
| Does not normally eliminate features completely | Can eliminate feature contributions      |

---

# 🎯 Practical Question

> **How does regularization affect model coefficients and generalization when we use multiple features?**

### Answer from today's experiment:

Regularization controls the size of model coefficients.

* Ridge shrinks coefficients.
* Lasso can drive coefficients to zero.
* Too little regularization may not sufficiently control complexity.
* Too much regularization can cause underfitting.
* The regularization parameter `alpha` therefore needs to be chosen carefully.

---

# 🛠️ Tools & Technologies

* Python
* Google Colab
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

# 📁 Project Structure

```text
DAY-04-REGULARIZATION-MULTIPLE-REGRESSION/
│
├── Day_04_Regularization_Multiple_Regression.ipynb
│
├── README.md
│
└── results/
    └── model_comparison.csv
```

---

# 💼 Interview Questions Covered

### 1. What is Multiple Linear Regression?

A regression technique that uses multiple input features to predict one continuous target.

### 2. What is Multicollinearity?

When two or more input features are strongly correlated with each other.

### 3. What is Regularization?

A technique used to control model complexity by adding a penalty to large coefficients.

### 4. What is Ridge Regression?

A regression method that uses L2 regularization to shrink coefficients.

### 5. What is Lasso Regression?

A regression method that uses L1 regularization and can drive coefficients to zero.

### 6. Why is feature scaling important for Ridge and Lasso?

Because regularization penalizes coefficient magnitudes, and features with different scales can otherwise affect the penalty differently.

### 7. What happens when alpha is too high?

The model can become too simple and underfit the training data.

### 8. What is Cross-Validation?

A technique that evaluates a model across multiple train-validation splits to get a more reliable estimate of generalization.

---

# 📝 Assignment

### Experiment with Alpha

Modify the Lasso regularization strength and observe how the model changes.

Try:

```python
alpha = 0.001
alpha = 0.01
alpha = 0.1
```

For each value, compare:

* Train R²
* Test R²
* Test MAE
* Coefficients
* Mean CV R²
* Number of zero coefficients

### Question

> **How does changing alpha affect model complexity, coefficients, and generalization?**

---

# 📌 Final Takeaway

Day 4 helped me move beyond simply training a regression model.

I learned how multiple features increase model complexity, why multicollinearity can make coefficients difficult to interpret, and how Ridge and Lasso regularization can control that complexity.

The practical experiment also showed an important real-world lesson:

> **A machine learning technique is only useful when its parameters are chosen appropriately.**

In this experiment, Ridge behaved very similarly to the baseline Linear Regression, while the selected Lasso configuration was too strongly regularized and resulted in underfitting.

This is why **model evaluation, coefficient analysis, cross-validation, and hyperparameter tuning** are important parts of the ML engineering workflow.

---

## 🚀 Day 4 Complete

**EXPLAIN → IMPLEMENT → INTERPRET → DOCUMENT → COMMIT**

### Progress

```text
Day 01 ✅
Day 02 ✅
Day 03 ✅
Day 04 ✅
...
Day 21 ⏳
```

---

## 🔜 Next Day

**Day 5 — Classification Fundamentals**

Moving from predicting continuous values to predicting **classes/categories** and understanding the fundamentals of classification models.

---

### 👨‍💻 Created by

**Bharath K. — Bharath Builds**

🚀 **21 Days to ML Engineer**

Learning ML Engineering by building, experimenting, analyzing, and documenting every day.
