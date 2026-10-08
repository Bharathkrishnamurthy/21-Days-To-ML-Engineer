# BHARATH BUILDS | 21 DAYS TO ML ENGINEER

# DAY 7 — DECISION TREES

> Understanding how a Decision Tree learns rules, creates splits, controls complexity, and makes predictions.

---

## DAY 7 OBJECTIVE

The objective of Day 7 is to understand how a Decision Tree creates **rule-based predictions** and how tree complexity affects model performance.

### What We Covered

- Decision Tree fundamentals
- Nodes, branches, and leaves
- Splits and impurity
- Gini Impurity
- Entropy
- Information Gain
- Recursive splitting
- Tree depth
- Underfitting and overfitting
- `max_depth`
- Stopping conditions
- Pruning concepts
- Feature importance
- Decision Tree visualization
- Practical experimentation with different tree depths

---

# 1. WHAT IS A DECISION TREE?

A Decision Tree is a supervised machine learning algorithm that makes predictions using a sequence of feature-based rules.

### Core Idea

```text
Feature
   ↓
Condition
   ↓
Split
   ↓
Branch
   ↓
Leaf
   ↓
Prediction

# DAY 7 — DECISION TREE WORKING PIPELINE

## Complete Working Pipeline

```text
Breast Cancer Dataset
        ↓
Load Features (X) + Target (y)
        ↓
Train / Test Split
        ↓
Decision Tree Classifier
        ↓
Learn Feature-Based Splits
        ↓
Calculate Impurity
        ↓
Gini / Entropy
        ↓
Choose Better Split
        ↓
Recursive Splitting
        ↓
Build Tree
        ↓
Control Tree Complexity
        ↓
Experiment with max_depth
        ↓
Train Accuracy vs Test Accuracy
        ↓
Select Suitable Tree Depth
        ↓
Final Decision Tree
        ↓
Make Predictions
        ↓
Evaluate Model
        ↓
Accuracy
        ↓
Confusion Matrix
        ↓
Classification Report
        ↓
Visualize Decision Tree
        ↓
Inspect Feature Importance
        ↓
Interpret Results
