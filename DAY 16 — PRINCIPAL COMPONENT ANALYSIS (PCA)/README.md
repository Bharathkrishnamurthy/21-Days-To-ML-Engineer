# BHARATH BUILDS
# Day 16 — Principal Component Analysis (PCA)

### Machine Learning in Kannada | Dimensionality Reduction

## Overview

This project demonstrates Principal Component Analysis (PCA), a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional representation while retaining important information.

The implementation uses the Wine Dataset from Scikit-learn to reduce 13 original features to 2 principal components, visualize the data, and compare classification performance.

## Objectives

- Understand PCA and dimensionality reduction.
- Analyze explained variance and principal components.
- Apply feature scaling before PCA.
- Reduce 13 features to 2 principal components.
- Visualize the dataset in 2D.
- Compare model performance before and after PCA.

## Theoretical Concepts Covered

- Dimensionality Reduction
- Curse of Dimensionality
- Principal Component Analysis
- Principal Components (PC1 and PC2)
- Explained Variance Ratio
- Feature Scaling
- Data Projection
- Advantages and Limitations of PCA

## Practical Implementation

- Loaded the Wine Dataset.
- Performed Train-Test Split.
- Applied StandardScaler.
- Implemented PCA with 2 components.
- Calculated explained variance ratio.
- Visualized cumulative explained variance.
- Created a PC1 vs PC2 scatter plot.
- Compared Random Forest classification accuracy using original and PCA-reduced features.
- Interpreted the results.

## Dataset

**Dataset:** Wine Dataset  
**Source:** Scikit-learn  
**Total Samples:** 178  
**Original Features:** 13  
**Target Classes:** 3

## Key Interpretation & Final Takeaways

### Explained Variance

- PC1 captures **35.79%** of total variance.
- PC2 captures **19.27%** of total variance.
- Together, they retain **55.06%** of the original variance.

### Class Distribution

- Class 0 is mostly located in the positive PC1 and PC2 region.
- Class 1 is concentrated around negative PC2 values.
- Class 2 is mostly distributed along negative PC1 values.

### Class Overlap

Some data points overlap in the PC1-PC2 space, making certain classes harder to distinguish.

PCA maximizes variance, not class separation.

### Model Performance Comparison

| Approach | Features | Test Accuracy |
|---|---:|---:|
| Original Features | 13 | 100% |
| PCA-Reduced Features | 2 | 88.89% |

The PCA-based model experienced an **11.11 percentage-point decrease in accuracy**.

This may be due to the loss of classification-relevant information during dimensionality reduction.

## Final Conclusion

PCA successfully reduced the Wine Dataset from 13 features to 2 principal components, making visualization easier and reducing feature complexity.

However, dimensionality reduction resulted in a decrease in classification accuracy.

The key takeaway is to balance dimensionality reduction, explained variance, and predictive performance when selecting principal components.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab

## Learning Principle

**Understand → Implement → Visualize → Evaluate → Interpret**

---

### Bharath Builds — Day 16
Principal Component Analysis (PCA)
