# Bharath Builds
## 21 Days to ML Engineer - Day 14
# K-Means Clustering

## Project Overview

Day 14 focuses on K-Means Clustering, an unsupervised machine learning algorithm used to identify patterns and group similar data points into clusters.

The objective of this project is to understand the working principles of K-Means Clustering and implement customer segmentation using Python and the Mall Customer Segmentation dataset.

This project follows a practical learning approach that combines theoretical understanding, model implementation, visualization, and result interpretation.

## Objective

The primary objectives of this project are:

- Understand the fundamentals of unsupervised machine learning.
- Learn how the K-Means algorithm groups similar data points.
- Understand centroids, Euclidean distance, and cluster assignment.
- Apply feature scaling before clustering.
- Evaluate different cluster counts using the Elbow Method and Silhouette Score.
- Implement K-Means using Scikit-learn.
- Visualize and interpret customer segments.
- Identify meaningful customer behavior patterns from the data.

## Dataset

### Mall Customer Segmentation Dataset

The Mall Customer Segmentation dataset contains customer information that can be used to identify different customer groups based on their income and spending behavior.

Dataset source:
https://raw.githubusercontent.com/abcom-mltutorials/mall/main/Mall_Customers.csv

### Dataset Information

- Total customer records: 200
- Total columns: 5
- Selected features:
  - Annual Income (k$)
  - Spending Score (1-100)

### Features Used

| Feature | Description |
|---|---|
| Annual Income (k$) | Customer's annual income in thousands of dollars |
| Spending Score (1-100) | Score assigned based on customer spending behavior |

CustomerID was excluded from clustering because it is an identifier and does not represent meaningful customer behavior.

## Tools and Technologies

| Tool / Library | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Development and execution environment |
| Pandas | Data loading, exploration, and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | StandardScaler, KMeans, and Silhouette Score |
| Jupyter Notebook | Step-by-step practical implementation |

## Theoretical Concepts Covered

### 1. Supervised vs Unsupervised Learning

Understanding the difference between supervised learning, which uses labeled data, and unsupervised learning, which identifies patterns in unlabeled data.

### 2. K-Means Clustering

K-Means is an unsupervised machine learning algorithm that divides data into K clusters based on similarity.

### 3. K-Means Algorithm Workflow

The algorithm follows these steps:

1. Select the number of clusters K.
2. Initialize cluster centroids.
3. Assign each data point to its nearest centroid.
4. Recalculate the centroids using the assigned points.
5. Repeat the assignment and centroid update steps until convergence.

### 4. Centroids and Euclidean Distance

A centroid represents the center of a cluster.

Euclidean distance is used to measure the distance between a data point and a centroid.

### 5. Feature Scaling

Feature scaling is important because K-Means relies on distance calculations. StandardScaler is used to standardize the selected features before training.

### 6. Inertia and Elbow Method

Inertia measures the sum of squared distances between data points and their assigned cluster centroids.

The Elbow Method helps identify a suitable number of clusters by observing the reduction in inertia as K increases.

### 7. Silhouette Score

The Silhouette Score evaluates how well data points fit within their assigned clusters compared with neighboring clusters.

A higher score generally indicates better-defined clusters.

### 8. Limitations of K-Means

- The number of clusters K must be selected in advance.
- The algorithm is sensitive to feature scaling.
- It can be affected by outliers.
- Different centroid initializations may produce different results.
- It works best when clusters are reasonably compact and separated.

## Practical Implementation

The practical implementation was completed using Google Colab.

### Step 1: Import Libraries

Imported the required Python libraries for data analysis, visualization, scaling, clustering, and evaluation.

### Step 2: Load and Explore the Dataset

Loaded the Mall Customer Segmentation dataset and performed basic data inspection using Pandas.

### Step 3: Feature Selection

Selected Annual Income and Spending Score as the input features for customer segmentation.

### Step 4: Feature Scaling

Applied StandardScaler to standardize the selected features before clustering.

### Step 5: Elbow Method

Evaluated inertia for different values of K and visualized the results to identify a suitable cluster count.

### Step 6: Silhouette Score Evaluation

Calculated Silhouette Scores for multiple cluster counts to evaluate cluster separation.

### Step 7: Train the K-Means Model

Trained the final K-Means model using K = 5, with K-Means++ initialization, n_init = 10, and random_state = 42.

### Step 8: Cluster Visualization

Visualized the customer groups using a scatter plot and displayed the cluster centroids.

### Step 9: Cluster Profiling

Calculated the number of customers, average annual income, and average spending score for each cluster.

### Step 10: Result Interpretation

Analyzed the identified customer segments to understand different income and spending patterns.

## Results and Interpretation

The final K-Means model identified five customer segments using Annual Income and Spending Score.

### Model Summary

| Metric | Result |
|---|---|
| Dataset | Mall Customer Segmentation |
| Total Customers | 200 |
| Number of Clusters | 5 |
| Silhouette Score | 0.555 |
| Selected Features | Annual Income, Spending Score |

### Cluster Profiles

| Cluster | Customers | Average Annual Income (k$) | Average Spending Score |
|---|---:|---:|---:|
| Cluster 0 | 81 | 55.3 | 49.5 |
| Cluster 1 | 39 | 86.5 | 82.1 |
| Cluster 2 | 22 | 25.7 | 79.4 |
| Cluster 3 | 35 | 88.2 | 17.1 |
| Cluster 4 | 23 | 26.3 | 20.9 |

### Key Interpretations

1. Cluster 0 represents a large group of customers with moderate average income and spending behavior.
2. Cluster 1 contains high-income customers with high spending scores, indicating strong spending activity.
3. Cluster 2 represents lower-income customers with high spending scores, showing that lower income does not necessarily mean lower spending.
4. Cluster 3 contains high-income customers with low spending scores, indicating potential opportunities to improve customer engagement.
5. Cluster 4 represents customers with lower income and lower spending scores.

The Silhouette Score of 0.555 indicates reasonably separated clusters, although the segmentation is not perfect.

The results demonstrate that customers with similar income levels can exhibit different spending behaviors. K-Means helps identify these patterns without predefined labels.

## Conclusion

This project demonstrates the practical application of K-Means Clustering for customer segmentation using Python.

The implementation covered feature selection, feature scaling, cluster evaluation, model training, visualization, and cluster profiling.

The project highlights how unsupervised machine learning can be used to discover hidden patterns in customer data and support data-driven business decisions.

## Learning Workflow

EXPLAIN → IMPLEMENT → INTERPRET → DOCUMENT → COMMIT

## Series Information

**Channel:** Bharath Builds

**Series:** 21 Days to ML Engineer

**Day:** 14

**Topic:** K-Means Clustering

**Focus:** Unsupervised Machine Learning and Customer Segmentation
