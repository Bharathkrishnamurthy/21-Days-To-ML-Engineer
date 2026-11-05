
# BHARATH BUILDS
# 21 Days to ML Engineer — Day 15

## K-Medoids and Hierarchical Clustering

### Session Overview
Day 15 focuses on understanding and implementing K-Medoids and Hierarchical Clustering algorithms for customer segmentation.

The session compares different clustering approaches and interprets the customer groups formed using income and spending behavior.

## Dataset Used

**Dataset:** Mall Customers Dataset  
**File:** `Mall_Customers.csv`

### Features Used
- Annual Income (k$)
- Spending Score (1–100)

### Objective
To segment customers based on their annual income and spending score, compare clustering algorithms, and understand the customer behavioral patterns identified by each model.

## Theoretical Concepts Covered

- Introduction to K-Medoids Clustering
- K-Means vs. K-Medoids
- Medoid concept and representative data points
- PAM (Partitioning Around Medoids) intuition
- Introduction to Hierarchical Clustering
- Agglomerative Clustering
- Bottom-up clustering approach
- Linkage methods
- Ward Linkage
- Dendrogram interpretation
- Cluster formation and customer segmentation
- Comparison of clustering algorithms

## Practical Aspects Covered

- Loading the Mall Customers dataset
- Selecting relevant features for clustering
- Applying feature scaling using StandardScaler
- Using Day 14 K-Means as a baseline
- Implementing K-Medoids clustering
- Implementing Agglomerative Hierarchical Clustering
- Visualizing customer clusters
- Creating and interpreting a dendrogram
- Generating cluster profiles using groupby
- Comparing cluster sizes and average feature values
- Interpreting customer behavior based on income and spending
- Comparing K-Means, K-Medoids, and Agglomerative Clustering

## Key Interpretation Points

### 1. K-Means Clustering
- Identified five customer segments.
- One group represents customers with moderate income and moderate spending.
- High-income, high-spending customers form a valuable customer segment.
- Low-income customers with high spending scores show strong spending behavior despite lower income.
- High-income customers with low spending scores may have potential for targeted marketing.
- Low-income, low-spending customers form a separate segment.

### 2. K-Medoids Clustering
- Identified similar customer behavioral patterns to K-Means.
- The medoid represents an actual customer data point within each cluster.
- High-income, high-spending customers remain a distinct group.
- Low-income, high-spending customers form a separate customer segment.
- The moderate-income, moderate-spending group contains the largest number of customers.

### 3. Agglomerative Hierarchical Clustering
- Identified five customer segments using a bottom-up clustering approach.
- The dendrogram helps visualize how individual customers and clusters merge.
- High-income, high-spending customers form a clear segment.
- High-income, low-spending customers show a different spending pattern.
- Moderate-income, moderate-spending customers form the largest group.

### 4. Comparative Interpretation
- All three algorithms identified similar customer behavioral patterns.
- High-income, high-spending customers consistently formed a group of 39 customers.
- Low-income, high-spending customers formed a group of approximately 21–22 customers.
- High-income, low-spending customers formed a group of approximately 32–35 customers.
- Low-income, low-spending customers formed a group of approximately 21–23 customers.
- The moderate-income, moderate-spending group was the largest segment across all three algorithms.
- Cluster labels are arbitrary and do not represent rankings.
- Differences in cluster sizes occur because each algorithm uses a different clustering strategy.

## Workflow Diagram

```text
Mall Customers Dataset
        |
        v
Select Relevant Features
(Annual Income, Spending Score)
        |
        v
Feature Scaling
(StandardScaler)
        |
        v
K-Means Clustering
(Day 14 Baseline)
        |
        v
K-Medoids Clustering
(Medoid-Based Segmentation)
        |
        v
Agglomerative Hierarchical Clustering
(Ward Linkage)
        |
        v
Dendrogram Visualization
        |
        v
Cluster Visualization
        |
        v
Cluster Profiling
(Count, Mean, Median)
        |
        v
Customer Behavior Interpretation
        |
        v
Compare All Three Algorithms
        |
        v
Final Conclusion
```

## Final Conclusion

K-Means, K-Medoids, and Agglomerative Hierarchical Clustering were applied to the Mall Customers dataset to perform customer segmentation.

Although the algorithms use different approaches to form clusters, they identified similar customer behavioral patterns based on annual income and spending score.

This practical session demonstrated how clustering algorithms can support customer segmentation, behavioral analysis, and data-driven marketing decisions.

## Learning Outcome

By the end of Day 15, I learned how to:

- Implement K-Medoids and Hierarchical Clustering.
- Understand the role of medoids and dendrograms.
- Interpret cluster profiles using statistical summaries.
- Compare clustering algorithms based on customer behavior.
- Convert clustering results into meaningful business insights.

## Day 15 Principle

**EXPLAIN → IMPLEMENT → INTERPRET → DOCUMENT → COMMIT**
```
