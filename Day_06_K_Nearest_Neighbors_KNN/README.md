# BHARATH BUILDS | 21 DAYS TO ML ENGINEER

## DAY 06 — K-NEAREST NEIGHBORS (KNN)

Day 6 focuses on KNN, an instance-based and distance-based classification algorithm.

## OBJECTIVE

- Understand how KNN makes predictions using nearby data points.
- Understand distance-based learning and the role of K.
- Understand bias vs variance and the effect of small and large K.
- Understand why feature scaling is important.
- Experiment with multiple K values and select a suitable K using validation.

## THEORY COVERED

- KNN Intuition
- Instance-Based Learning
- Lazy Learning
- Euclidean Distance
- Manhattan Distance
- Choosing K
- Small K vs Large K
- Bias vs Variance
- Overfitting vs Underfitting
- Feature Scaling
- Standardization
- Decision Boundary
- Advantages and Limitations
- Cross-Validation

## PRACTICAL COVERED

### Dataset: Wine Classification

The Scikit-learn Wine dataset was selected because it is a classification dataset with multiple numerical features, making it suitable for demonstrating distance-based learning, feature scaling, and K-value experiments.

### Implementation

- Loaded Wine Classification Dataset
- Train/Test Split
- Baseline KNN
- Feature Scaling with StandardScaler
- KNN Pipeline
- Multiple K-Value Experiment
- K vs Accuracy
- 5-Fold Cross-Validation
- Best K Selection
- Final KNN Model
- Confusion Matrix
- Classification Report
- Decision Boundary Visualization

## KEY OBSERVATIONS

| K | Mean CV Accuracy | Observation |
|---:|---:|---|
| 1 | 94.95% | High variance / sensitive to local points |
| 3 | 94.40% | Lower validation performance |
| 5 | 94.94% | Stable performance |
| 7 | **96.65%** | Highest mean CV accuracy |
| 9 | **96.63%** | Almost equal performance, more consistent |
| 11 | 95.52% | Performance decreases |
| 15 | 95.52% | Larger neighborhood |

### Decision Boundary Experiment

- K = 1 → 100% accuracy, highly complex boundary
- K = 5 → 95% accuracy, more balanced boundary
- K = 15 → 93% accuracy, smoother boundary

### Main Findings

- KNN depends heavily on distance.
- Feature scaling is important for meaningful distance calculations.
- Small K → Low Bias, High Variance, Overfitting Risk.
- Large K → High Bias, Low Variance, Underfitting Risk.
- K = 7 achieved the highest mean cross-validation accuracy at approximately **96.65%**.
- K = 9 achieved approximately **96.63%** with the lowest standard deviation among the tested K values.
- Cross-validation provides a better basis for selecting K than relying on a single split.

## CORE CONCEPT

```text
Distance
   ↓
Nearest Neighbors
   ↓
Majority Voting
   ↓
Prediction
