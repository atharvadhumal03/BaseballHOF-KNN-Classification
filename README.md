# BaseballHOF-KNN-Classification
A machine learning project implementing binary classification to predict Baseball Hall of Fame inductions based on career batting statistics. This project demonstrates the complete ML workflow from outlier analysis to model optimization, achieving 88% accuracy using K-Nearest Neighbor.

## 📊 Project Overview
This project uses K-Nearest Neighbors (KNN) to classify whether baseball players with 500+ career hits will be inducted into the Hall of Fame based on 14 career statistics:

1. Games played (G)
2. At bats (AB)
3. Runs (R)
4. Hits (H)
5. Doubles (2B)
6. Triples (3B)
7. Home runs (HR)
8. RBIs
9. Walks (BB)
10. Strikeouts (SO)
11. Stolen bases (SB)
12. Caught stealing (CS)
13. Batting average (BA)
14. Years played (YRS)

The project addresses the challenge of quantifying Hall of Fame worthiness using objective performance metrics, dealing with class imbalance and statistical outliers common in baseball data.

## 🎯 Results Summary
### Model Performance
| **Metric** | **Score** |
|------------|-----------|
| Test Accuracy | 88% |
| Test Total Samples | 93 |
| Macro Average F1 | 0.85 |
| Weighted Average F1 | 0.88 |

### Per-class performance
| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| Non-HOF (0) | 0.88 | 0.97 | 0.92 | 65 |
| HOF (1) | 0.90 | 0.68 | 0.78 | 28 |

### Confusion Matrix
| **Actual/Predicted** | **Non-HOF** | **HOF** |
|---------------------|-------------|---------|
| **Non-HOF** | 63 | 2 |
| **HOF** | 9 | 19 |

## 🔍 Methodology
### 1. Data Exploration and Outlier Analysis
- Loaded the 500 Hits Club dataset containing players with 500+ career hits
- Created box plots to visualize statistical distributions across all features
- Identified significant outliers using IQR method (1.5×IQR threshold)
- Detected outliers in power statistics (HR) and speed statistics (SB) representing exceptional players

### 2. Data Preprocessing
- Feature-target separation: Excluded player names, used 14 statistical features
- Train-test split: 80-20 split resulting in:

Training set: ~324 samples
Test set: 93 samples


- Applied RobustScaler to minimize the influence of outliers on feature scaling
- Test data transformed using scaling parameters learned from training data to prevent data leakage

## 3. Model Development
- Algorithm: K-Nearest Neighbors classifier
- Hyperparameter tuning: Tested k values from 1 to 20
- Used 5-fold cross-validation to identify optimal k
- Selected k=11 based on highest average cross-validation accuracy
- Distance metric: Euclidean distance (default)

## 4. Model Evaluation
Comprehensive evaluation revealed conservative prediction behavior:

- High precision (90%) but moderate recall (68%) for HOF predictions
- Excellent recall (97%) for non-HOF players
- Model requires strong statistical evidence before predicting HOF induction

## 📈 Visualizations
### 1. Feature Distribution Analysis
- Created box plots showing outliers in each feature.

### 2. Confusion Matrix
| **Actual/Predicted** | **Non-HOF** | **HOF** |
|---------------------|-------------|---------|
| **Non-HOF** | 63 | 2 |
| **HOF** | 9 | 19 |

### 3. Classification Report
| **Class** | **Precision** | **Recall** | **F1-Score** | **Support** |
|-----------|---------------|------------|--------------|-------------|
| Non-HOF (0) | 0.88 | 0.97 | 0.92 | 65 |
| HOF (1) | 0.90 | 0.68 | 0.78 | 28 |

## 🔑 Key Findings
- RobustScaler Effectiveness: Outperformed StandardScaler due to outliers in baseball statistics (e.g., Barry Bonds' 73 HRs, Rickey Henderson's 130 SBs)
- Conservative Model Behavior: With k=11, the model requires strong consensus among similar players before predicting HOF induction
- High Precision for HOF: When the model predicts Hall of Fame, it's correct 90% of the time
- Class Imbalance Impact: Model shows bias toward majority class (non-HOF), missing 32% of actual HOF players
- Statistical Predictability: 88% accuracy demonstrates that career statistics alone are strong predictors of HOF induction

## 📁 Dataset Information
The 500 Hits Club dataset contains comprehensive batting statistics for baseball players who achieved 500 or more career hits. The dataset includes:
- Players from various eras of baseball history
- 14 numerical features capturing offensive performance
- Binary classification target (HOF: 0 or 1)
- Natural class imbalance reflecting selective HOF voting

## Target Distribution
- Non-HOF: 70% of players
- HOF: 30% of players

This imbalance reflects the exclusive nature of Hall of Fame induction, where only the most exceptional players among an already elite group (500+ hits) achieve this honor.
