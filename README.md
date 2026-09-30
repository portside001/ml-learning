1) logistic regression 
2) svc_svm suport vector machine algo 
3) knn K-Nearest Neighbors
4) naive_bayes



# Machine Learning Practice Repository

This repository contains basic implementations of core Machine Learning algorithms, focusing on **Regression**, **Classification**, and **Clustering**, along with sample datasets.

---

## 📁 Repository Contents

### 1. Regression (Supervised Learning - Continuous Values)
Use these algorithms when you want to predict a continuous numerical value (such as price, salary, or age).

| File Name | Description |
| :--- | :--- |
| `linear_regression.py` | Linear Regression — fits a straight line to predict a continuous number based on input features. |

---

### 2. Classification (Supervised Learning - Categorical Values)
Use these algorithms when you want to predict a specific category, label, or binary outcome (such as Yes/No, 0/1).

| File Name | Description |
| :--- | :--- |
| `logistic_regression.py` | Predicts probabilities for binary classification (despite having "regression" in its name). |
| `logit classification.csv` | Sample dataset used for testing logistic regression models. |
| `svc.py` | Support Vector Classifier (SVC) — separates classes using an optimal boundary line (hyperplane). |
| `naive_bayes.py` | Fast probabilistic classifier based on Bayes' Theorem, commonly used for text data. |
| `knn.py` | K-Nearest Neighbors — classifies a data point based on the most common class among its nearest neighbors. |
| `Dtree.py` | Decision Tree — splits data step-by-step using simple if-else decision rules. |
| `xg_boost.py` / `.ipynb` | XGBoost — a powerful, high-performance boosted tree model. |
| `Churn_Modelling.csv` | Bank customer dataset to predict whether a customer will stay or leave (binary classification). |

---

### 3. Clustering (Unsupervised Learning - Grouping Data)
Use these algorithms when your data has **no labels** and you want to group similar items together.

| File Name | Description |
| :--- | :--- |
| `K-mean-cluster.py` | K-Means Clustering — divides data into $K$ separate groups based on distance from cluster centers. |
| `Hirearcial.py` | Hierarchical Clustering — builds a tree of clusters (dendrogram) by merging similar points step-by-step. |
| `Mall_Customers.csv` | Customer dataset used to find shopper groups based on Annual Income and Spending Score. |

---

## 💡 Quick Comparison

| Type | Target ($y$) | Goal | Common Metrics |
| :--- | :--- | :--- | :--- |
| **Regression** | Continuous Number | Predict numeric quantity | MSE, RMSE, MAE, $R^2$ |
| **Classification** | Discrete Category | Predict label or class | Accuracy, Precision, Recall, F1 |
| **Clustering** | None (No Target) | Group similar patterns | Silhouette Score, Inertia (Elbow) |

