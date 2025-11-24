# Rossmann Store Sales Prediction

## Project Overview
This project builds an end-to-end machine learning pipeline to predict daily sales for 1,115 Rossmann drug stores across Germany. The goal is to forecast 6 weeks of daily sales while minimizing the Root Mean Square Percentage Error (RMSPE).

This solution focuses on **Robust Feature Engineering** and **Target Transformation** rather than complex ensemble stacking, demonstrating how domain knowledge improves model performance more than raw compute.

## 📊 Performance Metric
The competition metric is RMSPE (Root Mean Square Percentage Error):

$$\textrm{RMSPE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\frac{y_i - \hat{y}_i}{y_i}\right)^2}$$

| Model Stage | Technique Used | RMSPE Score |
| :--- | :--- | :--- |
| **Baseline** | Basic XGBoost + Raw Sales | ~0.135 |
| **Engineering V1** | + Log Transformation (Log1p) | ~0.128 |
| **Engineering V2** | + Store Metadata (`store.csv`) | ~0.125 |
| **Final Model** | + Target Encoding (`StoreDowAvg`, `StoreDayPromoAvg`) | **~0.120** |

## 🛠️ Tech Stack
* **Python 3.10+**
* **Pandas / NumPy:** Data manipulation and time-series engineering.
* **XGBoost:** Gradient Boosting Regressor (Histogram-based).
* **Optuna:** Hyperparameter optimization (Bayesian Search).
* **Matplotlib:** Visualization of sales trends.

## 🧠 Engineering Decisions

### 1. Time-Based Splitting
* **Problem:** Random splitting (`train_test_split`) causes data leakage in time-series forecasting.
* **Solution:** We train on the first ~2.5 years and validate on the last 6 weeks of the dataset, mimicking the real-world forecasting scenario.

### 2. Handling "Closed" Days
* **Logic:** If a store is closed (`Open = 0`), Sales are 0.
* **Implementation:** The model is trained only on Open days. For prediction, any closed day in the test set is hard-coded to 0 post-inference.

### 3. Log-Transformation
* **Problem:** XGBoost optimizes MSE (Mean Squared Error), which focuses on absolute errors. A $100 error on a $1000 day is penalized the same as a $100 error on a $10,000 day.
* **Solution:** We train on `np.log1p(Sales)`. Minimizing MSE in log-space is mathematically equivalent to minimizing the relative percentage error (RMSPE).

### 4. Target Encoding (The "Nuclear" Features)
We injected "Store Personality" directly into the model to capture individual store behavior without high-cardinality One-Hot Encoding.

* **`StoreDowAvg`:** Average sales for Store X on Day Y (Captures weekly rhythm).
* **`StoreDayPromoAvg`:** Average sales for Store X on Day Y given Promo Z (Captures the specific impact of promotions on specific days).

> **Note:** Averages are calculated strictly on the Training set to prevent leakage.

## 🚀 How to Run

**1. Install Dependencies**
```bash
pip install pandas numpy xgboost optuna matplotlib
