# 🛒 Customer Segmentation & Classification — Online Retail

> RFM-based customer segmentation using K-Means clustering, followed by supervised classification with Random Forest, KNN, Decision Tree, SVM, XGBoost, and LightGBM.

---

## 📌 Overview

This project analyzes transactional data from an online retail store to segment customers into meaningful groups and then train machine learning classifiers to predict customer segments for new data. The pipeline covers the full data science lifecycle: data cleaning, feature engineering, unsupervised clustering, and supervised classification with model comparison.

---

## 📂 Dataset

**Source:** [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/online+retail)

| Column        | Description                                                   |
|---------------|---------------------------------------------------------------|
| `InvoiceNo`   | Unique 6-digit transaction ID (prefix `C` = cancellation)    |
| `StockCode`   | Unique 5-digit product code                                   |
| `Description` | Product name                                                  |
| `Quantity`    | Units purchased per transaction                               |
| `InvoiceDate` | Date and time of the transaction                              |
| `UnitPrice`   | Price per unit in GBP (£)                                     |
| `CustomerID`  | Unique 5-digit customer identifier                            |
| `Country`     | Country of the customer                                       |

---

## 🔧 Pipeline

### 1. Data Cleaning
- Filtered `InvoiceNo` to valid 6-digit numeric transactions (excluded cancellations prefixed with `C` or `A`)
- Filtered `StockCode` to valid product codes (`\d{5}`, `\d{5}[a-zA-Z]+`, and `PADS`); excluded special codes like `DOT`, `D`, `M`, `BANK CHARGES`, `AMAZONFEE`, `TESTXXX`, gift card codes, etc.
- Dropped rows with missing `CustomerID`
- Removed transactions with `UnitPrice ≤ 0`
- ~27% of raw records were dropped after cleaning

### 2. Feature Engineering — RFM Metrics
For each customer, three metrics are computed:

| Feature          | Description                                        |
|------------------|----------------------------------------------------|
| `MonetaryValue`  | Total spend (`Quantity × UnitPrice`)               |
| `Frequency`      | Number of unique invoices                          |
| `Recency`        | Days since last purchase (relative to dataset max) |

### 3. Outlier Removal
- IQR-based outlier removal applied on `MonetaryValue` and `Frequency`
- Outlier customers are excluded from clustering to improve cluster quality

### 4. Clustering — K-Means
- Features standardized with `StandardScaler`
- Optimal `k` selected using the **Elbow Method** and **Silhouette Score**
- Final model: **K-Means with k=4 clusters**
- Clusters visualized in 3D RFM space

### 5. Classification
K-Means cluster labels are used as targets for supervised classifiers:

| Model             | Notes                                          |
|-------------------|------------------------------------------------|
| Random Forest     | `n_estimators=200`, `max_depth=10`             |
| K-Nearest Neighbors (KNN) | `k=5`                                |
| Decision Tree     | Default parameters                             |
| SVM               | Linear kernel, `C=1`                           |
| XGBoost           | `n_estimators=200`, `learning_rate=0.1`        |
| LightGBM          | Default parameters                             |

Models are evaluated on **Accuracy**, **Confusion Matrix**, **Classification Report**, **MAE**, **MSE**, and **R² Score**.

### 6. Hyperparameter Tuning
- `GridSearchCV` applied to Random Forest over `n_estimators`, `max_depth`, and `min_samples_split`
- 5-fold cross-validation for robust evaluation

### 7. Model Export
- Best model (SVM) saved using `joblib` as `model.pkl`

---

## 📊 Results

All six classifiers are compared in a bar chart. Models generally achieve high accuracy given the well-separated RFM clusters. Final accuracy values are printed and plotted at the end of the notebook.

---

## 🗂 Project Structure

```
├── rel_dm_ml.ipynb       # Main analysis notebook
├── model.pkl             # Saved best model (SVM)
├── OnlineRetail.csv      # Raw dataset (not included — see Dataset section)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm joblib
```

### Run

1. Download the [Online Retail dataset](https://archive.ics.uci.edu/ml/datasets/online+retail) and place `OnlineRetail.csv` in the project root.
2. Open `rel_dm_ml.ipynb` in Jupyter and run all cells.

---

## 🧰 Tech Stack

- **Python 3.x**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — preprocessing, clustering, classification, evaluation
- `xgboost`, `lightgbm` — gradient boosting classifiers
- `joblib` — model serialization
- `streamlit` *(prototype)* — interactive prediction UI

---

## 📬 Contact

Feel free to open an issue or submit a pull request for improvements.
