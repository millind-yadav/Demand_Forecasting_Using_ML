# Demand Forecasting Using Machine Learning

Predicting retail product demand using supervised machine learning — replacing traditional statistical methods (ARIMA, SARIMA, SMA) with ensemble and boosting regressors trained on historical sales data.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Models](#models)
- [Results](#results)
- [Getting Started](#getting-started)
- [Dependencies](#dependencies)

---

## Overview

Traditional demand forecasting methods require significant domain expertise and manual tuning. This project explores an ML-based alternative that learns patterns directly from historical sales data and generalizes to unseen time periods with minimal manual intervention.

The core idea is to reframe the time series problem as a **supervised regression problem** by engineering lag features from past sales, then training regression models to predict future demand.

---

## Dataset

The dataset contains weekly retail sales records across multiple stores and SKUs.

| Column | Description |
|---|---|
| `record_ID` | Unique record identifier |
| `week` | Week date of the sale |
| `store_id` | Store identifier |
| `sku_id` | Product (SKU) identifier |
| `total_price` | Actual selling price |
| `base_price` | Base / list price |
| `is_featured_sku` | Whether the product was featured (binary) |
| `is_display_sku` | Whether the product was on display (binary) |
| `units_sold` | **Target variable** — units sold that week |

---

## Project Workflow

### 1. Data Loading & Inspection
Data is loaded using the `EasyPreProcessing` library, which provides quick access to field types, missing value summaries, and basic statistics.

### 2. Handling Missing Values
Numerical columns are imputed using `EasyPreProcessing`'s built-in imputer to handle any gaps in the dataset.

### 3. Feature Engineering
- A composite `key` column is created by combining `week` and `store_id` to ensure unique identification of each time-store combination.
- Columns irrelevant to the time series prediction (`record_ID`, `sku_id`, `total_price`, `base_price`, `is_featured_sku`, `is_display_sku`) are dropped.
- `units_sold` is aggregated per `key` using `groupby().sum()`.

### 4. Converting to Supervised Learning Format
The time series is transformed into a tabular regression dataset using **lag features**:

| Feature | Description |
|---|---|
| `day_1` | Units sold 1 period ago |
| `day_2` | Units sold 2 periods ago |
| `day_3` | Units sold 3 periods ago |
| `day_4` | Units sold 4 periods ago |
| `units_sold` | Target — current period sales |

### 5. Train-Test Split
An 85% / 15% chronological split is used to preserve temporal order — no data leakage.

### 6. Model Training
Two regression models are trained and compared (see [Models](#models) below).

### 7. Hyperparameter Tuning
`RandomizedSearchCV` with 3-fold cross-validation (10 iterations) is used to tune the Random Forest, searching over:
- `n_estimators`: 50 – 250
- `max_depth`: 0 – 120 + `None`
- `max_features`: `auto`, `sqrt`
- `min_samples_split`: 2, 5, 10
- `min_samples_leaf`: 1, 2, 4
- `bootstrap`: True / False

### 8. Evaluation
Models are evaluated using **R² Score** and **Adjusted R² Score** on the held-out test set, with prediction vs. actual plots generated for visual inspection.

---

## Models

| Model | Notes |
|---|---|
| **Random Forest Regressor** | Baseline ensemble model; also tuned with `RandomizedSearchCV` |
| **XGBoost Regressor** | Gradient boosting model; compared against Random Forest |
| SVM Regressor | Planned for future scope |

---

## Results

Predictions closely track actual sales across the test window. The final comparison plot (`final.png`) shows model output vs. ground truth over a 300-record window after hyperparameter tuning.

![Predictions vs Actual Sales](https://github.com/shreyas-jk/Demand-Forecasting-Using-ML/blob/main/final.png?raw=true)

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Demand_Forecasting_Using_ML.git
cd Demand_Forecasting_Using_ML
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
Open `Forecast.ipynb` in Jupyter and run all cells sequentially.

> **Note:** Ensure `data.csv` is present in the root directory before running. The notebook references it as `train.csv` — rename or update the path as needed.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Numerical computations |
| `pandas` | Data manipulation |
| `scikit-learn` | Random Forest, RandomizedSearchCV, metrics |
| `xgboost` | XGBoost Regressor |
| `easypreprocessing` | Preprocessing utilities (imputation, EDA) |
| `matplotlib` | Plotting predictions |
| `seaborn` | Statistical visualizations |
| `scipy` | Statistical utilities |

Install all at once:
```bash
pip install numpy pandas scikit-learn xgboost easypreprocessing matplotlib seaborn scipy
```

