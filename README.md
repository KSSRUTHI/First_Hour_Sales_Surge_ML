#  Sales Surge Prediction Pipeline

This repository contains a complete machine learning pipeline to predict **first-hour sales surges** for products listed on e-commerce platforms using structured product metadata scraped from the web.

## 📂 Project Structure

- `sales_surge_prediction_pipeline.ipynb`: Main Jupyter notebook with full implementation from preprocessing to evaluation.
- `strategically_balanced_dataset.xlsx`: Cleaned and engineered dataset with 5,000 product entries.

---

##  Objective

To predict whether a newly listed product will experience a **sales surge** in the first hour of launch based on key features such as pricing, brand, visibility, and user engagement signals (e.g., reviews, Prime tag, Amazon's Choice).

---

##  Dataset Overview

- Total Records: 5,000
- Target Variable: `surge` (1 = surge in first-hour sales, 0 = no surge)
- Key Features:
  - `price`, `rating`, `reviews`
  - `is_prime`, `is_amazon_choice`
  - `cart_adds`, `velocity_score`
  - `age_tier`, `brand`, `category`, `price_category`

---

##  Preprocessing & Feature Engineering

- **Missing Values:** Imputed using median (for `cart_adds`) and mode (for `price_category`)
- **Noise Injection:** Gaussian noise added to `price` (σ=2) and `rating` (σ=0.1)
- **Label Noise Simulation:** 2% of surge labels were flipped to mimic real-world uncertainty
- **Encoding:** Label Encoding used for categorical features
- **Imbalance Handling:** SMOTE applied to balance classes
- **Normalization:** z-score normalization used for models sensitive to scaling

---

##  Models Used

| Model              | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-------------------|----------|-----------|--------|----------|---------|
| Random Forest      | 95.92%   | 97.39%    | 94.16% | 95.75%   | 0.9774  |
| XGBoost            | 95.73%   | 96.99%    | 94.16% | 95.56%   | 0.9752  |
| Logistic Regression| 88.21%   | 93.07%    | 81.96% | 87.17%   | 0.9196  |
| SVM                | 53.11%   | 51.11%    | 91.91% | 65.69%   | 0.5306  |
| Naive Bayes        | 52.98%   | 100%      | 3.71%  | 7.16%    | 0.6392  |

---

## Visualizations

-  **Correlation Heatmap:** Shows feature relationships with `surge`
-  **Top Feature Importance (Random Forest):** `reviews`, `hourly_sales`, and `price` among top predictors

---

##  Tech Stack

- **Language:** Python 3.11
- **Libraries:** `pandas`, `numpy`, `scikit-learn`, `xgboost`, `imbalanced-learn`, `matplotlib`, `seaborn`
- **Tools:** Jupyter Notebook, Excel, GridSearchCV for hyperparameter tuning

---

## Evaluation Metrics

- Accuracy
- Precision, Recall, F1-Score (especially on positive surge class)
- ROC-AUC Score
- Confusion Matrix

---

##  How to Run

1. Clone the repo and navigate into it:
   ```bash
   git clone https://github.com/KSSRUTHI/First_Hour_Sales_Surge_ML.git

