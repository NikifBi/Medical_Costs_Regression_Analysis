# Medical Insurance Cost Prediction & Model Evaluation

An end-to-end data science project predicting individual medical insurance charges using statistical linear modeling (OLS, LASSO, Ridge, Post-LASSO) and non-parametric machine learning (Random Forest). All models are evaluated on a standardized dollar scale ($\$) using target log-back-transformation and 5-fold cross-validation.

---

## 📌 Project Overview & Highlights

- **Target Metric Standardization:** Applied log-transformation ($\log(y)$) during training to satisfy regression assumptions and utilized `TransformedTargetRegressor` to ensure all model evaluations (RMSE, MAE, $R^2$) are measured directly in original dollar amounts ($\$).
- **Domain Feature Engineering:** Engineered interaction terms (`obese_smoker`, `smoker_bmi`, `smoker_age`) to capture non-linear cost multipliers directly within linear frameworks.
- **Methodological Leakage Prevention:** Model selection was performed strictly using 5-fold Cross-Validation on training data (`X_train`), reserving the holdout test set (`X_test`) exclusively for final out-of-sample evaluation.
- **Model Validation & Feature Selection:** Utilized `LassoCV` as an automated feature selector, verifying feature retention and demonstrating Post-LASSO OLS estimation to eliminate shrinkage bias.

---

## 📊 Summary of Results

| Model Architecture | CV RMSE ($) | Test RMSE ($) | Test $R^2$ | Key Feature / Advantage |
| :--- | :---: | :---: | :---: | :--- |
| **Random Forest Regressor** | **~$4,561** | **~$4,350** | **~87.2%** | **Top Performer:** Maps non-linear decision boundaries automatically. |
| **Polynomial Regression** | ~$4,887 | ~$4,551 | ~85.6% | Strong non-linear parametric baseline. |
| **Ridge Regression** | ~$4,974 | ~$4,611 | ~85.2% | Regularized linear baseline; bounds coefficient variance. |
| **LASSO / Post-LASSO OLS** | ~$4,978 | ~$4,551 | ~85.5% | **Best Interpretable Model:** Validates feature relevance; unbiased estimates. |
| **Standard OLS** | ~$4,994 | ~$4,551 | ~85.5% | Baseline parametric specification. |

---

## 💡 Key Findings & Business Recommendations

1. **Primary Deployment Model (Max Accuracy):** **Random Forest** is recommended for automated risk scoring, achieving the lowest Test RMSE (~$4,350) and highest $R^2$ (>87%).
2. **Alternative Model (Interpretability & Policy Design):** **OLS (Post-LASSO Validated)** serves as an ideal alternative when regulatory transparency is required. Thanks to domain interaction terms, it performs within **~$200** (~5.6%) of Random Forest while explicitly quantifying marginal dollar risks (e.g., being an obese smoker).

---

## 📁 Repository Structure

```text
Medical_Costs_Regression_Analysis/
├── data/
│   └── insurance.csv          # Medical insurance dataset
├── Medical Costs.ipynb        # Main analysis notebook
├── .gitignore                 # Git ignore rules
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

---

## 🛠️ Installation & Quickstart

To run this repository locally:

1. Clone the repository:
   git clone [https://github.com/NikifBi/Medical_Costs_Regression_Analysis.git](https://github.com/NikifBi/Medical_Costs_Regression_Analysis.git)
   cd Medical_Costs_Regression_Analysis

2. Install dependencies:
   pip install -r requirements.txt

3. Run the notebook:
   jupyter notebook "Medical Costs.ipynb"