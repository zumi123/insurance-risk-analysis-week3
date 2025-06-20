# Task 4 Report: Predictive Modeling for Risk-Based Insurance Pricing

## Objective
To build and evaluate predictive models for:
- **Claim Severity** (`TotalClaims`)
- **Premium Optimization** (`CalculatedPremiumPerTerm`)
- **Probability of Claim** (`HasClaim`)

## 1. Data Preparation
- Loaded and cleaned the `MachineLearningRating_v3.txt` dataset.
- Removed rows with `NaN`, `inf`, or `-inf` values.
- One-hot encoded categorical variables.
- Feature engineered `Margin` and `HasClaim`.

## 2. Claim Severity Modeling (on rows where `TotalClaims > 0`)
| Model             | RMSE     | R²     |
|------------------|----------|--------|
| Linear Regression| **0.01** | **1.00** |
| Random Forest     | 5947.00  | 0.87   |
| XGBoost           | 3833.08  | 0.95   |

> **LIME (Local Interpretable Model-Agnostic Explanations)** was used to interpret the XGBoost model. The explanation was saved as `output/lime_claim_severity_explanation.html`.

## Example LIME Output
- **Prediction**: 669.85
- **Intercept**: 668.06
- This shows how local features impact the predicted claim amount.

## 3. Premium Prediction (Pricing Model)
- Target: `CalculatedPremiumPerTerm`
- Best model: XGBoost
  - **RMSE**: 32.57
  - **R²**: 0.98

## 4. Claim Probability Model (Binary Classification)
- Target: `HasClaim`
- Best model: XGBoost
  - **Accuracy**: 1.00
  - **Precision**: 0.71
  - **Recall**: 1.00
  - **F1-score**: 0.83
  - **AUC**: 1.00

## Key Takeaways
- All models performed well, with Linear Regression achieving near-perfect results on cleaned data.
- **LIME** helped us interpret local predictions of the regression model, enhancing trust and explainability.
- The classification model for claim prediction shows high potential for **risk-based pricing** strategies.

## Output Artifacts
- ✅ `task4_claim_severity_model.py`
- ✅ `output/lime_claim_severity_explanation.html`
- ✅ Model metrics logged in console
- ✅ Cleaned and encoded dataset used
- ✅ Premium and risk models implemented
