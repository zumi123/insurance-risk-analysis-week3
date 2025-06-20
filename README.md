
# Insurance Risk Analytics & Predictive Modeling Project

This repository contains a comprehensive end-to-end pipeline for insurance risk analytics and predictive modeling. The project is broken down into **4 major tasks**, each addressing a key part of the data science lifecycle in the insurance industry.

---

## Task 1: Exploratory Data Analysis (EDA)

**Goals:**
- Understand the structure and quality of the dataset.
- Uncover patterns and outliers in insurance risk and profitability.

**Key Steps:**
- Data summarization and statistics
- Univariate & bivariate analysis
- Outlier detection and visualizations
- Loss ratio analysis across demographics and geography

**Output:**
- 14+ insightful plots saved to `/plots`
- CSV summaries in `/results`

---

## Task 2: Data Version Control with DVC

**Goals:**
- Ensure reproducibility of data workflows.
- Track data changes with Git-like functionality using [DVC](https://dvc.org/).

**Key Steps:**
- Installed and initialized DVC
- Configured local remote storage
- Tracked the main dataset using `dvc add`
- Versioned the `.dvc` file with Git
- Pushed data to local DVC remote

**Output:**
- `.dvc` file for dataset
- Local DVC remote storage configured

---

## Task 3: Hypothesis Testing

**Goals:**
- Statistically validate risk hypotheses that inform segmentation strategies.

**Null Hypotheses Tested:**
1. No risk difference across provinces
2. No risk difference between zip codes
3. No significant margin difference between zip codes
4. No significant risk difference between genders

**Key Steps:**
- Defined KPIs: Claim Frequency, Claim Severity, Margin
- Grouped data for A/B testing
- Applied T-tests and Chi-squared tests
- Interpreted p-values and rejected/retained hypotheses

**Output:**
- Test results with business interpretation
- Evidence supporting potential segmentation strategies

---

## Task 4: Predictive Modeling

**Goals:**
- Build models for:
  - Claim severity (regression)
  - Premium prediction (regression)
  - Claim probability (classification)
- Interpret results using LIME

**Models Used:**
- Linear Regression
- Random Forest
- XGBoost

**Metrics:**
- RMSE, R² for regression
- Accuracy, Precision, Recall, F1, AUC for classification

**Output:**
- Predictions saved to CSV
- LIME explanation: `output/lime_claim_severity_explanation.html`
- Final report in `Task4_Report.md`

---

## Project Highlights

- End-to-end ML pipeline: EDA → version control → hypothesis testing → modeling
- Used best practices in DVC, Git, and explainable AI (LIME)
- All results are reproducible and interpretable

---



