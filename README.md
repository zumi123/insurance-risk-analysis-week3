# Task 1 – Exploratory Data Analysis (EDA) for Insurance Risk Analytics

This repository contains the work for **Task 1** of the project: *End-to-End Insurance Risk Analytics & Predictive Modeling*. The goal of this task is to explore and understand the insurance dataset through descriptive statistics, visualizations, and key insights.

---

## Objectives

- Summarize and assess data quality.
- Visualize distributions and relationships.
- Identify trends and outliers.
- Provide insights to guide modeling and risk analysis.

---

## EDA Breakdown

### 1. **Data Summarization**
- Descriptive statistics for: `TotalPremium`, `TotalClaims`, `CustomValueEstimate`
- Data types reviewed to confirm proper formatting

### 2. **Data Quality Assessment**
- Missing value counts per column

### 3. **Univariate Analysis**
- Histograms for numeric variables
- Bar chart for `CoverType` distribution

### 4. **Bivariate/Multivariate Analysis**
- Correlation heatmap
- Scatterplot: `TotalClaims` vs `TotalPremium`
- Monthly trend plot (if `TransactionMonth` exists)

### 5. **Geographical Comparison**
- Boxplot of `TotalPremium` by `Province`

### 6. **Outlier Detection**
- Boxplots for identifying outliers in numerical columns

### 7. **Creative Visualizations**
- Loss Ratio by `VehicleType`
- Heatmap of `CoverType` by `Province`
- Pairplot of key numeric features

---

## Key Insights

- Loss ratios vary widely across vehicle types.
- Significant outliers observed in `CustomValueEstimate` and `TotalClaims`.
- Provinces show different claim and premium distributions.
- Some categories dominate the `CoverType` distribution.

---

## Tools Used

- Python
  - pandas, numpy
  - matplotlib, seaborn

---

## How to Run

1. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

2. Run the EDA script:

   ```bash
   python eda_analysis.py
   ```

3. View results in the `plots/` folder.

---
