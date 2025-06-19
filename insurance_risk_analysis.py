import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv("MachineLearningRating_v3.txt", sep="|", dtype=str, low_memory=False)
print("Data loaded")

# Convert relevant numeric columns
numeric_cols = ['TotalPremium', 'TotalClaims', 'CustomValueEstimate']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# -----------------------------
# Data Summarization
# -----------------------------

# Descriptive Statistics
print("\n Descriptive Statistics:")
print(df[numeric_cols].describe())

# Data Structure / dtypes
print("\n Data Types:")
print(df.dtypes)

# -----------------------------
# Data Quality Assessment
# -----------------------------
print("\n Missing Values:")
print(df.isnull().sum())

# -----------------------------
# Univariate Analysis
# -----------------------------

# Histograms for numerical columns
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=40, kde=True)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"plots/hist_{col}.png")

# Bar chart for categorical column
if 'CoverType' in df.columns:
    plt.figure(figsize=(8, 4))
    df['CoverType'].value_counts().plot(kind='bar')
    plt.title("CoverType Distribution")
    plt.ylabel("Count")
    plt.savefig("plots/bar_cover_type.png")

# -----------------------------
# Bivariate/Multivariate Analysis
# -----------------------------

# Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("plots/correlation_matrix.png")

# Scatter: TotalClaims vs TotalPremium
plt.figure(figsize=(6, 4))
sns.scatterplot(x='TotalPremium', y='TotalClaims', data=df, alpha=0.5)
plt.title("TotalClaims vs TotalPremium")
plt.savefig("plots/scatter_claims_premium.png")

# Scatter: Monthly TotalPremium vs TotalClaims (if TransactionMonth available)
if 'TransactionMonth' in df.columns:
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
    df_monthly = df.groupby(df['TransactionMonth'].dt.to_period("M"))[['TotalPremium', 'TotalClaims']].sum()
    df_monthly.plot(figsize=(8, 4), marker='o')
    plt.title("Monthly TotalPremium vs TotalClaims")
    plt.ylabel("Amount")
    plt.savefig("plots/monthly_trend.png")

# -----------------------------
# Data Comparison (Geography)
# -----------------------------
if 'Province' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Province', y='TotalPremium', data=df)
    plt.xticks(rotation=45)
    plt.title("Total Premium by Province")
    plt.savefig("plots/premium_by_province.png")

# -----------------------------
# Outlier Detection
# -----------------------------
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Outliers in {col}")
    plt.savefig(f"plots/box_{col}.png")

# -----------------------------
# Creative Visualizations
# -----------------------------

# 1. Loss Ratio by VehicleType
if 'VehicleType' in df.columns:
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    loss_ratio = df.groupby('VehicleType')['LossRatio'].mean().sort_values()
    plt.figure(figsize=(10, 5))
    loss_ratio.plot(kind='bar', color='teal')
    plt.title("Average Loss Ratio by Vehicle Type")
    plt.ylabel("Loss Ratio")
    plt.savefig("plots/loss_ratio_by_vehicle_type.png")

# 2. Heatmap of CoverType by Province
if 'CoverType' in df.columns and 'Province' in df.columns:
    pivot = df.pivot_table(index='Province', columns='CoverType', aggfunc='size', fill_value=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Heatmap of CoverType by Province")
    plt.savefig("plots/cover_type_by_province.png")

# 3. Pairplot of numerical relationships (small sample)
sampled_df = df[numeric_cols].dropna().sample(min(500, len(df)), random_state=42)
sns.pairplot(sampled_df)
plt.savefig("plots/pairplot_numeric.png")

print("\n EDA completed. Plots saved in 'plots/' folder.")
