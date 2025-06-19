# src/ab_testing_pipeline.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# Create output directory for plots and results
os.makedirs("output", exist_ok=True)

# Logging list
test_logs = []

def prepare_data(df):
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    df['HasClaim'] = df['TotalClaims'] > 0
    return df

def test_province_risk(df):
    hypothesis = "[H₀₁] No risk differences across provinces (Claim Frequency)"
    print(f"\n{hypothesis}")

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x='Province', y='HasClaim')
    plt.xticks(rotation=90)
    plt.title("Claim Frequency by Province")
    plt.tight_layout()
    plt.savefig("output/claim_frequency_by_province.png")
    plt.close()

    groups = [df[df['Province'] == prov]['HasClaim'] for prov in df['Province'].dropna().unique()]
    f_stat, p_value = stats.f_oneway(*groups)
    interpret_result(hypothesis, p_value)

def test_zipcode_risk(df):
    hypothesis = "[H₀₂] No risk differences between zip codes (Claim Frequency)"
    print(f"\n{hypothesis}")

    top_zips = df['PostalCode'].value_counts().head(5).index.tolist()
    filtered = df[df['PostalCode'].isin(top_zips)]

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=filtered, x='PostalCode', y='HasClaim')
    plt.title("Claim Frequency by Top Zip Codes")
    plt.tight_layout()
    plt.savefig("output/claim_frequency_by_zipcode.png")
    plt.close()

    groups = [filtered[filtered['PostalCode'] == z]['HasClaim'] for z in top_zips]
    stat, p = stats.kruskal(*groups)
    interpret_result(hypothesis, p)

def test_zipcode_margin(df):
    hypothesis = "[H₀₃] No significant margin difference between zip codes"
    print(f"\n{hypothesis}")

    top_zips = df['PostalCode'].value_counts().head(5).index.tolist()
    zip1, zip2 = top_zips[0], top_zips[1]

    margin1 = df[df['PostalCode'] == zip1]['Margin']
    margin2 = df[df['PostalCode'] == zip2]['Margin']

    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df[df['PostalCode'].isin([zip1, zip2])], x='PostalCode', y='Margin')
    plt.title(f"Margin Comparison: {zip1} vs {zip2}")
    plt.tight_layout()
    plt.savefig("output/margin_comparison_zipcodes.png")
    plt.close()

    t_stat, p_val = stats.ttest_ind(margin1, margin2, nan_policy='omit')
    interpret_result(hypothesis, p_val)

def test_gender_risk(df):
    hypothesis = "[H₀₄] No risk difference between Women and Men (Claim Frequency)"
    print(f"\n{hypothesis}")

    female = df[df['Gender'] == 'Female']['HasClaim']
    male = df[df['Gender'] == 'Male']['HasClaim']

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df[df['Gender'].isin(['Female', 'Male'])], x='Gender', y='HasClaim')
    plt.title("Claim Frequency by Gender")
    plt.tight_layout()
    plt.savefig("output/claim_frequency_by_gender.png")
    plt.close()

    t_stat, p_val = stats.ttest_ind(female, male, nan_policy='omit')
    interpret_result(hypothesis, p_val)

def interpret_result(hypothesis, p_val, alpha=0.05):
    decision = (
        "Reject the null hypothesis: Statistically significant difference."
        if p_val < alpha
        else "Fail to reject the null hypothesis: No significant difference."
    )
    log = f"{hypothesis}\n  p-value = {p_val:.4f}\n  → {decision}\n"
    print(log)
    test_logs.append(log)

def run_all_tests(df):
    df = prepare_data(df)
    test_province_risk(df)
    test_zipcode_risk(df)
    test_zipcode_margin(df)
    test_gender_risk(df)

    # Write logs to file
    with open("output/ab_test_results.txt", "w") as f:
        f.write("A/B Testing Results\n=====================\n\n")
        f.writelines(test_logs)
