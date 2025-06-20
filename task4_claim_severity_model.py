import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
from sklearn.impute import SimpleImputer

# ---------------------------
# 1. Load and Clean Data
# ---------------------------
df = pd.read_csv("MachineLearningRating_v3.txt", sep='|', dtype=str, low_memory=False)

# Convert numeric columns to float32 for memory efficiency
numeric_cols = ['TotalPremium', 'TotalClaims', 'CalculatedPremiumPerTerm', 'CustomValueEstimate']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('float32')

# Add derived columns
df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

# Drop rows with essential missing data
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['TotalPremium', 'TotalClaims'], inplace=True)

# Sample a subset to reduce memory usage
if df.shape[0] > 15000:
    df = df.sample(n=15000, random_state=42)

# Drop high-cardinality columns and IDs
high_cardinality = ['Model', 'PostalCode', 'MainCrestaZone', 'SubCrestaZone', 'make', 'mmcode']
drop_cols = ['UnderwrittenCoverID', 'PolicyID', 'VehicleIntroDate'] + high_cardinality
df.drop(columns=drop_cols, errors='ignore', inplace=True)

# Encode categorical variables
categorical_cols = df.select_dtypes(include='object').columns
if len(categorical_cols) > 0:
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
else:
    df_encoded = df.copy()

# ---------------------------
# 2. Claim Severity Model (Regression)
# ---------------------------
df_severity = df_encoded[df_encoded['TotalClaims'] > 0]
if df_severity.empty:
    raise ValueError("No rows found with TotalClaims > 0 after cleaning. Check for over-filtering.")

# Optional: sample to reduce memory usage
if df_severity.shape[0] > 10000:
    df_severity = df_severity.sample(n=10000, random_state=42)

X = df_severity.drop(columns=['TotalClaims', 'HasClaim', 'CalculatedPremiumPerTerm'])
y = df_severity['TotalClaims']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Drop columns with all NaNs to avoid shape mismatch
X_train.dropna(axis=1, how='all', inplace=True)
X_test.drop(columns=X_train.columns.symmetric_difference(X_test.columns), inplace=True)

# Impute remaining missing values
imputer = SimpleImputer(strategy='mean')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
}

print("\n Claim Severity Model Performance")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name}: RMSE={rmse:.2f}, R²={r2:.2f}")

# ---------------------------
# 3. LIME for XGBoost Severity Model
# ---------------------------
print("\n Running LIME explanation on severity model...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lime_model = models["XGBoost"]
explainer = LimeTabularExplainer(
    X_train_scaled,
    feature_names=X.columns.tolist(),
    verbose=True,
    mode='regression'
)
exp = explainer.explain_instance(X_test_scaled[5], lime_model.predict, num_features=10)
exp.save_to_file("output/lime_claim_severity_explanation.html")

# ---------------------------
# 4. Premium Prediction Model
# ---------------------------
print("\n Predicting CalculatedPremiumPerTerm")
X2 = df_encoded.drop(columns=['CalculatedPremiumPerTerm', 'HasClaim', 'TotalClaims'])
y2 = df_encoded['CalculatedPremiumPerTerm']

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

model_premium = XGBRegressor(n_estimators=100, random_state=42)
model_premium.fit(X2_train, y2_train)
y2_pred = model_premium.predict(X2_test)

print(f"Premium Prediction: RMSE={np.sqrt(mean_squared_error(y2_test, y2_pred)):.2f}, R²={r2_score(y2_test, y2_pred):.2f}")

# ---------------------------
# 5. Probability of Claim (Classification)
# ---------------------------
print("\n Predicting HasClaim")
X3 = df_encoded.drop(columns=['HasClaim', 'TotalClaims', 'CalculatedPremiumPerTerm'])
y3 = df_encoded['HasClaim']

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)

model_claim_prob = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model_claim_prob.fit(X3_train, y3_train)
y3_pred = model_claim_prob.predict(X3_test)

acc = accuracy_score(y3_test, y3_pred)
prec = precision_score(y3_test, y3_pred)
rec = recall_score(y3_test, y3_pred)
f1 = f1_score(y3_test, y3_pred)
roc = roc_auc_score(y3_test, model_claim_prob.predict_proba(X3_test)[:, 1])

print(f"Classification - HasClaim: Accuracy={acc:.2f}, Precision={prec:.2f}, Recall={rec:.2f}, F1={f1:.2f}, AUC={roc:.2f}")

print("\n All Task 4 models completed. LIME output saved to HTML.")
