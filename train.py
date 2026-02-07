import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import json
import os

# --- Configuration ---
DATA_PATH = './data/churn.csv'
MODEL_PATH = 'model.pkl'
META_PATH = 'model_features.json'
TARGET_COLUMN = 'Churn Value'

# The exact names we WANT to use in our API
REQUIRED_FEATURES = [
    'tenure', 
    'MonthlyCharges', 
    'TotalCharges', 
    'Contract', 
    'PaymentMethod'
]

# --- 1. Data Loading ---
if not os.path.exists(DATA_PATH):
    print(f"[WARN] {DATA_PATH} not found. Generating dummy data...")
    # Generate dummy data with correct columns
    df = pd.DataFrame({
        'tenure': np.random.randint(1, 72, 100),
        'MonthlyCharges': np.random.uniform(20, 120, 100),
        'TotalCharges': np.random.uniform(20, 5000, 100),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], 100),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer'], 100),
        'Churn': np.random.choice(['Yes', 'No'], 100)
    })
else:
    df = pd.read_csv(DATA_PATH)
    print(f"DEBUG: Original columns in CSV: {list(df.columns)}")

# --- 2. Smart Column Normalization ---
# Specific mappings for the IBM Telco Dataset (which matches your logs)
SPECIFIC_MAPPINGS = {
    'Tenure Months': 'tenure',
    'Monthly Charges': 'MonthlyCharges',
    'Total Charges': 'TotalCharges',
    'Payment Method': 'PaymentMethod',
    'Churn Value': 'Churn'  # The IBM dataset uses 'Churn Value' (1/0) or 'Churn Label' (Yes/No)
}

# Apply specific renaming first
df.rename(columns=SPECIFIC_MAPPINGS, inplace=True)

# General cleanup: Remove internal spaces for other potential matches if needed
# (e.g. "Monthly Charges" -> "MonthlyCharges" if not caught above)
df.columns = [c.replace(' ', '') for c in df.columns]

col_map = {c.lower(): c for c in df.columns}

# Verify Features exist (case-insensitive fallback)
for req in REQUIRED_FEATURES:
    if req not in df.columns:
        if req.lower() in col_map:
            actual_name = col_map[req.lower()]
            print(f"[INFO] Renaming '{actual_name}' to '{req}'")
            df.rename(columns={actual_name: req}, inplace=True)

# Verify Target
if TARGET_COLUMN not in df.columns:
    if TARGET_COLUMN.lower() in col_map:
        actual_name = col_map[TARGET_COLUMN.lower()]
        df.rename(columns={actual_name: TARGET_COLUMN}, inplace=True)
    # Check for 'ChurnLabel' since we stripped spaces
    elif 'ChurnLabel' in df.columns:
         print(f"[INFO] Using 'ChurnLabel' as target")
         df.rename(columns={'ChurnLabel': TARGET_COLUMN}, inplace=True)

# --- 3. Final Validation ---
missing_feats = [col for col in REQUIRED_FEATURES if col not in df.columns]
if missing_feats:
    print(f"[ERROR] Still missing columns after normalization: {missing_feats}")
    print(f"Available columns: {list(df.columns)}")
    raise ValueError("Dataset schema mismatch")

# Select and Clean
X = df[REQUIRED_FEATURES].copy()

# Handle TotalCharges (often has empty strings " ")
if 'TotalCharges' in X.columns:
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
    X['TotalCharges'] = X['TotalCharges'].fillna(0)

# Handle Target Mapping
# If the target is already numeric (0/1), use it. If it's Yes/No, map it.
print(f"Target column unique values: {df[TARGET_COLUMN].unique()}")

if df[TARGET_COLUMN].dtype == 'object':
    y = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    # Fallback for unexpected strings
    if y.isnull().any():
        print("[WARN] Target contained unknown values. forcing categorical codes.")
        y = df[TARGET_COLUMN].astype('category').cat.codes
else:
    y = df[TARGET_COLUMN]

# --- 4. Identify Types ---
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Training with: {list(X.columns)}")

# --- 5. Build & Train Pipeline ---
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

clf.fit(X, y)
joblib.dump(clf, MODEL_PATH)

# Save Metadata
with open(META_PATH, 'w') as f:
    json.dump({
        "feature_names": list(X.columns),
        "numeric": numeric_features,
        "categorical": categorical_features
    }, f)

print("Success! Model trained and saved.")