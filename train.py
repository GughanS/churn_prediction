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

# Define the exact features we want to use (Validation consistency)
SELECTED_FEATURES = [
    'tenure', 
    'MonthlyCharges', 
    'TotalCharges', 
    'Contract', 
    'PaymentMethod'
]

# --- 1. Data Loading ---
if not os.path.exists(DATA_PATH):
    print(f"[WARN] {DATA_PATH} not found. Generating dummy data...")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
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

# --- 2. Preprocessing ---
# Clean Target
if TARGET_COLUMN not in df.columns:
    # Handle capitalization mismatch (e.g., 'churn' vs 'Churn')
    cols_lower = {c.lower(): c for c in df.columns}
    if TARGET_COLUMN.lower() in cols_lower:
        TARGET_COLUMN = cols_lower[TARGET_COLUMN.lower()]
    else:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

y = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})
# Fallback if mapping failed
if y.isnull().any():
    y = df[TARGET_COLUMN].astype('category').cat.codes

# Force consistency: Select only the features we expect
# This prevents the '20 columns vs 5 columns' error
missing_feats = [col for col in SELECTED_FEATURES if col not in df.columns]
if missing_feats:
    raise ValueError(f"Dataset missing required features: {missing_feats}")

X = df[SELECTED_FEATURES].copy()

# Handle TotalCharges if it's a string (common in Telco dataset)
if X['TotalCharges'].dtype == 'object':
    X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

# Identify Feature Types dynamically
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Training with features: {list(X.columns)}")

# --- 3. Build Pipeline ---
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

# --- 4. Train & Save ---
clf.fit(X, y)
joblib.dump(clf, MODEL_PATH)

# Save metadata
with open(META_PATH, 'w') as f:
    json.dump({
        "feature_names": list(X.columns),
        "numeric": numeric_features,
        "categorical": categorical_features
    }, f)

print("Success! Model trained and saved.")