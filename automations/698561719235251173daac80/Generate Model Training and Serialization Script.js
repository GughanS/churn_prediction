// Python train.py template for customer churn prediction (Scikit-learn + joblib)
const fs = require("fs")
const trainScript = `
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import json
import os

# --- Configuration ---
DATA_PATH = './data/churn.csv'
MODEL_PATH = 'model.pkl'
META_PATH = 'model_features.json'
TARGET_COLUMN = 'Churn' # Changed to standard Telco name, adjustable

# --- 1. Data Loading (with Dummy Fallback) ---
if not os.path.exists(DATA_PATH):
    print(f"[WARN] {DATA_PATH} not found. Generating dummy data for testing...")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    
    # Generate dummy data
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
print(f"Columns available: {list(df.columns)}")

# Handle Target
if TARGET_COLUMN not in df.columns:
    # Try to find a likely target column if the config name is wrong
    possible_targets = [c for c in df.columns if 'churn' in c.lower()]
    if possible_targets:
        TARGET_COLUMN = possible_targets[0]
        print(f"[INFO] Switched target to '{TARGET_COLUMN}'")
    else:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

# Separate Features (X) and Target (y)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN].map({'Yes': 1, 'No': 0})

# If map failed (values weren't Yes/No), try to force convert
if y.isnull().any():
    y = df[TARGET_COLUMN].astype('category').cat.codes

# Identify Feature Types dynamically
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# --- 3. Build Pipeline ---
# Transformers
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Full Pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# --- 4. Train & Save ---
print("Training model...")
clf.fit(X, y)

print("Saving artifacts...")
joblib.dump(clf, MODEL_PATH)

# CRITICAL: Save feature names so the API knows what to expect
feature_metadata = {
    "feature_names": list(X.columns),
    "numeric": numeric_features,
    "categorical": categorical_features
}

with open(META_PATH, 'w') as f:
    json.dump(feature_metadata, f)

print(f"Success! Model saved to {MODEL_PATH}")
print(f"Metadata saved to {META_PATH}")
# Save feature names for inference
import json
with open('model_features.json', 'w') as f:
    json.dump(feature_names, f)
`
fs.writeFileSync("train.py", trainScript)
console.log("Python training script train.py generated. Please run this file in Python 3.9+ environment to train and save the model (model.pkl).")
setContext("model_file", "model.pkl")
