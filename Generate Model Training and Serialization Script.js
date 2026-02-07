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
import joblib

data = pd.read_csv('./data/churn.csv')
# Features and target 
X = data.drop(columns=['Churn'])
y = data['Churn'].map({'No':0, 'Yes':1})

# Identify numeric and categorical columns
numeric_features = ['tenure', 'monthly_charges', 'total_charges']
categorical_features = ['contract_type']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
clf = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf)
])
clf.fit(X, y)
joblib.dump(clf, './model.pkl')
print('Training complete. Model artifact: model.pkl created.')
`
fs.writeFileSync("train.py", trainScript)
console.log("Python training script train.py generated. Please run this file in Python 3.9+ environment to train and save the model (model.pkl).")
setContext("model_file", "model.pkl")
