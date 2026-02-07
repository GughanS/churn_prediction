// FastAPI app.py Python script template (serves model.pkl as REST API with /predict)
const fs = require("fs")
const appCode = `
import joblib
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI(title="Churn Prediction API")

# --- Load Artifacts ---
try:
    model = joblib.load('model.pkl')
    with open('model_features.json', 'r') as f:
        meta = json.load(f)
        feature_names = meta['feature_names']
except FileNotFoundError:
    print("[ERROR] Model files not found. Run train.py first.")
    feature_names = []

# --- Input Validation ---
class PredictionRequest(BaseModel):
    # We accept a dictionary of features
    features: Dict[str, Any]

    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "tenure": 12,
                    "MonthlyCharges": 70.5,
                    "TotalCharges": 840.0,
                    "Contract": "Month-to-month",
                    "PaymentMethod": "Electronic check"
                }
            }
        }

@app.get("/")
def home():
    return {"status": "alive", "model_features": feature_names}

@app.post("/predict")
def predict(request: PredictionRequest):
    if not feature_names:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 1. Extract data
    input_data = request.features

    # 2. Validation: Ensure all required columns are present
    missing_cols = [col for col in feature_names if col not in input_data]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing_cols}")

    # 3. Convert to DataFrame (Crucial for ColumnTransformer)
    # The model expects a DataFrame with named columns, not just a list of numbers
    df_input = pd.DataFrame([input_data])

    # 4. Predict
    try:
        prediction_idx = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1]
        
        return {
            "prediction": "Churn" if prediction_idx == 1 else "No Churn",
            "probability": round(float(probability), 4),
            "risk_level": "High" if probability > 0.7 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# INPUT SCHEMA: Expects dict with feature keys matching training, as chosen dynamically in training step.
`
fs.writeFileSync("app.py", appCode)
console.log("Python FastAPI app app.py generated. Please run it in the same directory as model.pkl and model_features.json in a Python 3.9+ environment.")
setContext("api_code", "app.py")
