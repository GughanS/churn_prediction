import joblib
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn

app = FastAPI(title="Churn Prediction API")

# --- DEBUG PRINT ---
print("\n" + "="*50)
print("CORS IS ENABLED - SERVER IS RELOADING")
print("="*50 + "\n")

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# --- Load Artifacts ---
try:
    # Try loading from current directory first, then root
    try:
        model = joblib.load('model.pkl')
        with open('model_features.json', 'r') as f:
            meta = json.load(f)
    except FileNotFoundError:
        # Fallback for when running from src/
        model = joblib.load('../model.pkl')
        with open('../model_features.json', 'r') as f:
            meta = json.load(f)
            
    feature_names = meta['feature_names']
except Exception as e:
    print(f"[ERROR] Could not load model: {str(e)}")
    feature_names = []

class PredictionRequest(BaseModel):
    features: Dict[str, Any]

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>Churn API is Online (Port 8000)</h1>"

@app.post("/predict")
def predict(request: PredictionRequest):
    if not feature_names:
        raise HTTPException(status_code=500, detail="Model not loaded")

    input_data = request.features
    
    # Validation
    missing_cols = [col for col in feature_names if col not in input_data]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing_cols}")

    df_input = pd.DataFrame([input_data])

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)