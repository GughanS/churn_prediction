// FastAPI app.py Python script template (serves model.pkl as REST API with /predict)
const fs = require("fs")
const appCode = `
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
model = joblib.load('model.pkl')

class Input(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: str

class Output(BaseModel):
    prediction: str
    probability: float
    risk_level: str

@app.post('/predict', response_model=Output)
def predict(input: Input):
    features = [[input.tenure, input.monthly_charges, input.total_charges, input.contract_type]]
    proba = model.predict_proba(features)[0][1]
    prediction = 'Churn' if proba > 0.5 else 'No Churn'
    risk_level = 'High' if proba > 0.75 else ('Medium' if proba > 0.5 else 'Low')
    return Output(
        prediction=prediction,
        probability=round(proba, 4),
        risk_level=risk_level
    )
`
fs.writeFileSync("app.py", appCode)
console.log("Python FastAPI app app.py generated. Please run it in the same directory as model.pkl in a Python 3.9+ environment.")
setContext("api_code", "app.py")
