import sys
import os
from fastapi.testclient import TestClient

# Ensure the root directory is in sys.path so we can import 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Explicitly import from src.app since you confirmed it is inside src/
from src.app import app

client = TestClient(app)

def test_root_endpoint():
    """Check if the API is alive (HTML response)."""
    response = client.get("/")
    assert response.status_code == 200
    # FIX: Check text content instead of JSON because the root returns HTML now
    assert "Churn API is Online" in response.text

def test_prediction_endpoint():
    """Check if the model returns a valid prediction."""
    payload = {
        "features": {
            "tenure": 12,
            "MonthlyCharges": 70.5,
            "TotalCharges": 840.0,
            "Contract": "Month-to-month",
            "PaymentMethod": "Electronic check"
        }
    }
    
    response = client.post("/predict", json=payload)
    
    # Assertions
    assert response.status_code == 200, f"Request failed: {response.text}"
    
    data = response.json()
    assert "prediction" in data
    assert "probability" in data