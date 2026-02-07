from fastapi.testclient import TestClient
from app import app

# Create a test client
client = TestClient(app)

def test_root_endpoint():
    """Check if the API is alive."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"

def test_prediction_endpoint():
    """Check if the model returns a valid prediction."""
    # Dummy data matches the columns used in train.py
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
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data