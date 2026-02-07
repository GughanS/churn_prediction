from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"

def test_prediction_endpoint():
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
    
    # Check status code, AND print the error message if it fails
    assert response.status_code == 200, f"Request failed: {response.text}"
    
    data = response.json()
    assert "prediction" in data
    assert "probability" in data