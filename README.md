# End-to-End Customer Churn Prediction MLOps Pipeline

A production-ready Machine Learning API that predicts customer churn in real-time. Designed with Shift-Left engineering principles, featuring automated testing, containerization, and cloud deployment.

## Architecture

The system follows a microservice architecture automated via GitHub Actions:

![alt text](<Untitled diagram-2026-02-07-062947.png>)


## Features

Auto-Healing Data Pipeline: Automatically handles missing columns or schema mismatches (e.g., IBM Telco vs. Kaggle datasets).

Reproducibility: Scikit-Learn Pipelines ensure preprocessing logic (scaling, encoding) matches exactly between training and inference.

CI/CD Automation: Every commit triggers a full suite of unit tests and model validation checks.

Containerized: Runs identically on local Dev machines and Production servers using Docker.

## Tech Stack

ML Core: Scikit-Learn, Pandas, NumPy

API: FastAPI (Asynchronous Python web server)

DevOps: Docker, GitHub Actions

Testing: Pytest

## How to Run Locally

Clone the repo

`git clone [https://github.com/YOUR_USERNAME/churn-mlops.git](https://github.com/YOUR_USERNAME/churn-mlops.git)
cd churn-mlops`


Run with Docker (Recommended)

`docker build -t churn-api .
docker run -p 8000:8000 churn-api`


## Test the API
Open http://localhost:8000/docs to verify.

## API Usage

Endpoint: `POST /predict`

Payload:
```
{
  "features": {
    "tenure": 12,
    "MonthlyCharges": 70.5,
    "TotalCharges": 840.0,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
  }
}
```

Response:
```
{
  "prediction": "No Churn",
  "probability": 0.13,
  "risk_level": "Low"
}
```