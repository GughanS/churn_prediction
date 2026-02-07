// Dockerfile template for containerizing FastAPI app and model artifact
const fs = require("fs")
const dockerfile = `# Dockerfile for churn FastAPI app
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py app.py
COPY model.pkl model.pkl
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
`
fs.writeFileSync("Dockerfile", dockerfile)
console.log("Dockerfile generated. Please ensure requirements.txt has fastapi, uvicorn, joblib, scikit-learn, pandas, numpy, etc.")
setContext("dockerfile", "Dockerfile")
