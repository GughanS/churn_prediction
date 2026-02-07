# Dockerfile for churn FastAPI app
FROM python:3.9-slim

WORKDIR /app

# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy ALL files (app.py, train.py, data/, etc.)
# We copy everything so the build has access to the csv data
COPY . .

# 3. Train the model during the build
# This ensures model.pkl is generated inside the container
RUN python train.py

# 4. Expose & Run
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]