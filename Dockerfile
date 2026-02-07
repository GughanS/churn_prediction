# Dockerfile for churn FastAPI app
FROM python:3.9-slim

WORKDIR /src/app

# 1. Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy ALL files (includes src/ folder)
COPY . .

# 3. Train the model during the build
# This ensures model.pkl is generated inside the container
RUN python train.py

# 4. Expose the port
EXPOSE 8000

# 5. START COMMAND (CRITICAL FIX)
# We use "src.app:app" because app.py is inside the src folder
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]