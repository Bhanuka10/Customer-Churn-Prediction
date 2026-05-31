FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY notebooks/mlruns/605156741829617426/models/m-902218b983324538b9111e02df91ff3f/artifacts/model.pkl ./artifacts/model.pkl
COPY Data/ ./Data/
COPY notebooks/mlruns/ /mlflow/mlruns/

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:/mlflow/mlruns

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]