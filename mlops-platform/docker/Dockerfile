FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY Data/ ./Data/
COPY notebooks/mlruns/ /mlflow/mlruns/

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:/mlflow/mlruns

CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]