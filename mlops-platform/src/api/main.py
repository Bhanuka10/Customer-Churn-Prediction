import glob
import os
import sys
import time
import joblib
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.preprocessor import preprocess_input, load_scaler

app = FastAPI()

# ── Prometheus metrics ──────────────────────────────────────────
REQUEST_COUNT   = Counter('predictions_total', 'Total predictions', ['result'])
REQUEST_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')

# ── Load model at startup ────────────────────────────────────────
MODEL      = None
SCALER     = None
MODEL_NAME = os.getenv("MODEL_NAME", "churn_xgboost")
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def _find_latest_model():
    """Find the latest model.pkl file in the mlruns directory."""
    pattern = os.path.join(PROJECT_ROOT, "notebooks", "mlruns", "*", "models", "*", "artifacts", "model.pkl")
    matches = sorted(glob.glob(pattern), reverse=True)
    if matches:
        return matches[0]
    return None

@app.on_event("startup")
def load_model():
    global MODEL, SCALER
    if os.getenv("SKIP_MODEL_LOAD", "").lower() in {"1", "true", "yes"}:
        print("Skipping model load (SKIP_MODEL_LOAD set).")
        return
    
    try:
        # Try to find and load the model from local artifacts
        model_path = _find_latest_model()
        if not model_path:
            raise FileNotFoundError("No model.pkl found in mlruns directory")
        
        MODEL = joblib.load(model_path)
        print(f"Loaded model from: {model_path}")
    except Exception as exc:
        print(f"Error loading model: {exc}")
        raise
    
    SCALER = load_scaler("artifacts/scaler.pkl")
    print(f"Loaded model: {MODEL_NAME}/Production")

# ── Request / response schemas ───────────────────────────────────
class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: Optional[float] = 0.0

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    model_version: str

# ── Endpoints ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "loaded": MODEL is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    
    start = time.time()
    
    features_df  = preprocess_input(customer.dict())
    features_sc  = SCALER.transform(features_df)
    proba        = float(MODEL.predict_proba(features_sc)[0][1])
    prediction   = proba > 0.5
    risk         = "HIGH" if proba > 0.7 else "MEDIUM" if proba > 0.4 else "LOW"
    
    REQUEST_LATENCY.observe(time.time() - start)
    REQUEST_COUNT.labels(result="churn" if prediction else "no_churn").inc()
    
    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=prediction,
        risk_level=risk,
        model_version=MODEL_NAME
    )

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

