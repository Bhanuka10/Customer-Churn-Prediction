import glob
import os
import pickle
import sys
import time
import warnings
from typing import Optional

# Suppress MLflow deprecation warnings without changing API calls
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

import mlflow
import mlflow.sklearn
try:
    from mlflow.exceptions import MlflowException
except Exception:  # Fallback for test stubs without mlflow.exceptions
    MlflowException = Exception
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
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DEFAULT_MLFLOW_URI = f"file:{os.path.join(PROJECT_ROOT, 'notebooks', 'mlruns')}"
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_URI)

def _read_meta_value(lines, key):
    prefix = f"{key}:"
    for line in lines:
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None

def _find_model_id(model_name, stage):
    registry_dir = os.path.join(PROJECT_ROOT, "notebooks", "mlruns", "models", model_name)
    for meta_path in glob.glob(os.path.join(registry_dir, "version-*", "meta.yaml")):
        with open(meta_path, "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle.readlines()]
        current_stage = _read_meta_value(lines, "current_stage")
        if current_stage == stage:
            return _read_meta_value(lines, "model_id")
    return None

def _load_model_from_artifacts():
    """Load model directly from the local artifacts directory using pickle.

    Returns the loaded model if artifacts/model.pkl exists, otherwise None.
    This is the primary load path in production where the Docker image ships
    the serialised model at /app/artifacts/model.pkl.
    """
    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model from {model_path}")
    return model

@app.on_event("startup")
def load_model():
    global MODEL, SCALER

    if os.getenv("SKIP_MODEL_LOAD", "").lower() in {"1", "true", "yes"}:
        print("Skipping model load (SKIP_MODEL_LOAD set).")
        return

    # ── Primary path: local artifacts directory (production / Docker) ──
    MODEL = _load_model_from_artifacts()

    # ── Fallback: MLflow registry (development / testing) ─────────────
    if MODEL is None:
        mlflow.set_tracking_uri(MLFLOW_URI)
        try:
            client = mlflow.MlflowClient()
            versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if versions:
                mv = versions[0]
                run_uri = f"runs:/{mv.run_id}/model"
                MODEL = mlflow.sklearn.load_model(run_uri)
                print(f"Loaded model from MLflow: {MODEL_NAME} v{mv.version} (run {mv.run_id})")
            else:
                raise MlflowException(f"No Production-stage version found for model '{MODEL_NAME}'")
        except MlflowException as exc:
            if "No such artifact" not in str(exc):
                print(f"Warning: MLflow error loading model: {exc}")
            else:
                model_id = _find_model_id(MODEL_NAME, "Production")
                if model_id:
                    pattern = os.path.join(
                        PROJECT_ROOT, "notebooks", "mlruns", "*", "models", model_id, "artifacts"
                    )
                    matches = glob.glob(pattern)
                    if matches:
                        try:
                            MODEL = mlflow.sklearn.load_model(matches[0])
                            print(f"Loaded model from MLflow run artifacts: {matches[0]}")
                        except Exception as inner_exc:
                            print(f"Warning: Could not load model from MLflow run artifacts: {inner_exc}")
                    else:
                        print(f"Warning: No MLflow run artifacts found for model '{MODEL_NAME}'")
                else:
                    print(f"Warning: No Production-stage entry found in MLflow registry for '{MODEL_NAME}'")
        except Exception as exc:
            print(f"Warning: Unexpected error loading model from MLflow: {exc}")


    if MODEL is None:
        print(
            "Warning: Model could not be loaded from artifacts or MLflow. "
            "The /predict endpoint will return 503 until a model is available."
        )

    SCALER = load_scaler("artifacts/scaler.pkl")
    print("Loaded scaler from artifacts/scaler.pkl")

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
    model_config = {"protected_namespaces": ()}

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