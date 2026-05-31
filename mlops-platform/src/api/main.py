import glob
import os
import pickle
import time
import warnings
from typing import Optional

import mlflow
import mlflow.sklearn
try:
    from mlflow.exceptions import MlflowException
except Exception:  # Fallback for test stubs without mlflow.exceptions
    MlflowException = Exception
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel

from ..utils.preprocessor import preprocess_input, load_scaler

# Suppress MLflow deprecation warnings without changing API calls
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

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


def _dashboard_html() -> str:
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta name="color-scheme" content="dark" />
    <title>Customer Churn Prediction Studio</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg: #07101c;
            --bg-2: #0d1730;
            --panel: rgba(11, 19, 36, 0.78);
            --panel-strong: rgba(8, 15, 29, 0.94);
            --border: rgba(181, 205, 255, 0.14);
            --border-strong: rgba(181, 205, 255, 0.26);
            --text: #f5f8ff;
            --muted: #9fb0d0;
            --accent: #73f0d0;
            --accent-2: #89a7ff;
            --danger: #ff7d96;
            --warning: #ffd36c;
            --success: #7deb9d;
            --shadow: 0 28px 90px rgba(0, 0, 0, 0.36);
            --radius: 28px;
            font-family: "Inter", system-ui, sans-serif;
        }

        * { box-sizing: border-box; }

        html { scroll-behavior: smooth; }

        body {
            margin: 0;
            min-height: 100vh;
            color: var(--text);
            background:
                radial-gradient(circle at 15% 15%, rgba(115, 240, 208, 0.16), transparent 24%),
                radial-gradient(circle at 85% 10%, rgba(137, 167, 255, 0.2), transparent 26%),
                radial-gradient(circle at 80% 82%, rgba(255, 125, 150, 0.09), transparent 22%),
                linear-gradient(155deg, var(--bg), var(--bg-2));
        }

        body::before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image: linear-gradient(rgba(255, 255, 255, 0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
            background-size: 38px 38px;
            mask-image: radial-gradient(circle at center, black, transparent 86%);
            opacity: 0.45;
        }

        .page {
            width: min(1280px, calc(100% - 28px));
            margin: 0 auto;
            padding: 24px 0 36px;
            position: relative;
            z-index: 1;
        }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            margin-bottom: 18px;
            padding: 6px 4px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            box-shadow: 0 16px 30px rgba(137, 167, 255, 0.22);
        }

        .brand-text {
            display: grid;
            gap: 3px;
        }

        .brand-text strong {
            font: 700 1rem/1.1 "Space Grotesk", Inter, sans-serif;
            letter-spacing: -0.02em;
        }

        .brand-text span,
        .eyebrow,
        .micro,
        .status-chip,
        .pill,
        .section-copy,
        .meta,
        .score-caption,
        .label,
        .footer,
        .helper {
            color: var(--muted);
        }

        .url-chip {
            padding: 10px 14px;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: rgba(255, 255, 255, 0.04);
            font-size: 0.88rem;
        }

        .hero {
            display: grid;
            grid-template-columns: 1.35fr 0.65fr;
            gap: 20px;
            margin-bottom: 20px;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            backdrop-filter: blur(18px);
            overflow: hidden;
        }

        .hero-main,
        .hero-side,
        .form-panel,
        .insight-panel {
            padding: 26px;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(115, 240, 208, 0.12);
            color: var(--accent);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-size: 0.76rem;
            font-weight: 700;
        }

        h1,
        h2,
        h3 {
            margin: 0;
            font-family: "Space Grotesk", Inter, sans-serif;
            letter-spacing: -0.04em;
        }

        h1 {
            margin-top: 16px;
            font-size: clamp(2.4rem, 5vw, 4.7rem);
            line-height: 0.95;
        }

        .lede {
            margin: 16px 0 0;
            max-width: 64ch;
            font-size: 1.02rem;
            line-height: 1.75;
            color: var(--muted);
        }

        .hero-metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 24px;
        }

        .metric {
            padding: 16px;
            border-radius: 20px;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
        }

        .metric strong {
            display: block;
            font-family: "Space Grotesk", Inter, sans-serif;
            font-size: 1.15rem;
            margin-bottom: 5px;
        }

        .metric span {
            font-size: 0.86rem;
            color: var(--muted);
            line-height: 1.55;
        }

        .hero-side {
            display: grid;
            align-content: space-between;
            gap: 18px;
            background: linear-gradient(180deg, rgba(13, 24, 49, 0.92), rgba(8, 14, 26, 0.9));
        }

        .status-chip {
            width: fit-content;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            border: 1px solid rgba(125, 235, 157, 0.22);
            background: rgba(125, 235, 157, 0.12);
            color: var(--success);
            font-size: 0.84rem;
            font-weight: 700;
        }

        .status-chip.offline {
            border-color: rgba(255, 125, 150, 0.24);
            background: rgba(255, 125, 150, 0.12);
            color: var(--danger);
        }

        .status-details {
            margin-top: 12px;
            color: var(--muted);
            line-height: 1.7;
            font-size: 0.96rem;
        }

        .support-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
        }

        .pill {
            padding: 12px 14px;
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.07);
            line-height: 1.45;
            font-size: 0.92rem;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 20px;
            align-items: start;
        }

        .form-panel h2,
        .insight-panel h2 {
            font-size: 1.4rem;
            margin-bottom: 8px;
        }

        .section-copy {
            margin: 0 0 20px;
            line-height: 1.65;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
        }

        .field {
            display: grid;
            gap: 8px;
        }

        .field.full {
            grid-column: 1 / -1;
        }

        label {
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            color: #dfe8fb;
        }

        input,
        select,
        button {
            width: 100%;
            border-radius: 16px;
            border: 1px solid rgba(181, 205, 255, 0.16);
            background: rgba(7, 13, 24, 0.96);
            color: var(--text);
            padding: 13px 14px;
            font: inherit;
            outline: none;
            transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s ease;
        }

        input:focus,
        select:focus {
            border-color: rgba(115, 240, 208, 0.8);
            box-shadow: 0 0 0 4px rgba(115, 240, 208, 0.12);
        }

        button {
            cursor: pointer;
            font-weight: 800;
        }

        .actions {
            display: flex;
            gap: 12px;
            margin-top: 18px;
        }

        .actions .primary {
            background: linear-gradient(135deg, var(--accent), var(--accent-2));
            color: #03111d;
            box-shadow: 0 18px 36px rgba(115, 240, 208, 0.22);
            border: none;
        }

        .actions .secondary {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: var(--text);
        }

        .actions button:hover {
            transform: translateY(-1px);
        }

        .insight-panel {
            display: grid;
            gap: 16px;
            background: linear-gradient(180deg, rgba(11, 19, 36, 0.9), rgba(7, 13, 24, 0.96));
        }

        .error {
            display: none;
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255, 125, 150, 0.1);
            border: 1px solid rgba(255, 125, 150, 0.2);
            color: #ffcad5;
            line-height: 1.6;
            white-space: pre-wrap;
        }

        .score-card {
            padding: 20px;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(115, 240, 208, 0.12), rgba(137, 167, 255, 0.1));
            border: 1px solid rgba(181, 205, 255, 0.16);
        }

        .score-top {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 16px;
        }

        .score-value {
            display: block;
            font-family: "Space Grotesk", Inter, sans-serif;
            font-size: clamp(3rem, 8vw, 4.4rem);
            line-height: 0.92;
            letter-spacing: -0.06em;
        }

        .score-caption {
            margin-top: 10px;
            font-size: 0.92rem;
            line-height: 1.55;
        }

        .risk {
            width: fit-content;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
            font-size: 0.78rem;
        }

        .risk.low { background: rgba(125, 235, 157, 0.14); color: var(--success); }
        .risk.medium { background: rgba(255, 211, 108, 0.14); color: var(--warning); }
        .risk.high { background: rgba(255, 125, 150, 0.14); color: var(--danger); }

        .meter {
            margin-top: 16px;
            height: 14px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .meter-fill {
            height: 100%;
            width: 0%;
            border-radius: inherit;
            background: linear-gradient(90deg, var(--success), var(--warning), var(--danger));
            transition: width 0.35s ease;
        }

        .info-stack {
            display: grid;
            gap: 12px;
        }

        .kv {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            padding: 14px 16px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.07);
        }

        .kv span { color: var(--muted); }

        .kv strong { font-weight: 700; }

        .recommendation {
            padding: 16px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.07);
            line-height: 1.7;
        }

        .micro-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }

        .micro-card {
            padding: 14px;
            border-radius: 18px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.07);
        }

        .micro-card strong {
            display: block;
            font-size: 1rem;
            margin-top: 2px;
            font-family: "Space Grotesk", Inter, sans-serif;
        }

        .footer {
            text-align: center;
            margin-top: 18px;
            padding: 8px 0 2px;
            font-size: 0.86rem;
        }

        .fade-in {
            animation: rise 0.58s ease both;
        }

        @keyframes rise {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @media (max-width: 1040px) {
            .hero,
            .main-grid {
                grid-template-columns: 1fr;
            }

            .hero-metrics,
            .micro-grid {
                grid-template-columns: 1fr;
            }

            .support-grid,
            .grid {
                grid-template-columns: 1fr;
            }

            .actions {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <main class="page">
        <div class="topbar">
            <div class="brand">
                <div class="brand-mark" aria-hidden="true"></div>
                <div class="brand-text">
                    <strong>Customer Churn Prediction Studio</strong>
                    <span>FastAPI inference UI on Railway</span>
                </div>
            </div>
            <div class="url-chip">customer-churn-prediction-production-df94.up.railway.app</div>
        </div>

        <section class="hero">
            <div class="panel hero-main fade-in">
                <div class="eyebrow">Customer churn prediction · Live prediction workspace</div>
                <h1>Turn raw customer features into a clear churn decision.</h1>
                <p class="lede">
                    This interface gives you a cleaner way to inspect the model, submit a customer profile, and read the result in a format that is easier to act on.
                    It keeps the API contract the same while making the deployed URL feel like a real product instead of a JSON endpoint.
                </p>

                <div class="hero-metrics">
                    <div class="metric">
                        <strong>/predict</strong>
                        <span>Submits the customer profile and returns churn probability, risk level, and model version.</span>
                    </div>
                    <div class="metric">
                        <strong>/health</strong>
                        <span>Shows whether the service and model are available before you start predicting.</span>
                    </div>
                    <div class="metric">
                        <strong>/metrics</strong>
                        <span>Prometheus output for deployment monitoring and operational visibility.</span>
                    </div>
                </div>
            </div>

            <aside class="panel hero-side fade-in">
                <div>
                    <div id="service-status" class="status-chip offline">Checking service...</div>
                    <div class="status-details" id="service-details">Loading model status from the API.</div>
                </div>

                <div class="support-grid">
                    <div class="pill">Focus on tenure, contract type, monthly charges, and internet service for the strongest prediction signal.</div>
                    <div class="pill">Use the sample customer button to quickly test the model and compare changes across profiles.</div>
                </div>
            </aside>
        </section>

        <section class="main-grid">
            <div class="panel form-panel fade-in">
                <h2>Customer profile</h2>
                <p class="section-copy">Fill in the details below and generate a prediction in one click.</p>

                <form id="prediction-form">
                    <div class="grid">
                        <div class="field">
                            <label for="gender">Gender</label>
                            <select id="gender" name="gender">
                                <option>Female</option>
                                <option>Male</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="SeniorCitizen">Senior Citizen</label>
                            <select id="SeniorCitizen" name="SeniorCitizen">
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="Partner">Partner</label>
                            <select id="Partner" name="Partner">
                                <option>Yes</option>
                                <option>No</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="Dependents">Dependents</label>
                            <select id="Dependents" name="Dependents">
                                <option>No</option>
                                <option>Yes</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="tenure">Tenure</label>
                            <input id="tenure" name="tenure" type="number" min="0" step="1" value="1" />
                        </div>
                        <div class="field">
                            <label for="PhoneService">Phone Service</label>
                            <select id="PhoneService" name="PhoneService">
                                <option>Yes</option>
                                <option>No</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="MultipleLines">Multiple Lines</label>
                            <select id="MultipleLines" name="MultipleLines">
                                <option>No phone service</option>
                                <option>No</option>
                                <option>Yes</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="InternetService">Internet Service</label>
                            <select id="InternetService" name="InternetService">
                                <option>DSL</option>
                                <option>Fiber optic</option>
                                <option>No</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="OnlineSecurity">Online Security</label>
                            <select id="OnlineSecurity" name="OnlineSecurity">
                                <option>No</option>
                                <option>Yes</option>
                                <option>No internet service</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="OnlineBackup">Online Backup</label>
                            <select id="OnlineBackup" name="OnlineBackup">
                                <option>Yes</option>
                                <option>No</option>
                                <option>No internet service</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="DeviceProtection">Device Protection</label>
                            <select id="DeviceProtection" name="DeviceProtection">
                                <option>No</option>
                                <option>Yes</option>
                                <option>No internet service</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="TechSupport">Tech Support</label>
                            <select id="TechSupport" name="TechSupport">
                                <option>No</option>
                                <option>Yes</option>
                                <option>No internet service</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="StreamingTV">Streaming TV</label>
                            <select id="StreamingTV" name="StreamingTV">
                                <option>No</option>
                                <option>Yes</option>
                                <option>No internet service</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="StreamingMovies">Streaming Movies</label>
                            <select id="StreamingMovies" name="StreamingMovies">
                                <option>No</option>
                                <option>Yes</option>
                                <option>No internet service</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="Contract">Contract</label>
                            <select id="Contract" name="Contract">
                                <option>Month-to-month</option>
                                <option>One year</option>
                                <option>Two year</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="PaperlessBilling">Paperless Billing</label>
                            <select id="PaperlessBilling" name="PaperlessBilling">
                                <option>Yes</option>
                                <option>No</option>
                            </select>
                        </div>
                        <div class="field full">
                            <label for="PaymentMethod">Payment Method</label>
                            <select id="PaymentMethod" name="PaymentMethod">
                                <option>Electronic check</option>
                                <option>Mailed check</option>
                                <option>Bank transfer (automatic)</option>
                                <option>Credit card (automatic)</option>
                            </select>
                        </div>
                        <div class="field">
                            <label for="MonthlyCharges">Monthly Charges</label>
                            <input id="MonthlyCharges" name="MonthlyCharges" type="number" min="0" step="0.01" value="29.85" />
                        </div>
                        <div class="field">
                            <label for="TotalCharges">Total Charges</label>
                            <input id="TotalCharges" name="TotalCharges" type="number" min="0" step="0.01" value="29.85" />
                        </div>
                    </div>

                    <div class="actions">
                        <button type="submit" class="primary" id="predict-button">Predict churn</button>
                        <button type="button" class="secondary" id="load-sample">Load sample customer</button>
                    </div>
                </form>
            </div>

            <aside class="panel insight-panel fade-in">
                <div>
                    <h2>Prediction result</h2>
                    <p class="section-copy">The model output will update live after submission.</p>
                </div>

                <div class="error" id="error-box"></div>

                <div class="score-card">
                    <div class="score-top">
                        <div>
                            <span class="score-value" id="probability">--</span>
                            <div class="score-caption">Churn probability from the current model</div>
                        </div>
                        <div id="risk-level" class="risk low">Waiting</div>
                    </div>
                    <div class="meter" aria-hidden="true">
                        <div class="meter-fill" id="meter-fill"></div>
                    </div>
                </div>

                <div class="info-stack">
                    <div class="kv"><span>Prediction</span><strong id="prediction">Waiting</strong></div>
                    <div class="kv"><span>Model version</span><strong id="model-version">Waiting</strong></div>
                    <div class="kv"><span>Confidence</span><strong id="confidence">Waiting</strong></div>
                </div>

                <div class="recommendation" id="recommendation">
                    Submit a customer profile to get a recommendation and next-step guidance.
                </div>

                <div class="micro-grid">
                    <div class="micro-card">
                        <span class="micro">Status</span>
                        <strong id="service-mini">Checking...</strong>
                    </div>
                    <div class="micro-card">
                        <span class="micro">Endpoint</span>
                        <strong>/predict</strong>
                    </div>
                    <div class="micro-card">
                        <span class="micro">Model</span>
                        <strong id="model-mini">Waiting</strong>
                    </div>
                </div>
            </aside>
        </section>

        <div class="footer">FastAPI + Railway prediction studio</div>
    </main>

    <script>
        const sampleCustomer = {
            gender: 'Female',
            SeniorCitizen: 0,
            Partner: 'Yes',
            Dependents: 'No',
            tenure: 1,
            PhoneService: 'No',
            MultipleLines: 'No phone service',
            InternetService: 'DSL',
            OnlineSecurity: 'No',
            OnlineBackup: 'Yes',
            DeviceProtection: 'No',
            TechSupport: 'No',
            StreamingTV: 'No',
            StreamingMovies: 'No',
            Contract: 'Month-to-month',
            PaperlessBilling: 'Yes',
            PaymentMethod: 'Electronic check',
            MonthlyCharges: 29.85,
            TotalCharges: 29.85,
        };

        const form = document.getElementById('prediction-form');
        const predictButton = document.getElementById('predict-button');
        const errorBox = document.getElementById('error-box');
        const probability = document.getElementById('probability');
        const prediction = document.getElementById('prediction');
        const riskLevel = document.getElementById('risk-level');
        const modelVersion = document.getElementById('model-version');
        const confidence = document.getElementById('confidence');
        const recommendation = document.getElementById('recommendation');
        const meterFill = document.getElementById('meter-fill');
        const serviceStatus = document.getElementById('service-status');
        const serviceDetails = document.getElementById('service-details');
        const serviceMini = document.getElementById('service-mini');
        const modelMini = document.getElementById('model-mini');
        const loadSample = document.getElementById('load-sample');

        function setFieldValues(values) {
            Object.entries(values).forEach(([key, value]) => {
                const field = document.getElementById(key);
                if (field) {
                    field.value = value;
                }
            });
        }

        function readPayload() {
            return {
                gender: document.getElementById('gender').value,
                SeniorCitizen: Number(document.getElementById('SeniorCitizen').value),
                Partner: document.getElementById('Partner').value,
                Dependents: document.getElementById('Dependents').value,
                tenure: Number(document.getElementById('tenure').value),
                PhoneService: document.getElementById('PhoneService').value,
                MultipleLines: document.getElementById('MultipleLines').value,
                InternetService: document.getElementById('InternetService').value,
                OnlineSecurity: document.getElementById('OnlineSecurity').value,
                OnlineBackup: document.getElementById('OnlineBackup').value,
                DeviceProtection: document.getElementById('DeviceProtection').value,
                TechSupport: document.getElementById('TechSupport').value,
                StreamingTV: document.getElementById('StreamingTV').value,
                StreamingMovies: document.getElementById('StreamingMovies').value,
                Contract: document.getElementById('Contract').value,
                PaperlessBilling: document.getElementById('PaperlessBilling').value,
                PaymentMethod: document.getElementById('PaymentMethod').value,
                MonthlyCharges: Number(document.getElementById('MonthlyCharges').value),
                TotalCharges: Number(document.getElementById('TotalCharges').value),
            };
        }

        function riskClass(level) {
            return (level || 'LOW').toLowerCase();
        }

        function recommendationText(risk, proba) {
            if (risk === 'HIGH') {
                return `High risk detected at ${Math.round(proba * 100)}%. Review the contract, discount structure, and service add-ons before renewal outreach.`;
            }
            if (risk === 'MEDIUM') {
                return `Moderate risk detected at ${Math.round(proba * 100)}%. A retention offer or service bundling could reduce churn pressure.`;
            }
            return `Low risk detected at ${Math.round(proba * 100)}%. Keep the current experience stable and monitor for contract changes.`;
        }

        function setError(message) {
            errorBox.textContent = message;
            errorBox.style.display = 'block';
        }

        function clearError() {
            errorBox.textContent = '';
            errorBox.style.display = 'none';
        }

        function setLoading(isLoading) {
            predictButton.disabled = isLoading;
            predictButton.textContent = isLoading ? 'Predicting…' : 'Predict churn';
        }

        function updateResult(data) {
            const proba = Number(data.churn_probability) || 0;
            const percent = Math.round(proba * 100);
            const risk = String(data.risk_level || 'LOW').toUpperCase();

            probability.textContent = `${percent}%`;
            prediction.textContent = data.churn_prediction ? 'Churn likely' : 'Customer likely to stay';
            riskLevel.textContent = risk;
            riskLevel.className = `risk ${riskClass(risk)}`;
            modelVersion.textContent = data.model_version || 'Unknown';
            modelMini.textContent = data.model_version || 'Unknown';
            confidence.textContent = `${Math.max(55, 100 - Math.abs(50 - percent) * 2)}% relative confidence`;
            meterFill.style.width = `${percent}%`;
            recommendation.textContent = recommendationText(risk, proba);
            serviceMini.textContent = 'Model active';
        }

        async function refreshStatus() {
            try {
                const response = await fetch('/health');
                const data = await response.json();
                const online = response.ok && data.loaded;
                serviceStatus.textContent = online ? 'Model online' : 'Model unavailable';
                serviceStatus.classList.toggle('offline', !online);
                serviceDetails.textContent = online
                    ? `Service is healthy and serving ${data.model}.`
                    : 'API is reachable, but the model is not loaded yet.';
                serviceMini.textContent = online ? 'Healthy' : 'Unavailable';
                if (online) {
                    modelMini.textContent = data.model || 'Loaded';
                }
            } catch (error) {
                serviceStatus.textContent = 'Offline';
                serviceStatus.classList.add('offline');
                serviceDetails.textContent = 'Unable to reach the health endpoint.';
                serviceMini.textContent = 'Offline';
            }
        }

        loadSample.addEventListener('click', () => {
            setFieldValues(sampleCustomer);
            clearError();
        });

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            clearError();
            setLoading(true);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(readPayload()),
                });

                const data = await response.json();
                if (!response.ok) {
                    throw new Error(data.detail || 'Prediction failed');
                }

                updateResult(data);
            } catch (error) {
                setError(error.message || 'Prediction request failed');
            } finally {
                setLoading(false);
            }
        });

        refreshStatus();
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def root():
        return HTMLResponse(content=_dashboard_html())

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