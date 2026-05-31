import importlib
import os
import sys
import types
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

sample_customer = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes",
    "Dependents": "No", "tenure": 1, "PhoneService": "No",
    "MultipleLines": "No phone service", "InternetService": "DSL",
    "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
    "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}


def _install_test_stubs() -> None:
    if "mlflow" not in sys.modules:
        mlflow_module = types.ModuleType("mlflow")

        def _noop(*_args, **_kwargs):
            return None

        mlflow_module.set_tracking_uri = _noop
        sklearn_module = types.ModuleType("mlflow.sklearn")
        sklearn_module.load_model = _noop
        mlflow_module.sklearn = sklearn_module
        sys.modules["mlflow"] = mlflow_module
        sys.modules["mlflow.sklearn"] = sklearn_module

    if "utils.preprocessor" not in sys.modules:
        utils_module = types.ModuleType("utils")
        preprocessor_module = types.ModuleType("utils.preprocessor")
        preprocessor_module.preprocess_input = lambda data: data
        preprocessor_module.load_scaler = lambda _path: MagicMock()
        sys.modules["utils"] = utils_module
        sys.modules["utils.preprocessor"] = preprocessor_module

    if "prometheus_client" not in sys.modules:
        prometheus_module = types.ModuleType("prometheus_client")

        class _DummyMetric:
            def labels(self, **_kwargs):
                return self

            def inc(self, *_args, **_kwargs):
                return None

            def observe(self, *_args, **_kwargs):
                return None

        prometheus_module.Counter = lambda *_args, **_kwargs: _DummyMetric()
        prometheus_module.Histogram = lambda *_args, **_kwargs: _DummyMetric()
        prometheus_module.generate_latest = lambda *_args, **_kwargs: b""
        prometheus_module.CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
        sys.modules["prometheus_client"] = prometheus_module


def _get_app_module():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
    _install_test_stubs()
    return importlib.import_module("src.api.main")

def test_health_endpoint():
    main = _get_app_module()
    with patch.object(main, "MODEL", MagicMock()):
        client = TestClient(main.app)
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"

def test_predict_returns_probability():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = [[0.8, 0.2]]
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = [[0.0] * 19]

    main = _get_app_module()
    with patch.object(main, "MODEL", mock_model), \
         patch.object(main, "SCALER", mock_scaler):
        client = TestClient(main.app)
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_predict_returns_503_when_model_missing():
    main = _get_app_module()
    with patch.object(main, "MODEL", None), \
         patch.object(main, "SCALER", MagicMock()):
        client = TestClient(main.app)
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 503