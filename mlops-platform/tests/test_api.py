from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

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

def test_health_endpoint():
    with patch("src.api.main.MODEL", MagicMock()):
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

def test_predict_returns_probability():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.zeros((1, 19))

    with patch("src.api.main.MODEL", mock_model), \
         patch("src.api.main.SCALER", mock_scaler):
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/predict", json=sample_customer)
        assert response.status_code == 200
        data = response.json()
        assert "churn_probability" in data
        assert 0.0 <= data["churn_probability"] <= 1.0
        assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]