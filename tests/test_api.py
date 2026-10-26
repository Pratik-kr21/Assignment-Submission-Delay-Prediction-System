"""Unit tests for the FastAPI service endpoints."""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "Assignment Submission Delay Prediction API"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_endpoint():
    sample_payload = {
        "student_id": "STU_9999",
        "hist_avg_delay_days": 1.2,
        "hist_late_rate": 0.4,
        "hist_max_delay_days": 3.0,
        "clicks_last_7_days": 15,
        "clicks_last_14_days": 40,
        "activity_decay_ratio": 0.6,
        "days_to_first_vle_access": 3.0,
        "concurrent_deadlines_window": 1,
        "studied_credits": 60,
        "assessment_weight": 20.0,
        "assessment_type": "TMA"
    }
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_late_prediction" in data
    assert "delay_probability" in data
    assert "risk_tier" in data
    assert data["risk_tier"] in ["Low", "Moderate", "High"]
