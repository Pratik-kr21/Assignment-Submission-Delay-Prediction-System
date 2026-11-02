import os
import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from train import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN, load_data, train
from main import app, evaluate_risk


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def model():
    if not os.path.exists("model.joblib"):
        m, _ = train()
        return m
    return joblib.load("model.joblib")


def test_data_integrity():
    df = load_data("data.csv")
    assert not df.empty, "Dataset must not be empty"
    assert len(df) >= 100, f"Expected at least 100 records, got {len(df)}"

    required_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    assert df[TARGET_COLUMN].isin([0, 1]).all(), "Target column must be binary (0 or 1)"
    assert df[NUMERIC_FEATURES].isnull().sum().sum() == 0, "Numeric features must not contain null values"


def test_train_pipeline():
    pipeline, metrics = train(data_path="data.csv", model_output_path="model.joblib")
    assert os.path.exists("model.joblib"), "model.joblib should be created"
    assert metrics["accuracy"] >= 0.70, f"Accuracy too low: {metrics['accuracy']:.4f}"
    assert metrics["roc_auc"] >= 0.70, f"ROC-AUC too low: {metrics['roc_auc']:.4f}"


def test_model_inference(model):
    sample_data = pd.DataFrame([{
        "hist_avg_delay_days": 2.5,
        "hist_late_rate": 0.6,
        "hist_max_delay_days": 4.0,
        "clicks_last_7_days": 5,
        "clicks_last_14_days": 20,
        "activity_decay_ratio": 0.15,
        "days_to_first_vle_access": 7.0,
        "concurrent_deadlines_window": 3,
        "studied_credits": 90,
        "assessment_weight": 40.0,
        "assessment_type": "TMA",
    }])

    prob = float(model.predict_proba(sample_data)[0, 1])
    assert 0.0 <= prob <= 1.0, f"Probability out of bounds: {prob}"

    pred = int(model.predict(sample_data)[0])
    assert pred in [0, 1], f"Prediction out of bounds: {pred}"

    tier, action = evaluate_risk(prob)
    assert tier in ["High", "Moderate", "Low"]
    assert len(action) > 0


def test_api_root_and_health(client):
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert res_root.json()["status"] == "online"

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"
    assert res_health.json()["model_loaded"] is True


def test_api_predict_endpoint(client):
    payload = {
        "student_id": "STU_TEST_999",
        "hist_avg_delay_days": 3.0,
        "hist_late_rate": 0.75,
        "hist_max_delay_days": 5.0,
        "clicks_last_7_days": 4,
        "clicks_last_14_days": 30,
        "activity_decay_ratio": 0.12,
        "days_to_first_vle_access": 8.0,
        "concurrent_deadlines_window": 2,
        "studied_credits": 60,
        "assessment_weight": 30.0,
        "assessment_type": "TMA",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["student_id"] == "STU_TEST_999"
    assert data["is_late_prediction"] in [0, 1]
    assert 0.0 <= data["delay_probability"] <= 1.0
    assert data["risk_tier"] in ["High", "Moderate", "Low"]
    assert "recommended_action" in data


def test_api_batch_predict_endpoint(client):
    payload = {
        "students": [
            {
                "student_id": "STU_BATCH_1",
                "hist_avg_delay_days": -1.0,
                "hist_late_rate": 0.05,
                "hist_max_delay_days": 0.0,
                "clicks_last_7_days": 80,
                "clicks_last_14_days": 180,
                "activity_decay_ratio": 0.85,
                "days_to_first_vle_access": 1.0,
                "concurrent_deadlines_window": 0,
                "studied_credits": 60,
                "assessment_weight": 20.0,
                "assessment_type": "TMA",
            },
            {
                "student_id": "STU_BATCH_2",
                "hist_avg_delay_days": 4.5,
                "hist_late_rate": 0.85,
                "hist_max_delay_days": 7.0,
                "clicks_last_7_days": 2,
                "clicks_last_14_days": 25,
                "activity_decay_ratio": 0.08,
                "days_to_first_vle_access": 12.0,
                "concurrent_deadlines_window": 4,
                "studied_credits": 120,
                "assessment_weight": 50.0,
                "assessment_type": "Exam",
            },
        ]
    }
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["total_evaluated"] == 2
    assert len(data["predictions"]) == 2
    assert "high_risk_count" in data


if __name__ == "__main__":
    print("Running test suite manually...")
    test_data_integrity()
    print("[OK] test_data_integrity passed")

    test_train_pipeline()
    print("[OK] test_train_pipeline passed")

    loaded_model = joblib.load("model.joblib")
    test_model_inference(loaded_model)
    print("[OK] test_model_inference passed")

    with TestClient(app) as tc:
        test_api_root_and_health(tc)
        print("[OK] test_api_root_and_health passed")

        test_api_predict_endpoint(tc)
        print("[OK] test_api_predict_endpoint passed")

        test_api_batch_predict_endpoint(tc)
        print("[OK] test_api_batch_predict_endpoint passed")

    print("\nALL TESTS PASSED SUCCESSFULLY!")
