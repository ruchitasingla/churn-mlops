import pytest
from fastapi.testclient import TestClient

from churn_mlops import api, config, train


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("models")
    config.MODEL_DIR, config.MODEL_PATH, config.METRICS_PATH = (
        d,
        d / "model.joblib",
        d / "metrics.json",
    )
    train.mlflow = None
    assert train.main(["--n-samples", "1500", "--n-estimators", "50"]) == 0
    with TestClient(api.app) as c:
        yield c


VALID = {
    "tenure_months": 3,
    "monthly_charges": 95.0,
    "total_charges": 285.0,
    "support_tickets": 4,
    "avg_monthly_usage_gb": 80.0,
    "contract_type": "month-to-month",
    "payment_method": "e-wallet",
    "has_internet_backup": "no",
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_returns_probability(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["will_churn"], bool)


def test_risky_customer_scores_higher_than_loyal_one(client):
    loyal = {
        **VALID,
        "tenure_months": 60,
        "support_tickets": 0,
        "contract_type": "two-year",
        "monthly_charges": 40.0,
        "total_charges": 2400.0,
        "has_internet_backup": "yes",
        "payment_method": "credit-card",
    }
    risky = client.post("/predict", json=VALID).json()["churn_probability"]
    safe = client.post("/predict", json=loyal).json()["churn_probability"]
    assert risky > safe


def test_invalid_input_rejected(client):
    r = client.post("/predict", json={**VALID, "contract_type": "lifetime"})
    assert r.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.post("/predict", json=VALID)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "predictions_total" in r.text and "churn_probability_bucket" in r.text


def test_homepage_serves_ui(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Churn risk" in r.text
