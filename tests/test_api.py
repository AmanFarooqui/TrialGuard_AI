from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_get_root_returns_running_status():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "project": "TrialGuard AI",
        "status": "running",
    }


def test_get_summary_returns_expected_metrics():
    response = client.get("/summary")

    assert response.status_code == 200
    assert {
        "total_sites",
        "total_site_period_records",
        "critical_records",
        "high_risk_records",
        "anomaly_records",
        "deteriorating_sites",
    }.issubset(response.json())


def test_get_sites_returns_at_least_one_site():
    response = client.get("/sites")

    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert response.json()


def test_get_existing_site_returns_trend_and_records():
    response = client.get("/sites/SITE0057")

    assert response.status_code == 200
    assert {"trend", "records"}.issubset(response.json())


def test_get_invalid_site_returns_not_found():
    response = client.get("/sites/INVALID_SITE")

    assert response.status_code == 404


def test_get_risk_returns_only_high_risk_records():
    response = client.get("/risk")

    assert response.status_code == 200
    records = response.json()
    assert isinstance(records, list)
    assert all(record["risk_status"] == "High Risk" for record in records)


def test_get_anomalies_returns_only_anomaly_records():
    response = client.get("/anomalies")

    assert response.status_code == 200
    records = response.json()
    assert isinstance(records, list)
    assert all(record["anomaly_status"] == "Anomaly" for record in records)


def test_get_critical_returns_only_critical_records():
    response = client.get("/critical")

    assert response.status_code == 200
    records = response.json()
    assert isinstance(records, list)
    assert all(record["priority"] == "Critical" for record in records)


def test_get_site_explanation_returns_expected_fields():
    response = client.get("/sites/SITE0057/explanation")

    assert response.status_code == 200
    assert {
        "site_id",
        "monitoring_period",
        "risk_probability",
        "risk_status",
        "risk_increasing_factors",
        "risk_reducing_factors",
    }.issubset(response.json())


def test_get_invalid_site_explanation_returns_not_found():
    response = client.get("/sites/INVALID_SITE/explanation")

    assert response.status_code == 404