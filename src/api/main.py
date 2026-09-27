import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "tuned_random_forest.joblib"
ML_DATASET_PATH = PROCESSED_DATA_DIR / "ml_dataset.csv"
TARGET_COLUMN = "high_risk_next_period"
TIME_COLUMN = "monitoring_period"
EXCLUDED_MODEL_COLUMNS = {
    "site_id",
    TIME_COLUMN,
    TARGET_COLUMN,
    "behavior",
}
RISK_THRESHOLD = 0.40


class HealthResponse(BaseModel):
    project: str
    status: str


class SummaryResponse(BaseModel):
    total_sites: int
    total_site_period_records: int
    critical_records: int
    high_risk_records: int
    anomaly_records: int
    deteriorating_sites: int


class SiteDetailResponse(BaseModel):
    trend: dict[str, Any]
    records: list[dict[str, Any]]


class ExplanationFactor(BaseModel):
    feature: str
    feature_value: float
    shap_value: float


class SiteExplanationResponse(BaseModel):
    site_id: str
    monitoring_period: int | float
    risk_probability: float
    risk_status: str
    risk_increasing_factors: list[ExplanationFactor]
    risk_reducing_factors: list[ExplanationFactor]


@lru_cache(maxsize=1)
def load_data() -> dict[str, pd.DataFrame]:
    """Load the processed datasets used by the API."""
    filenames = {
        "site_intelligence": "site_intelligence.csv",
        "site_trends": "site_trends.csv",
        "anomaly_results": "anomaly_results.csv",
    }
    try:
        return {
            dataset_name: pd.read_csv(PROCESSED_DATA_DIR / filename)
            for dataset_name, filename in filenames.items()
        }
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Required processed data file was not found: {error.filename}"
        ) from error


def get_data() -> dict[str, pd.DataFrame]:
    """Return loaded data and expose data-loading failures as API errors."""
    try:
        return load_data()
    except (OSError, RuntimeError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


def records_from_frame(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert pandas records to JSON-safe native values."""
    return json.loads(frame.to_json(orient="records"))


@lru_cache(maxsize=1)
def load_ml_dataset() -> pd.DataFrame:
    """Load the dataset used to train the tuned Random Forest."""
    try:
        return pd.read_csv(ML_DATASET_PATH)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Required ML dataset was not found: {error.filename}"
        ) from error


@lru_cache(maxsize=1)
def load_tuned_model():
    """Load the persisted tuned Random Forest without retraining it."""
    if not MODEL_PATH.is_file():
        raise RuntimeError(f"Tuned model was not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def get_model_features(dataset: pd.DataFrame) -> list[str]:
    """Return the same numeric feature columns used during training."""
    feature_columns = [
        column_name
        for column_name in dataset.select_dtypes(include="number").columns
        if column_name not in EXCLUDED_MODEL_COLUMNS
    ]
    if len(feature_columns) != 18:
        raise RuntimeError(
            "Expected 18 model features, found "
            f"{len(feature_columns)}: {feature_columns}"
        )
    return feature_columns


def get_positive_class_shap_values(explainer, features: pd.DataFrame):
    """Normalize SHAP output to one row of positive-class contributions."""
    raw_values = explainer.shap_values(features)
    if isinstance(raw_values, list):
        if len(raw_values) != 2:
            raise RuntimeError("Expected binary-class SHAP values.")
        return raw_values[1][0]
    if raw_values.ndim == 3:
        if raw_values.shape[2] != 2:
            raise RuntimeError("Expected binary-class SHAP values.")
        return raw_values[0, :, 1]
    return raw_values[0]


def get_site_explanation(site_id: str) -> SiteExplanationResponse:
    """Calculate the latest-record SHAP explanation for one site."""
    try:
        dataset = load_ml_dataset()
        site_records = dataset[dataset["site_id"].eq(site_id)]
    except (OSError, RuntimeError, KeyError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    if site_records.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Site '{site_id}' was not found.",
        )

    latest_record = site_records.sort_values(TIME_COLUMN).iloc[-1]
    try:
        feature_columns = get_model_features(dataset)
        model = load_tuned_model()
        import shap

        explainer = shap.TreeExplainer(model)
        site_features = latest_record[feature_columns].to_frame().T
        shap_values = get_positive_class_shap_values(explainer, site_features)
        risk_probability = float(model.predict_proba(site_features)[0, 1])
    except ImportError as error:
        raise HTTPException(
            status_code=500,
            detail="SHAP is not installed. Install it with `pip install shap`.",
        ) from error
    except (OSError, RuntimeError, ValueError, AttributeError) as error:
        raise HTTPException(status_code=500, detail=str(error)) from error

    contributions = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_value": site_features.iloc[0].astype(float).to_numpy(),
            "shap_value": shap_values.astype(float),
        }
    )
    increasing = contributions[contributions["shap_value"] > 0]
    reducing = contributions[contributions["shap_value"] < 0]
    increasing = increasing.assign(
        absolute_shap_value=increasing["shap_value"].abs()
    )
    reducing = reducing.assign(
        absolute_shap_value=reducing["shap_value"].abs()
    )

    return SiteExplanationResponse(
        site_id=site_id,
        monitoring_period=latest_record[TIME_COLUMN],
        risk_probability=risk_probability,
        risk_status=(
            "High Risk" if risk_probability >= RISK_THRESHOLD else "Low Risk"
        ),
        risk_increasing_factors=records_from_frame(
            increasing.sort_values(
                "absolute_shap_value", ascending=False
            ).head(5).drop(columns="absolute_shap_value")
        ),
        risk_reducing_factors=records_from_frame(
            reducing.sort_values(
                "absolute_shap_value", ascending=False
            ).head(5).drop(columns="absolute_shap_value")
        ),
    )


app = FastAPI(title="TrialGuard AI API")


@app.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return the API health status."""
    return HealthResponse(project="TrialGuard AI", status="running")


@app.get("/summary", response_model=SummaryResponse)
def get_summary() -> SummaryResponse:
    """Return aggregate site risk, anomaly, and trend counts."""
    data = get_data()
    site_intelligence = data["site_intelligence"]
    site_trends = data["site_trends"]

    return SummaryResponse(
        total_sites=int(site_trends["site_id"].nunique()),
        total_site_period_records=len(site_intelligence),
        critical_records=int(site_intelligence["priority"].eq("Critical").sum()),
        high_risk_records=int(
            site_intelligence["risk_status"].eq("High Risk").sum()
        ),
        anomaly_records=int(
            site_intelligence["anomaly_status"].eq("Anomaly").sum()
        ),
        deteriorating_sites=int(
            site_trends["trend_status"].eq("Deteriorating").sum()
        ),
    )


@app.get("/sites", response_model=list[dict[str, Any]])
def get_sites() -> list[dict[str, Any]]:
    """Return site-level trend information for every site."""
    return records_from_frame(get_data()["site_trends"])


@app.get("/sites/{site_id}", response_model=SiteDetailResponse)
def get_site(site_id: str) -> SiteDetailResponse:
    """Return one site's trend information and monitoring-period history."""
    data = get_data()
    site_trends = data["site_trends"]
    site_intelligence = data["site_intelligence"]

    trend_records = site_trends[site_trends["site_id"].eq(site_id)]
    if trend_records.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Site '{site_id}' was not found.",
        )

    history_records = site_intelligence[
        site_intelligence["site_id"].eq(site_id)
    ].sort_values("monitoring_period")
    return SiteDetailResponse(
        trend=records_from_frame(trend_records)[0],
        records=records_from_frame(history_records),
    )


@app.get(
    "/sites/{site_id}/explanation",
    response_model=SiteExplanationResponse,
)
def get_site_explanation_endpoint(site_id: str) -> SiteExplanationResponse:
    """Return positive and negative SHAP factors for a site's latest record."""
    return get_site_explanation(site_id)


@app.get("/risk", response_model=list[dict[str, Any]])
def get_high_risk_records() -> list[dict[str, Any]]:
    """Return site-period records currently marked High Risk."""
    site_intelligence = get_data()["site_intelligence"]
    return records_from_frame(
        site_intelligence[site_intelligence["risk_status"].eq("High Risk")]
    )


@app.get("/anomalies", response_model=list[dict[str, Any]])
def get_anomaly_records() -> list[dict[str, Any]]:
    """Return anomaly records from the anomaly results dataset."""
    anomaly_results = get_data()["anomaly_results"]
    return records_from_frame(
        anomaly_results[anomaly_results["anomaly_status"].eq("Anomaly")]
    )


@app.get("/critical", response_model=list[dict[str, Any]])
def get_critical_records() -> list[dict[str, Any]]:
    """Return site-period records marked Critical."""
    site_intelligence = get_data()["site_intelligence"]
    return records_from_frame(
        site_intelligence[site_intelligence["priority"].eq("Critical")]
    )
