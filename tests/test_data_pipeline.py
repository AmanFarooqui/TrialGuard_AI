from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
ML_DATASET_PATH = PROCESSED_DATA_DIR / "ml_dataset.csv"
TARGET_COLUMN = "high_risk_next_period"
EXCLUDED_FEATURE_COLUMNS = {
    "site_id",
    "monitoring_period",
    TARGET_COLUMN,
    "behavior",
}
REQUIRED_ML_COLUMNS = {
    "site_id",
    "monitoring_period",
    "missing_data_count",
    "protocol_deviation_count",
    "data_entry_delay_avg",
    "query_resolution_avg",
    "patient_dropout_rate",
    "average_visit_delay_days",
    TARGET_COLUMN,
}
MODEL_FILENAMES = (
    "logistic_regression_baseline.joblib",
    "random_forest_baseline.joblib",
    "gradient_boosting_baseline.joblib",
    "tuned_random_forest.joblib",
)


def load_ml_dataset() -> pd.DataFrame:
    return pd.read_csv(ML_DATASET_PATH)


def get_ml_features(dataset: pd.DataFrame) -> list[str]:
    return [
        column_name
        for column_name in dataset.select_dtypes(include="number").columns
        if column_name not in EXCLUDED_FEATURE_COLUMNS
    ]


def test_ml_dataset_exists():
    assert ML_DATASET_PATH.is_file()


def test_ml_dataset_structure():
    dataset = load_ml_dataset()

    assert REQUIRED_ML_COLUMNS.issubset(dataset.columns)


def test_ml_dataset_has_no_missing_values():
    dataset = load_ml_dataset()
    feature_columns = get_ml_features(dataset)

    assert not dataset[feature_columns + [TARGET_COLUMN]].isna().any().any()


def test_target_contains_both_classes():
    dataset = load_ml_dataset()

    assert set(dataset[TARGET_COLUMN].unique()) == {0, 1}


def test_model_files_exist():
    for model_filename in MODEL_FILENAMES:
        assert (MODELS_DIR / model_filename).is_file()


def test_tuned_random_forest_can_predict():
    dataset = load_ml_dataset()
    model = joblib.load(MODELS_DIR / "tuned_random_forest.joblib")
    probabilities = model.predict_proba(dataset[get_ml_features(dataset)])[:, 1]

    assert len(probabilities) == len(dataset)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert not np.isnan(probabilities).any()


def test_anomaly_results_structure():
    anomaly_results = pd.read_csv(PROCESSED_DATA_DIR / "anomaly_results.csv")
    required_columns = {
        "site_id",
        "monitoring_period",
        "anomaly_prediction",
        "anomaly_score",
        "anomaly_status",
    }

    assert required_columns.issubset(anomaly_results.columns)
    assert {"Normal", "Anomaly"}.issubset(
        set(anomaly_results["anomaly_status"].unique())
    )


def test_site_intelligence_structure():
    site_intelligence = pd.read_csv(PROCESSED_DATA_DIR / "site_intelligence.csv")
    required_columns = {
        "site_id",
        "monitoring_period",
        "risk_probability",
        "risk_status",
        "anomaly_status",
        "priority",
    }

    assert required_columns.issubset(site_intelligence.columns)
    assert {"Critical", "High", "Medium", "Low"}.issubset(
        set(site_intelligence["priority"].unique())
    )


def test_site_trends_structure():
    site_trends = pd.read_csv(PROCESSED_DATA_DIR / "site_trends.csv")
    required_columns = {
        "site_id",
        "first_risk_probability",
        "latest_risk_probability",
        "risk_probability_change",
        "trend_slope",
        "trend_status",
        "priority_status",
    }

    assert required_columns.issubset(site_trends.columns)
    assert {"Deteriorating", "Stable", "Improving"}.issubset(
        set(site_trends["trend_status"].unique())
    )


def test_dataset_site_count():
    dataset = load_ml_dataset()

    assert dataset["site_id"].nunique() == 250
