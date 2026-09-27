from pathlib import Path

import joblib
import pandas as pd


ML_DATASET_FILE = "ml_dataset.csv"
ANOMALY_RESULTS_FILE = "anomaly_results.csv"
MODEL_FILE = "tuned_random_forest.joblib"
OUTPUT_FILE = "site_intelligence.csv"
TARGET_COLUMN = "high_risk_next_period"
TIME_COLUMN = "monitoring_period"
TRAINING_EXCLUDED_COLUMNS = {
    "site_id",
    TIME_COLUMN,
    TARGET_COLUMN,
    "behavior",
}
CURRENT_PERIODS = range(8, 12)
RISK_THRESHOLD = 0.40
OPERATIONAL_FEATURES = [
    "missing_data_count",
    "protocol_deviation_count",
    "data_entry_delay_avg",
    "query_resolution_avg",
    "patient_dropout_rate",
    "average_visit_delay_days",
    "total_adverse_events",
    "serious_event_rate",
    "severe_event_rate",
]
OUTPUT_COLUMNS = [
    "site_id",
    TIME_COLUMN,
    "risk_probability",
    "risk_status",
    "anomaly_score",
    "anomaly_status",
    *OPERATIONAL_FEATURES,
]
PRIORITY_ORDER = {
    "Critical": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3,
}


def load_inputs(project_root):
    """Load the ML dataset and anomaly results."""
    processed_data_dir = project_root / "data" / "processed"
    ml_dataset = pd.read_csv(processed_data_dir / ML_DATASET_FILE)
    anomaly_results = pd.read_csv(processed_data_dir / ANOMALY_RESULTS_FILE)
    return ml_dataset, anomaly_results


def validate_columns(dataset, required_columns, dataset_name):
    """Ensure an input dataset contains the required columns."""
    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}"
        )


def select_current_period_data(ml_dataset):
    """Sort the ML dataset and select the current evaluation periods."""
    validate_columns(
        ml_dataset,
        ["site_id", TIME_COLUMN, TARGET_COLUMN, *OPERATIONAL_FEATURES],
        "ML dataset",
    )
    sorted_dataset = ml_dataset.sort_values(
        [TIME_COLUMN, "site_id"]
    ).reset_index(drop=True)
    current_data = sorted_dataset[
        sorted_dataset[TIME_COLUMN].isin(CURRENT_PERIODS)
    ].copy()
    if current_data.empty:
        raise ValueError("Monitoring periods 8-11 contain no records.")
    return current_data


def prepare_model_features(current_data):
    """Select the numeric feature columns expected by the trained model."""
    feature_columns = [
        column_name
        for column_name in current_data.select_dtypes(include="number").columns
        if column_name not in TRAINING_EXCLUDED_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No numeric model features are available for scoring.")
    return current_data[feature_columns]


def load_model(project_root):
    """Load the fitted tuned Random Forest without retraining it."""
    model_path = project_root / "models" / MODEL_FILE
    if not model_path.is_file():
        raise FileNotFoundError(f"Tuned model was not found at {model_path}")
    return joblib.load(model_path)


def add_risk_predictions(current_data, model):
    """Generate risk probabilities and fixed-threshold statuses."""
    features = prepare_model_features(current_data)
    risk_probability = model.predict_proba(features)[:, 1]
    results = current_data.copy()
    results["risk_probability"] = risk_probability
    results["risk_status"] = results["risk_probability"].ge(
        RISK_THRESHOLD
    ).map({True: "High Risk", False: "Low Risk"})
    return results


def merge_anomaly_results(risk_results, anomaly_results):
    """Merge anomaly scores and statuses by site and monitoring period."""
    join_keys = ["site_id", TIME_COLUMN]
    validate_columns(
        anomaly_results,
        [*join_keys, "anomaly_score", "anomaly_status"],
        "Anomaly results",
    )
    anomaly_data = anomaly_results[
        [*join_keys, "anomaly_score", "anomaly_status"]
    ]
    if anomaly_data.duplicated(join_keys).any():
        raise ValueError("Anomaly results contain duplicate site-period keys.")

    merged = risk_results.merge(
        anomaly_data,
        on=join_keys,
        how="left",
        validate="one_to_one",
    )
    if merged["anomaly_status"].isna().any():
        raise ValueError("Some current-period records have no anomaly result.")
    return merged


def add_priority(results):
    """Assign investigation priority from risk and anomaly status."""
    priority_conditions = {
        ("High Risk", "Anomaly"): "Critical",
        ("High Risk", "Normal"): "High",
        ("Low Risk", "Anomaly"): "Medium",
        ("Low Risk", "Normal"): "Low",
    }
    results = results.copy()
    results["priority"] = [
        priority_conditions[(risk_status, anomaly_status)]
        for risk_status, anomaly_status in zip(
            results["risk_status"],
            results["anomaly_status"],
        )
    ]
    return results


def build_site_intelligence(ml_dataset, anomaly_results, model):
    """Build the prioritized current-period site intelligence table."""
    current_data = select_current_period_data(ml_dataset)
    risk_results = add_risk_predictions(current_data, model)
    merged_results = merge_anomaly_results(risk_results, anomaly_results)
    prioritized_results = add_priority(merged_results)
    prioritized_results["priority_order"] = prioritized_results["priority"].map(
        PRIORITY_ORDER
    )
    return prioritized_results.sort_values(
        ["priority_order", "risk_probability"],
        ascending=[True, False],
    ).drop(columns="priority_order")[
        [*OUTPUT_COLUMNS, "priority"]
    ].reset_index(drop=True)


def print_summary(results):
    """Print counts and the highest-priority records."""
    print(f"Total records: {len(results)}")
    for priority in PRIORITY_ORDER:
        print(
            f"Number of {priority}: "
            f"{results['priority'].eq(priority).sum()}"
        )
    print("Top 10 highest-priority records:")
    print(results.head(10).to_string(index=False))


def save_results(results, project_root):
    """Create the processed-data directory and save site intelligence."""
    processed_data_dir = project_root / "data" / "processed"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_data_dir / OUTPUT_FILE
    results.to_csv(output_path, index=False)
    return output_path


def main():
    """Score current periods and build the prioritized intelligence output."""
    project_root = Path(__file__).resolve().parents[2]
    ml_dataset, anomaly_results = load_inputs(project_root)
    model = load_model(project_root)
    results = build_site_intelligence(ml_dataset, anomaly_results, model)
    print_summary(results)
    output_path = save_results(results, project_root)
    print(f"Saved site intelligence to {output_path}")


if __name__ == "__main__":
    main()