from pathlib import Path

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "master_feature_dataset.csv"
OUTPUT_FILE = "anomaly_results.csv"
FEATURE_COLUMNS = [
    "missing_data_count",
    "protocol_deviation_count",
    "data_entry_delay_avg",
    "query_resolution_avg",
]
ANOMALY_ESTIMATORS = 200
CONTAMINATION = 0.05
RANDOM_STATE = 42


def load_dataset(project_root):
    """Load the master site-monitoring-period feature dataset."""
    return pd.read_csv(project_root / "data" / "processed" / INPUT_FILE)


def prepare_features(dataset):
    """Validate and standardize the operational features for anomaly detection."""
    missing_columns = [
        column_name
        for column_name in FEATURE_COLUMNS
        if column_name not in dataset.columns
    ]
    if missing_columns:
        raise ValueError(
            "Required operational feature columns are missing: "
            f"{missing_columns}"
        )

    features = dataset[FEATURE_COLUMNS].fillna(0)
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    return standardized_features


def build_model():
    """Build the configured Isolation Forest detector."""
    return IsolationForest(
        n_estimators=ANOMALY_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
    )


def detect_anomalies(dataset, standardized_features):
    """Add anomaly predictions, scores, and readable statuses to the dataset."""
    model = build_model()
    predictions = model.fit_predict(standardized_features)
    scores = model.decision_function(standardized_features)

    results = dataset.copy()
    results["anomaly_prediction"] = (predictions == -1).astype(int)
    results["anomaly_score"] = scores
    results["anomaly_status"] = results["anomaly_prediction"].map(
        {1: "Anomaly", 0: "Normal"}
    )
    return results


def print_summary(results):
    """Print counts, anomaly percentage, and the most unusual records."""
    anomaly_count = int(results["anomaly_prediction"].sum())
    total_records = len(results)
    anomaly_percentage = anomaly_count / total_records * 100

    print(f"Total records: {total_records}")
    print(f"Number of anomalies: {anomaly_count}")
    print(f"Anomaly percentage: {anomaly_percentage:.2f}%")
    print("Top 10 most unusual records:")
    print(
        results.sort_values("anomaly_score")
        .head(10)
        .to_string(index=False)
    )


def save_results(results, project_root):
    """Create the processed-data directory and save anomaly results."""
    processed_data_dir = project_root / "data" / "processed"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_data_dir / OUTPUT_FILE
    results.to_csv(output_path, index=False)
    return output_path


def main():
    """Detect unusual site-monitoring-period records and save the results."""
    project_root = Path(__file__).resolve().parents[2]
    dataset = load_dataset(project_root)
    standardized_features = prepare_features(dataset)
    results = detect_anomalies(dataset, standardized_features)

    print_summary(results)
    output_path = save_results(results, project_root)
    print(f"Saved anomaly results to {output_path}")


if __name__ == "__main__":
    main()