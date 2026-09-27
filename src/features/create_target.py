from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = "master_feature_dataset.csv"
OUTPUT_FILE = "ml_dataset.csv"
MERGE_KEYS = ["site_id", "monitoring_period"]
RISK_PERCENTILE = 0.75

# Caps convert different units into comparable 0-100 component scores.
RISK_CAPS = {
    "missing_data_count": 20,
    "protocol_deviation_count": 10,
    "average_visit_delay_days": 14,
    "data_entry_delay_avg": 14,
    "query_resolution_avg": 20,
}


def load_master_features(processed_data_dir):
    """Load the master feature table."""
    return pd.read_csv(processed_data_dir / INPUT_FILE)


def add_enrollment_deterioration(master_features):
    """Calculate each site's current-period patient decline from the prior period."""
    feature_table = master_features.sort_values(MERGE_KEYS).copy()
    previous_patients = feature_table.groupby("site_id")["total_patients"].shift(1)

    patient_decline = (
        (previous_patients - feature_table["total_patients"])
        .div(previous_patients.replace(0, np.nan))
        .clip(lower=0)
        .fillna(0)
    )
    feature_table["enrollment_deterioration"] = patient_decline
    return feature_table


def normalize_feature(values, cap):
    """Scale a non-negative feature to a 0-100 range using a practical cap."""
    return values.clip(lower=0).div(cap).clip(upper=1).mul(100)


def calculate_risk_score(feature_table):
    """Calculate a weighted current-period synthetic risk score from 0 to 100."""
    components = {
        "missing_data_count": 0.15,
        "protocol_deviation_count": 0.15,
        "patient_dropout_rate": 0.15,
        "average_visit_delay_days": 0.10,
        "data_entry_delay_avg": 0.15,
        "query_resolution_avg": 0.10,
        "enrollment_deterioration": 0.20,
    }

    normalized_components = []
    for feature_name, weight in components.items():
        if feature_name == "patient_dropout_rate":
            normalized = feature_table[feature_name].clip(0, 1).mul(100)
        elif feature_name == "enrollment_deterioration":
            normalized = feature_table[feature_name].clip(0, 1).mul(100)
        else:
            normalized = normalize_feature(
                feature_table[feature_name],
                RISK_CAPS[feature_name],
            )
        normalized_components.append(normalized * weight)

    feature_table["risk_score"] = sum(normalized_components).clip(0, 100)
    return feature_table


def create_next_period_target(feature_table):
    """Label the next period's highest-risk records and shift labels back."""
    next_period_risk_score = feature_table.groupby("site_id")["risk_score"].shift(-1)
    risk_threshold = next_period_risk_score.dropna().quantile(RISK_PERCENTILE)
    feature_table["high_risk_next_period"] = (
        next_period_risk_score >= risk_threshold
    ).where(next_period_risk_score.notna())
    feature_table = feature_table.dropna(subset=["high_risk_next_period"]).copy()
    feature_table["high_risk_next_period"] = feature_table[
        "high_risk_next_period"
    ].astype(int)

    if feature_table["high_risk_next_period"].nunique() != 2:
        raise ValueError("The generated target must contain both 0 and 1 classes.")

    return feature_table


def prepare_ml_dataset(feature_table):
    """Remove labels, reference metadata, and temporary risk calculations."""
    columns_to_remove = [
        "behavior",
        "risk_score",
    ]
    return feature_table.drop(columns=columns_to_remove, errors="ignore")


def save_ml_dataset(ml_dataset, processed_data_dir):
    """Create the processed-data directory and save the ML dataset."""
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_data_dir / OUTPUT_FILE
    ml_dataset.to_csv(output_path, index=False)
    return output_path


def main():
    """Create, inspect, and save the next-period target dataset."""
    project_root = Path(__file__).resolve().parents[2]
    processed_data_dir = project_root / "data" / "processed"

    master_features = load_master_features(processed_data_dir)
    feature_table = add_enrollment_deterioration(master_features)
    feature_table = calculate_risk_score(feature_table)
    feature_table = create_next_period_target(feature_table)
    ml_dataset = prepare_ml_dataset(feature_table)
    output_path = save_ml_dataset(ml_dataset, processed_data_dir)

    print(f"Final dataset shape: {ml_dataset.shape}")
    print("Target value counts:")
    print(ml_dataset["high_risk_next_period"].value_counts().sort_index())
    high_risk_percentage = ml_dataset["high_risk_next_period"].mean() * 100
    print(f"Percentage of high-risk records: {high_risk_percentage:.2f}%")
    print("First 5 rows:")
    print(ml_dataset.head(5))
    print(f"Saved ML dataset to {output_path}")


if __name__ == "__main__":
    main()
