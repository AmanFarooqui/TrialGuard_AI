from pathlib import Path
from functools import reduce

import pandas as pd


INPUT_FILES = [
    "site_operational_features.csv",
    "site_patient_visit_features.csv",
    "site_adverse_event_features.csv",
]
OUTPUT_FILE = "master_feature_dataset.csv"
MERGE_KEYS = ["site_id", "monitoring_period"]


def load_feature_tables(processed_data_dir):
    """Load all processed feature tables from the processed-data directory."""
    return [
        pd.read_csv(processed_data_dir / filename)
        for filename in INPUT_FILES
    ]


def merge_feature_tables(feature_tables):
    """Outer-merge feature tables on site and monitoring period."""
    return reduce(
        lambda left, right: left.merge(
            right,
            on=MERGE_KEYS,
            how="outer",
        ),
        feature_tables,
    )


def fill_missing_numeric_features(master_features):
    """Fill missing values in numeric feature columns with zero."""
    feature_columns = [
        column_name
        for column_name in master_features.select_dtypes(
            include="number"
        ).columns
        if column_name not in MERGE_KEYS
    ]
    master_features[feature_columns] = master_features[feature_columns].fillna(0)
    return master_features


def sort_master_features(master_features):
    """Sort the master table by site and monitoring period."""
    return master_features.sort_values(MERGE_KEYS).reset_index(drop=True)


def validate_master_features(master_features):
    """Print basic validation results for the master feature table."""
    duplicate_count = master_features.duplicated(subset=MERGE_KEYS).sum()
    missing_site_count = master_features["site_id"].isna().sum()
    missing_period_count = master_features["monitoring_period"].isna().sum()

    print("Master feature validation:")
    print(f"  duplicate site-period combinations: {duplicate_count}")
    print(f"  missing site_id values: {missing_site_count}")
    print(f"  missing monitoring_period values: {missing_period_count}")
    print(f"  final dataset shape: {master_features.shape}")
    print(f"  columns: {list(master_features.columns)}")
    print("  first 5 rows:")
    print(master_features.head(5))


def save_master_features(master_features, processed_data_dir):
    """Create the processed-data directory and save the master table."""
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_data_dir / OUTPUT_FILE
    master_features.to_csv(output_path, index=False)
    return output_path


def main():
    """Build, validate, and save the master feature dataset."""
    project_root = Path(__file__).resolve().parents[2]
    processed_data_dir = project_root / "data" / "processed"

    feature_tables = load_feature_tables(processed_data_dir)
    master_features = merge_feature_tables(feature_tables)
    master_features = fill_missing_numeric_features(master_features)
    master_features = sort_master_features(master_features)
    validate_master_features(master_features)
    output_path = save_master_features(master_features, processed_data_dir)

    print(f"Saved master features to {output_path}")


if __name__ == "__main__":
    main()
