from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILES = {
    "patients": "patients.csv",
    "visits": "visits.csv",
}
OUTPUT_FILE = "site_patient_visit_features.csv"
MONITORING_START = pd.Timestamp("2025-01-01")
MONITORING_END = pd.Timestamp("2026-08-31")


def load_input_data(data_dir):
    """Load patient and visit data from synthetic CSV files."""
    patients = pd.read_csv(data_dir / INPUT_FILES["patients"])
    visits = pd.read_csv(data_dir / INPUT_FILES["visits"])
    return patients, visits


def derive_monitoring_period(visits):
    """Assign each visit to one of the project's 12 monitoring periods."""
    visit_data = visits.copy()
    visit_data["scheduled_date"] = pd.to_datetime(
        visit_data["scheduled_date"], errors="coerce"
    )
    visit_data["actual_date"] = pd.to_datetime(
        visit_data["actual_date"], errors="coerce"
    )
    visit_data["visit_date"] = visit_data["actual_date"].fillna(
        visit_data["scheduled_date"]
    )

    period_edges = pd.date_range(
        start=MONITORING_START,
        end=MONITORING_END + pd.Timedelta(days=1),
        periods=13,
    )
    period_index = np.searchsorted(
        period_edges.values,
        visit_data["visit_date"].values.astype("datetime64[ns]"),
        side="right",
    )
    visit_data["monitoring_period"] = period_index.clip(1, 12)
    return visit_data


def connect_visits_to_sites(patients, visits):
    """Add site IDs to visits through the patient-to-site relationship."""
    patient_sites = patients[["patient_id", "site_id", "status"]]
    return visits.merge(patient_sites, on="patient_id", how="inner")


def build_patient_visit_features(patients, visits):
    """Aggregate patient and visit metrics by site and monitoring period."""
    visits_with_period = derive_monitoring_period(visits)
    visits_with_sites = connect_visits_to_sites(patients, visits_with_period)

    feature_table = (
        visits_with_sites.groupby(["site_id", "monitoring_period"])
        .agg(
            total_patients=("patient_id", "nunique"),
            dropped_out_patients=(
                "status",
                lambda statuses: (statuses == "Dropped Out").sum(),
            ),
            total_visits=("visit_id", "count"),
            completed_visits=("completed", "sum"),
            average_visit_delay_days=("visit_delay_days", "mean"),
        )
        .reset_index()
    )

    feature_table["patient_dropout_rate"] = (
        feature_table["dropped_out_patients"]
        / feature_table["total_patients"]
    )
    feature_table["visit_completion_rate"] = (
        feature_table["completed_visits"] / feature_table["total_visits"]
    )

    return feature_table[
        [
            "site_id",
            "monitoring_period",
            "total_patients",
            "dropped_out_patients",
            "patient_dropout_rate",
            "total_visits",
            "completed_visits",
            "visit_completion_rate",
            "average_visit_delay_days",
        ]
    ]


def save_features(feature_table, output_dir):
    """Create the processed-data directory and save the feature table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILE
    feature_table.to_csv(output_path, index=False)
    return output_path


def main():
    """Build and save site-level patient and visit features."""
    project_root = Path(__file__).resolve().parents[2]
    synthetic_data_dir = project_root / "data" / "synthetic"
    processed_data_dir = project_root / "data" / "processed"

    patients, visits = load_input_data(synthetic_data_dir)
    feature_table = build_patient_visit_features(patients, visits)
    output_path = save_features(feature_table, processed_data_dir)

    print(f"Saved {len(feature_table)} feature rows to {output_path}")


if __name__ == "__main__":
    main()
