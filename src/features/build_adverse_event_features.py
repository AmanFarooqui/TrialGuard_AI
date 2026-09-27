from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = "adverse_events.csv"
OUTPUT_FILE = "site_adverse_event_features.csv"
MONITORING_START = pd.Timestamp("2025-01-01")
MONITORING_END = pd.Timestamp("2026-08-31")


def load_adverse_events(data_dir):
    """Load adverse events from the synthetic data directory."""
    return pd.read_csv(data_dir / INPUT_FILE)


def derive_monitoring_period(adverse_events):
    """Assign each adverse event to one of the project's 12 periods."""
    event_data = adverse_events.copy()
    event_data["event_date"] = pd.to_datetime(
        event_data["event_date"], errors="coerce"
    )

    period_edges = pd.date_range(
        start=MONITORING_START,
        end=MONITORING_END + pd.Timedelta(days=1),
        periods=13,
    )
    period_index = np.searchsorted(
        period_edges.values,
        event_data["event_date"].values.astype("datetime64[ns]"),
        side="right",
    )
    event_data["monitoring_period"] = period_index.clip(1, 12)
    return event_data


def build_adverse_event_features(adverse_events):
    """Aggregate adverse-event counts and delays by site and period."""
    event_data = derive_monitoring_period(adverse_events)
    event_data["serious_event"] = event_data["serious"].astype(int)
    event_data["severe_event"] = (
        event_data["severity"].eq("Severe").astype(int)
    )

    feature_table = (
        event_data.groupby(["site_id", "monitoring_period"])
        .agg(
            total_adverse_events=("event_id", "count"),
            serious_adverse_events=("serious_event", "sum"),
            severe_adverse_events=("severe_event", "sum"),
            average_reported_delay_days=("reported_delay_days", "mean"),
        )
        .reset_index()
    )

    feature_table["serious_event_rate"] = (
        feature_table["serious_adverse_events"]
        .div(feature_table["total_adverse_events"])
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )
    feature_table["severe_event_rate"] = (
        feature_table["severe_adverse_events"]
        .div(feature_table["total_adverse_events"])
        .replace([np.inf, -np.inf], 0)
        .fillna(0)
    )

    return feature_table[
        [
            "site_id",
            "monitoring_period",
            "total_adverse_events",
            "serious_adverse_events",
            "severe_adverse_events",
            "average_reported_delay_days",
            "serious_event_rate",
            "severe_event_rate",
        ]
    ]


def save_features(feature_table, output_dir):
    """Create the processed-data directory and save the feature table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILE
    feature_table.to_csv(output_path, index=False)
    return output_path


def main():
    """Build and save site-level adverse-event features."""
    project_root = Path(__file__).resolve().parents[2]
    synthetic_data_dir = project_root / "data" / "synthetic"
    processed_data_dir = project_root / "data" / "processed"

    adverse_events = load_adverse_events(synthetic_data_dir)
    feature_table = build_adverse_event_features(adverse_events)
    output_path = save_features(feature_table, processed_data_dir)

    print(f"Saved {len(feature_table)} feature rows to {output_path}")


if __name__ == "__main__":
    main()
