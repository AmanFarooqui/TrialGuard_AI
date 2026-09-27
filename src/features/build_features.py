from pathlib import Path

import pandas as pd


INPUT_FILES = {
    "operational_events": "operational_events.csv",
    "sites": "sites.csv",
}
OUTPUT_FILE = "site_operational_features.csv"


# Event-type summaries become model-ready columns after the pivot.
FEATURE_COLUMN_MAP = {
    ("event_count", "Missing Data"): "missing_data_count",
    ("event_count", "Protocol Deviation"): "protocol_deviation_count",
    ("value_avg", "Data Entry Delay"): "data_entry_delay_avg",
    ("value_avg", "Query Resolution"): "query_resolution_avg",
}


def load_input_data(data_dir):
    """Load operational events and site metadata from synthetic data."""
    operational_events = pd.read_csv(data_dir / INPUT_FILES["operational_events"])
    sites = pd.read_csv(data_dir / INPUT_FILES["sites"])
    return operational_events, sites


def build_operational_features(operational_events):
    """Aggregate event counts and values by site, period, and event type."""
    grouped_events = (
        operational_events.groupby(
            ["site_id", "monitoring_period", "event_type"]
        )
        .agg(
            event_count=("event_id", "count"),
            value_avg=("value", "mean"),
        )
        .reset_index()
    )

    pivoted_events = grouped_events.pivot_table(
        index=["site_id", "monitoring_period"],
        columns="event_type",
        values=["event_count", "value_avg"],
        fill_value=0,
        aggfunc="first",
    ).reset_index()

    pivoted_events.columns = [
        "_".join(str(level) for level in column if str(level))
        if isinstance(column, tuple)
        else str(column)
        for column in pivoted_events.columns
    ]

    feature_table = pivoted_events.rename(
        columns={
            "event_count_Missing Data": "missing_data_count",
            "event_count_Protocol Deviation": "protocol_deviation_count",
            "value_avg_Data Entry Delay": "data_entry_delay_avg",
            "value_avg_Query Resolution": "query_resolution_avg",
        }
    )

    for feature_name in FEATURE_COLUMN_MAP.values():
        if feature_name not in feature_table.columns:
            feature_table[feature_name] = 0

    return feature_table[
        [
            "site_id",
            "monitoring_period",
            "missing_data_count",
            "protocol_deviation_count",
            "data_entry_delay_avg",
            "query_resolution_avg",
        ]
    ]


def add_behavior_for_inspection(feature_table, sites):
    """Attach site behavior for inspection, outside the feature calculations."""
    return feature_table.merge(
        sites[["site_id", "behavior"]],
        on="site_id",
        how="left",
    )


def save_features(feature_table, output_dir):
    """Create the processed-data directory and save the feature table."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILE
    feature_table.to_csv(output_path, index=False)
    return output_path


def main():
    """Build and save site-level operational monitoring features."""
    project_root = Path(__file__).resolve().parents[2]
    synthetic_data_dir = project_root / "data" / "synthetic"
    processed_data_dir = project_root / "data" / "processed"

    operational_events, sites = load_input_data(synthetic_data_dir)
    feature_table = build_operational_features(operational_events)
    feature_table = add_behavior_for_inspection(feature_table, sites)
    output_path = save_features(feature_table, processed_data_dir)

    print(f"Saved {len(feature_table)} feature rows to {output_path}")


if __name__ == "__main__":
    main()
