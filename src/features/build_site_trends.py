from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = "site_intelligence.csv"
OUTPUT_FILE = "site_trends.csv"
TIME_COLUMN = "monitoring_period"
REQUIRED_COLUMNS = [
    "site_id",
    TIME_COLUMN,
    "risk_probability",
    "risk_status",
    "anomaly_status",
    "priority",
]
PRIORITY_ORDER = {
    "Critical": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3,
}


def load_site_intelligence(processed_data_dir):
    """Load site-period intelligence records."""
    site_intelligence = pd.read_csv(processed_data_dir / INPUT_FILE)
    validate_columns(site_intelligence)
    return site_intelligence


def validate_columns(site_intelligence):
    """Ensure the input contains all columns needed for trend analysis."""
    missing_columns = [
        column_name
        for column_name in REQUIRED_COLUMNS
        if column_name not in site_intelligence.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Site intelligence is missing required columns: {missing_columns}"
        )


def calculate_trend_slope(site_data):
    """Calculate the risk-probability regression slope for one site."""
    if len(site_data) < 2:
        return 0.0
    return float(
        np.polyfit(
            site_data[TIME_COLUMN].to_numpy(),
            site_data["risk_probability"].to_numpy(),
            1,
        )[0]
    )


def classify_trend(slope):
    """Convert a numeric trend slope into a readable status."""
    if slope > 0.03:
        return "Deteriorating"
    if slope < -0.03:
        return "Improving"
    return "Stable"


def classify_priority(site_data):
    """Assign site-level priority from repeated critical conditions."""
    critical_periods = site_data["priority"].eq("Critical").sum()
    high_risk_periods = site_data["risk_status"].eq("High Risk").sum()
    anomaly_periods = site_data["anomaly_status"].eq("Anomaly").sum()

    if critical_periods >= 2:
        return "Critical"
    if high_risk_periods >= 2:
        return "High"
    if anomaly_periods >= 2:
        return "Medium"
    return "Low"


def build_site_trend(site_data):
    """Build one site-level trend record from sorted site-period data."""
    sorted_site_data = site_data.sort_values(TIME_COLUMN)
    first_risk_probability = sorted_site_data["risk_probability"].iloc[0]
    latest_risk_probability = sorted_site_data["risk_probability"].iloc[-1]
    trend_slope = calculate_trend_slope(sorted_site_data)

    return {
        "site_id": sorted_site_data["site_id"].iloc[0],
        "first_risk_probability": first_risk_probability,
        "latest_risk_probability": latest_risk_probability,
        "risk_probability_change": (
            latest_risk_probability - first_risk_probability
        ),
        "average_risk_probability": sorted_site_data[
            "risk_probability"
        ].mean(),
        "maximum_risk_probability": sorted_site_data[
            "risk_probability"
        ].max(),
        "high_risk_periods": sorted_site_data["risk_status"].eq(
            "High Risk"
        ).sum(),
        "anomaly_periods": sorted_site_data["anomaly_status"].eq(
            "Anomaly"
        ).sum(),
        "critical_periods": sorted_site_data["priority"].eq(
            "Critical"
        ).sum(),
        "latest_priority": sorted_site_data["priority"].iloc[-1],
        "trend_slope": trend_slope,
        "trend_status": classify_trend(trend_slope),
        "priority_status": classify_priority(sorted_site_data),
    }


def build_site_trends(site_intelligence):
    """Create, prioritize, and sort one trend record per site."""
    validate_columns(site_intelligence)
    sorted_data = site_intelligence.sort_values(
        ["site_id", TIME_COLUMN]
    )
    trend_records = [
        build_site_trend(site_data)
        for _, site_data in sorted_data.groupby("site_id", sort=False)
    ]
    trends = pd.DataFrame(trend_records)
    trends["priority_order"] = trends["priority_status"].map(PRIORITY_ORDER)
    return trends.sort_values(
        ["priority_order", "latest_risk_probability", "trend_slope"],
        ascending=[True, False, False],
    ).drop(columns="priority_order").reset_index(drop=True)


def print_summary(trends):
    """Print the requested trend and priority summary."""
    print(f"Total unique sites: {trends['site_id'].nunique()}")
    for trend_status in ["Deteriorating", "Stable", "Improving"]:
        print(
            f"Number of {trend_status} sites: "
            f"{trends['trend_status'].eq(trend_status).sum()}"
        )
    print(
        "Number of Critical priority sites: "
        f"{trends['priority_status'].eq('Critical').sum()}"
    )
    print("Top 10 sites by priority:")
    print(trends.head(10).to_string(index=False))


def save_site_trends(trends, processed_data_dir):
    """Save the site-level trend table."""
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    output_path = processed_data_dir / OUTPUT_FILE
    trends.to_csv(output_path, index=False)
    return output_path


def main():
    """Build and save site-level risk trends."""
    project_root = Path(__file__).resolve().parents[2]
    processed_data_dir = project_root / "data" / "processed"
    site_intelligence = load_site_intelligence(processed_data_dir)
    trends = build_site_trends(site_intelligence)
    print_summary(trends)
    output_path = save_site_trends(trends, processed_data_dir)
    print(f"Saved site trends to {output_path}")


if __name__ == "__main__":
    main()
