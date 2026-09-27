from pathlib import Path

import pandas as pd

from explain_random_forest import (
    create_explainer,
    get_final_test_features,
    get_positive_class_shap_values,
    load_dataset,
    load_model,
    select_model_features,
)


OUTPUT_FILE = "sample_site_risk_report.txt"
RISK_THRESHOLD = 0.40
FEATURE_LABELS = {
    "missing_data_count": "Missing Data",
    "protocol_deviation_count": "Protocol Deviations",
    "data_entry_delay_avg": "Average Data Entry Delay",
    "patient_dropout_rate": "Patient Dropout Rate",
    "average_visit_delay_days": "Average Visit Delay",
}


def readable_feature_name(feature_name):
    """Convert a model feature name into a user-facing label."""
    if feature_name in FEATURE_LABELS:
        return FEATURE_LABELS[feature_name]
    return feature_name.replace("_", " ").title()


def format_feature_value(value):
    """Format numeric feature values compactly for the report."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def calculate_site_contributions(site_record, explainer, feature_columns):
    """Return one site's feature values and positive-class SHAP contributions."""
    site_features = site_record[feature_columns].to_frame().T
    shap_values = get_positive_class_shap_values(explainer, site_features)
    return pd.DataFrame(
        {
            "feature_name": feature_columns,
            "feature_value": site_features.iloc[0].to_numpy(),
            "shap_contribution": shap_values[0],
        }
    )


def format_contribution_section(contributions, positive):
    """Format the five strongest positive or negative contributions."""
    if positive:
        selected = contributions[contributions["shap_contribution"] > 0]
        selected = selected.sort_values("shap_contribution", ascending=False)
    else:
        selected = contributions[contributions["shap_contribution"] < 0]
        selected = selected.sort_values("shap_contribution", ascending=True)

    selected = selected.head(5)
    if selected.empty:
        return "None\n"

    lines = []
    for _, row in selected.iterrows():
        lines.append(
            "- "
            f"{readable_feature_name(row['feature_name'])}: "
            f"value={format_feature_value(row['feature_value'])}, "
            f"SHAP contribution={row['shap_contribution']:.6f}"
        )
    return "\n".join(lines)


def build_report(site_record, probability, contributions):
    """Build the human-readable risk report for one site record."""
    risk_status = "HIGH RISK" if probability >= RISK_THRESHOLD else "NOT HIGH RISK"
    return "\n".join(
        [
            "TrialGuard AI Site Risk Report",
            "================================",
            f"site_id: {site_record['site_id']}",
            f"monitoring_period: {site_record['monitoring_period']}",
            f"predicted risk probability: {probability:.4f}",
            f"risk status (threshold {RISK_THRESHOLD:.2f}): {risk_status}",
            "",
            "Top risk drivers",
            "-----------------",
            format_contribution_section(contributions, positive=True),
            "",
            "Protective factors",
            "------------------",
            format_contribution_section(contributions, positive=False),
            "",
        ]
    )


def save_report(report, project_root):
    """Create the reports directory and save the text report."""
    reports_dir = project_root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / OUTPUT_FILE
    output_path.write_text(report, encoding="utf-8")
    return output_path


def main():
    """Generate a risk report for the first final-period site record."""
    project_root = Path(__file__).resolve().parents[2]
    dataset = load_dataset(project_root)
    feature_columns = select_model_features(dataset)
    model = load_model(project_root)
    test_data, test_features = get_final_test_features(dataset, feature_columns)
    site_record = test_data.iloc[0]

    probability = model.predict_proba(test_features.iloc[[0]])[0, 1]
    explainer = create_explainer(model)
    contributions = calculate_site_contributions(
        site_record,
        explainer,
        feature_columns,
    )
    report = build_report(site_record, probability, contributions)
    output_path = save_report(report, project_root)

    print(report)
    print(f"Saved risk report to {output_path}")


if __name__ == "__main__":
    main()