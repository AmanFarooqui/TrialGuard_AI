from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


INPUT_FILE = "ml_dataset.csv"
MODEL_FILE = "tuned_random_forest.joblib"
FEATURE_IMPORTANCE_PLOT = "shap_feature_importance.png"
SUMMARY_PLOT = "shap_summary.png"
TARGET_COLUMN = "high_risk_next_period"
TIME_COLUMN = "monitoring_period"
EXCLUDED_COLUMNS = {
    "site_id",
    TIME_COLUMN,
    TARGET_COLUMN,
    "behavior",
}
TEST_PERIODS = range(8, 12)


def load_dataset(project_root):
    """Load and sort the processed machine-learning dataset."""
    dataset = pd.read_csv(project_root / "data" / "processed" / INPUT_FILE)
    if TIME_COLUMN not in dataset.columns:
        raise ValueError(f"Required time column is missing: {TIME_COLUMN}")
    return dataset.sort_values(TIME_COLUMN).reset_index(drop=True)


def select_model_features(dataset):
    """Select the numeric feature columns used by model training."""
    feature_columns = [
        column_name
        for column_name in dataset.select_dtypes(include="number").columns
        if column_name not in EXCLUDED_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns are available for explanation.")
    return feature_columns


def load_model(project_root):
    """Load the tuned Random Forest model artifact."""
    model_path = project_root / "models" / MODEL_FILE
    if not model_path.is_file():
        raise FileNotFoundError(f"Tuned model was not found at {model_path}")
    return joblib.load(model_path)


def get_final_test_features(dataset, feature_columns):
    """Return only final test-period feature rows."""
    test_data = dataset[dataset[TIME_COLUMN].isin(TEST_PERIODS)].copy()
    if test_data.empty:
        raise ValueError("Final test periods 8-11 contain no records.")
    return test_data, test_data[feature_columns]


def create_explainer(model):
    """Create a TreeExplainer for the fitted Random Forest."""
    return shap.TreeExplainer(model)


def get_positive_class_shap_values(explainer, features):
    """Return SHAP values for the positive high-risk class."""
    raw_values = explainer.shap_values(features)
    if isinstance(raw_values, list):
        if len(raw_values) != 2:
            raise ValueError("Expected binary-class SHAP values.")
        return raw_values[1]

    if raw_values.ndim == 3:
        if raw_values.shape[2] != 2:
            raise ValueError("Expected binary-class SHAP values.")
        return raw_values[:, :, 1]
    return raw_values


def calculate_feature_importance(shap_values, feature_columns):
    """Calculate mean absolute SHAP importance for each feature."""
    importance = pd.Series(
        abs(shap_values).mean(axis=0),
        index=feature_columns,
        name="mean_absolute_shap_value",
    )
    return importance.sort_values(ascending=False)


def save_plots(shap_values, features, feature_importance, figure_dir):
    """Save the global SHAP bar and beeswarm plots."""
    figure_dir.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(
        shap_values,
        features,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(figure_dir / FEATURE_IMPORTANCE_PLOT, dpi=150, bbox_inches="tight")
    plt.close()

    shap.summary_plot(
        shap_values,
        features,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(figure_dir / SUMMARY_PLOT, dpi=150, bbox_inches="tight")
    plt.close()


def explain_site_record(site_record, explainer, feature_columns, top_n=5):
    """Return the top contributing features for one site record."""
    if isinstance(site_record, pd.Series):
        site_features = site_record.reindex(feature_columns).to_frame().T
    else:
        site_features = site_record[feature_columns]
        if isinstance(site_features, pd.Series):
            site_features = site_features.to_frame().T

    if site_features.shape[0] != 1:
        raise ValueError("site_record must contain exactly one site record.")

    shap_values = get_positive_class_shap_values(explainer, site_features)
    contributions = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "feature_value": site_features.iloc[0].to_numpy(),
            "shap_contribution": shap_values[0],
        }
    )
    contributions["absolute_contribution"] = contributions[
        "shap_contribution"
    ].abs()
    return contributions.sort_values(
        "absolute_contribution",
        ascending=False,
    ).head(top_n).drop(columns="absolute_contribution")


def main():
    """Explain final-period predictions from the tuned Random Forest."""
    project_root = Path(__file__).resolve().parents[2]
    dataset = load_dataset(project_root)
    feature_columns = select_model_features(dataset)
    model = load_model(project_root)
    test_data, test_features = get_final_test_features(dataset, feature_columns)
    explainer = create_explainer(model)
    shap_values = get_positive_class_shap_values(explainer, test_features)
    feature_importance = calculate_feature_importance(
        shap_values,
        feature_columns,
    )

    save_plots(
        shap_values,
        test_features,
        feature_importance,
        project_root / "outputs" / "figures",
    )

    print("Mean absolute SHAP value by feature:")
    print(feature_importance.to_string())
    print(f"Explained final test records: {len(test_data)}")
    print("Top 5 contributions for the first final test record:")
    print(
        explain_site_record(
            test_data.iloc[0],
            explainer,
            feature_columns,
        ).to_string(index=False)
    )
    print(
        "Saved SHAP plots to "
        f"{project_root / 'outputs' / 'figures'}"
    )


if __name__ == "__main__":
    main()