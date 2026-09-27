from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


INPUT_FILE = "ml_dataset.csv"
TARGET_COLUMN = "high_risk_next_period"
IDENTIFIER_COLUMNS = {"site_id", "monitoring_period"}
FIGURE_NAMES = {
    "target_distribution": "target_class_distribution.png",
    "missing_data": "missing_data_vs_high_risk.png",
    "dropout_rate": "patient_dropout_rate_vs_high_risk.png",
    "visit_delay": "average_visit_delay_vs_high_risk.png",
    "correlation_heatmap": "numeric_feature_correlation_heatmap.png",
}


def load_dataset(processed_data_dir):
    """Load the processed ML dataset."""
    return pd.read_csv(processed_data_dir / INPUT_FILE)


def get_model_numeric_columns(dataset):
    """Return numeric model features, excluding identifiers and behavior."""
    excluded_columns = IDENTIFIER_COLUMNS | {TARGET_COLUMN, "behavior"}
    return [
        column_name
        for column_name in dataset.select_dtypes(include="number").columns
        if column_name not in excluded_columns
    ]


def print_dataset_overview(dataset):
    """Print the requested dataset-level EDA information."""
    print(f"Dataset shape: {dataset.shape}")
    print(f"Column names: {list(dataset.columns)}")
    print("Data types:")
    print(dataset.dtypes)
    print("Missing values by column:")
    print(dataset.isna().sum())
    print(f"Duplicate rows: {dataset.duplicated().sum()}")
    print("Descriptive statistics for numeric columns:")
    print(dataset.describe(include="number"))
    print("High-risk target distribution:")
    print(dataset[TARGET_COLUMN].value_counts().sort_index())
    print(f"Unique sites: {dataset['site_id'].nunique()}")
    print(f"Monitoring periods: {dataset['monitoring_period'].nunique()}")


def calculate_correlations(dataset):
    """Calculate feature correlations with the target, excluding identifiers."""
    numeric_features = get_model_numeric_columns(dataset)
    correlation_data = dataset[numeric_features + [TARGET_COLUMN]]
    correlations = correlation_data.corr()[TARGET_COLUMN].drop(TARGET_COLUMN)
    return correlations.sort_values(key=lambda values: values.abs(), ascending=False)


def save_target_distribution(dataset, figure_dir):
    """Save the target class distribution chart."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=dataset, x=TARGET_COLUMN)
    plt.title("High-Risk Next-Period Target Distribution")
    plt.xlabel("High risk next period")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(figure_dir / FIGURE_NAMES["target_distribution"], dpi=150)
    plt.close()


def save_target_boxplot(dataset, feature_name, title, filename, figure_dir):
    """Save a feature distribution grouped by the target."""
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=dataset, x=TARGET_COLUMN, y=feature_name)
    plt.title(title)
    plt.xlabel("High risk next period")
    plt.ylabel(feature_name.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(figure_dir / filename, dpi=150)
    plt.close()


def save_correlation_heatmap(dataset, figure_dir):
    """Save a heatmap for numeric model features and the target."""
    numeric_features = get_model_numeric_columns(dataset)
    correlation_data = dataset[numeric_features + [TARGET_COLUMN]].corr()

    figure_size = max(10, len(correlation_data.columns) * 0.6)
    plt.figure(figsize=(figure_size, figure_size))
    sns.heatmap(correlation_data, cmap="coolwarm", center=0, annot=False)
    plt.title("Numeric Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(figure_dir / FIGURE_NAMES["correlation_heatmap"], dpi=150)
    plt.close()


def save_visualizations(dataset, figure_dir):
    """Create the requested EDA figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_target_distribution(dataset, figure_dir)
    save_target_boxplot(
        dataset,
        "missing_data_count",
        "Missing Data Count vs High-Risk Target",
        FIGURE_NAMES["missing_data"],
        figure_dir,
    )
    save_target_boxplot(
        dataset,
        "patient_dropout_rate",
        "Patient Dropout Rate vs High-Risk Target",
        FIGURE_NAMES["dropout_rate"],
        figure_dir,
    )
    save_target_boxplot(
        dataset,
        "average_visit_delay_days",
        "Average Visit Delay vs High-Risk Target",
        FIGURE_NAMES["visit_delay"],
        figure_dir,
    )
    save_correlation_heatmap(dataset, figure_dir)


def print_pattern_summary(dataset, correlations):
    """Print a short summary of the most visible target relationships."""
    print("\nImportant patterns:")
    target_means = dataset.groupby(TARGET_COLUMN)[
        [
            "missing_data_count",
            "patient_dropout_rate",
            "average_visit_delay_days",
        ]
    ].mean()
    print("Average selected features by target class:")
    print(target_means)

    if not correlations.empty:
        strongest_feature = correlations.index[0]
        strongest_value = correlations.iloc[0]
        print(
            "Strongest absolute numeric-feature correlation with the target: "
            f"{strongest_feature} ({strongest_value:.3f})"
        )
    else:
        print("No numeric model features were available for correlation analysis.")


def main():
    """Run exploratory analysis and save the requested visualizations."""
    project_root = Path(__file__).resolve().parents[2]
    processed_data_dir = project_root / "data" / "processed"
    figure_dir = project_root / "outputs" / "figures"

    dataset = load_dataset(processed_data_dir)
    print_dataset_overview(dataset)

    correlations = calculate_correlations(dataset)
    print("\nCorrelations with high_risk_next_period:")
    print(correlations)

    save_visualizations(dataset, figure_dir)
    print_pattern_summary(dataset, correlations)
    print(f"\nSaved visualizations to {figure_dir}")


if __name__ == "__main__":
    main()
