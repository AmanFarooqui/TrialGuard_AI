from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


INPUT_FILE = "ml_dataset.csv"
TARGET_COLUMN = "high_risk_next_period"
OUTPUT_FILE = "threshold_analysis.csv"
EXCLUDED_COLUMNS = {
    "site_id",
    TARGET_COLUMN,
    "behavior",
    "monitoring_period",
}
RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLDS = np.arange(0.20, 0.70 + 0.05, 0.05)


def load_dataset(project_root):
    """Load the ML dataset from the requested path, with a project fallback."""
    requested_path = project_root / "data" / "synthetic" / INPUT_FILE
    fallback_path = project_root / "data" / "processed" / INPUT_FILE

    if requested_path.is_file():
        return pd.read_csv(requested_path)
    if fallback_path.is_file():
        print(
            f"Requested dataset not found at {requested_path}; "
            f"using {fallback_path}."
        )
        return pd.read_csv(fallback_path)
    raise FileNotFoundError(
        f"Could not find the ML dataset at {requested_path} or {fallback_path}."
    )


def prepare_features(dataset):
    """Select numeric current-period features and separate the target."""
    feature_columns = [
        column_name
        for column_name in dataset.select_dtypes(include="number").columns
        if column_name not in EXCLUDED_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns are available for training.")

    return dataset[feature_columns], dataset[TARGET_COLUMN]


def split_dataset(features, target):
    """Create the fixed stratified training and testing partitions."""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def build_model():
    """Build the Random Forest configuration used by the baseline model."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )


def calculate_threshold_metrics(target_test, positive_probabilities):
    """Evaluate precision, recall, and F1 across candidate thresholds."""
    threshold_rows = []
    for threshold in THRESHOLDS:
        predictions = (positive_probabilities >= threshold).astype(int)
        threshold_rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(
                    target_test,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    target_test,
                    predictions,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    target_test,
                    predictions,
                    zero_division=0,
                ),
            }
        )
    return pd.DataFrame(threshold_rows).sort_values("threshold").reset_index(drop=True)


def print_analysis(threshold_results):
    """Print threshold metrics, the best F1 threshold, and default metrics."""
    print("Threshold analysis:")
    print(threshold_results.to_string(index=False))

    best_row = threshold_results.loc[threshold_results["f1_score"].idxmax()]
    print("\nBest F1-score threshold:")
    print(
        f"Threshold: {best_row['threshold']:.2f}, "
        f"Precision: {best_row['precision']:.4f}, "
        f"Recall: {best_row['recall']:.4f}, "
        f"F1-score: {best_row['f1_score']:.4f}"
    )

    default_row = threshold_results.loc[
        threshold_results["threshold"].eq(0.50)
    ].iloc[0]
    print("\nMetrics at default threshold 0.50:")
    print(f"Precision: {default_row['precision']:.4f}")
    print(f"Recall: {default_row['recall']:.4f}")
    print(f"F1-score: {default_row['f1_score']:.4f}")


def save_results(threshold_results, project_root):
    """Create the reports directory and save threshold results."""
    reports_dir = project_root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / OUTPUT_FILE
    threshold_results.to_csv(output_path, index=False)
    return output_path


def main():
    """Train Random Forest probabilities and analyze classification thresholds."""
    project_root = Path(__file__).resolve().parents[2]
    dataset = load_dataset(project_root)
    features, target = prepare_features(dataset)
    features_train, features_test, target_train, target_test = split_dataset(
        features, target
    )

    model = build_model()
    model.fit(features_train, target_train)
    positive_probabilities = model.predict_proba(features_test)[:, 1]
    threshold_results = calculate_threshold_metrics(
        target_test,
        positive_probabilities,
    )

    print(f"Training records: {len(features_train)}")
    print(f"Testing records: {len(features_test)}")
    print(f"Number of features: {features.shape[1]}")
    print_analysis(threshold_results)

    output_path = save_results(threshold_results, project_root)
    print(f"Saved threshold analysis to {output_path}")


if __name__ == "__main__":
    main()
