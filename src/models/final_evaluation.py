from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


INPUT_FILE = "ml_dataset.csv"
MODEL_FILE = "tuned_random_forest.joblib"
OUTPUT_FILE = "final_model_evaluation.csv"
TARGET_COLUMN = "high_risk_next_period"
TIME_COLUMN = "monitoring_period"
EXCLUDED_COLUMNS = {
    "site_id",
    TIME_COLUMN,
    TARGET_COLUMN,
    "behavior",
}
TRAIN_PERIODS = range(1, 8)
TEST_PERIODS = range(8, 12)
DEFAULT_THRESHOLD = 0.50
PREVIOUSLY_SELECTED_THRESHOLD = 0.40


def load_dataset(project_root):
    """Load and sort the processed machine-learning dataset."""
    dataset = pd.read_csv(project_root / "data" / "processed" / INPUT_FILE)
    if TIME_COLUMN not in dataset.columns:
        raise ValueError(f"Required time column is missing: {TIME_COLUMN}")
    return dataset.sort_values(TIME_COLUMN).reset_index(drop=True)


def split_dataset(dataset):
    """Create development and final test slices without shuffling."""
    training_data = dataset[dataset[TIME_COLUMN].isin(TRAIN_PERIODS)].copy()
    test_data = dataset[dataset[TIME_COLUMN].isin(TEST_PERIODS)].copy()
    if training_data.empty or test_data.empty:
        raise ValueError("Training and final test periods must contain records.")
    return training_data, test_data


def prepare_features(dataset):
    """Select numeric model features and separate the target column."""
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Required target column is missing: {TARGET_COLUMN}")

    feature_columns = [
        column_name
        for column_name in dataset.select_dtypes(include="number").columns
        if column_name not in EXCLUDED_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns are available for evaluation.")

    return dataset[feature_columns], dataset[TARGET_COLUMN]


def load_and_train_model(project_root, features_train, target_train):
    """Load the tuned model configuration and fit it on periods 1-7."""
    model_path = project_root / "models" / MODEL_FILE
    if not model_path.is_file():
        raise FileNotFoundError(f"Tuned model was not found at {model_path}")

    model = joblib.load(model_path)
    model.fit(features_train, target_train)
    return model


def evaluate_threshold(target_test, probabilities, threshold):
    """Calculate final-test metrics for a fixed probability threshold."""
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(target_test, predictions, labels=[0, 1])
    return {
        "accuracy": accuracy_score(target_test, predictions),
        "precision": precision_score(target_test, predictions, zero_division=0),
        "recall": recall_score(target_test, predictions, zero_division=0),
        "f1_score": f1_score(target_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(target_test, probabilities),
        "confusion_matrix": str(matrix.tolist()),
        "classification_report": classification_report(
            target_test,
            predictions,
            labels=[0, 1],
            target_names=["not_high_risk", "high_risk"],
            zero_division=0,
        ),
    }


def create_results(target_test, probabilities):
    """Compare the default and previously selected fixed thresholds."""
    threshold_definitions = (
        (DEFAULT_THRESHOLD, "Default threshold"),
        (
            PREVIOUSLY_SELECTED_THRESHOLD,
            "Previously selected threshold",
        ),
    )
    rows = []
    for threshold, label in threshold_definitions:
        rows.append(
            {
                "threshold_label": label,
                "threshold": threshold,
                **evaluate_threshold(target_test, probabilities, threshold),
            }
        )
    return pd.DataFrame(rows)


def save_results(results, project_root):
    """Create the reports directory and save final evaluation results."""
    reports_dir = project_root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / OUTPUT_FILE
    results.to_csv(output_path, index=False)
    return output_path


def print_results(results):
    """Print both threshold evaluations with detailed reports."""
    print("Final evaluation results:")
    for _, result in results.iterrows():
        print(f"\n{result['threshold_label']} ({result['threshold']:.2f}):")
        for metric_name in (
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
        ):
            print(f"{metric_name}: {result[metric_name]:.4f}")
        print(f"Confusion matrix:\n{result['confusion_matrix']}")
        print(f"Classification report:\n{result['classification_report']}")


def main():
    """Train the tuned model on development data and evaluate final periods."""
    project_root = Path(__file__).resolve().parents[2]
    dataset = load_dataset(project_root)
    training_data, test_data = split_dataset(dataset)
    features_train, target_train = prepare_features(training_data)
    features_test, target_test = prepare_features(test_data)

    model = load_and_train_model(project_root, features_train, target_train)
    probabilities = model.predict_proba(features_test)[:, 1]
    results = create_results(target_test, probabilities)
    output_path = save_results(results, project_root)

    print(f"Training records: {len(training_data)}")
    print(f"Final test records: {len(test_data)}")
    print_results(results)
    print(f"Saved final evaluation results to {output_path}")


if __name__ == "__main__":
    main()