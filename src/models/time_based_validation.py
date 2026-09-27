from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "ml_dataset.csv"
OUTPUT_FILE = "time_based_validation.csv"
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
RANDOM_STATE = 42


def load_dataset(project_root):
    """Load the processed machine-learning dataset."""
    return pd.read_csv(project_root / "data" / "processed" / INPUT_FILE)


def split_by_monitoring_period(dataset):
    """Sort records and split them into fixed historical train/test periods."""
    if TIME_COLUMN not in dataset.columns:
        raise ValueError(f"Required time column is missing: {TIME_COLUMN}")

    sorted_dataset = dataset.sort_values(TIME_COLUMN).reset_index(drop=True)
    training_data = sorted_dataset[
        sorted_dataset[TIME_COLUMN].isin(TRAIN_PERIODS)
    ].copy()
    testing_data = sorted_dataset[
        sorted_dataset[TIME_COLUMN].isin(TEST_PERIODS)
    ].copy()

    if training_data.empty or testing_data.empty:
        raise ValueError("Training and testing periods must both contain records.")

    return training_data, testing_data


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
        raise ValueError("No numeric feature columns are available for training.")

    return dataset[feature_columns], dataset[TARGET_COLUMN]


def build_models():
    """Build the three baseline models used for comparison."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(model, features_train, target_train, features_test, target_test):
    """Fit one model on historical periods and evaluate it on later periods."""
    model.fit(features_train, target_train)
    predictions = model.predict(features_test)
    probabilities = model.predict_proba(features_test)[:, 1]
    matrix = confusion_matrix(target_test, predictions, labels=[0, 1])

    return {
        "Accuracy": accuracy_score(target_test, predictions),
        "Precision": precision_score(target_test, predictions, zero_division=0),
        "Recall": recall_score(target_test, predictions, zero_division=0),
        "F1-score": f1_score(target_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(target_test, probabilities),
        "Confusion matrix": str(matrix.tolist()),
    }


def create_comparison_table(
    models,
    features_train,
    target_train,
    features_test,
    target_test,
):
    """Evaluate all models and return a comparison table."""
    results = []
    for model_name, model in models.items():
        metrics = evaluate_model(
            model,
            features_train,
            target_train,
            features_test,
            target_test,
        )
        results.append({"Model": model_name, **metrics})
    return pd.DataFrame(results)


def save_results(results, project_root):
    """Create the reports directory and save validation results."""
    reports_dir = project_root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / OUTPUT_FILE
    results.to_csv(output_path, index=False)
    return output_path


def main():
    """Run chronological validation and save the model comparison."""
    project_root = Path(__file__).resolve().parents[2]
    dataset = load_dataset(project_root)
    training_data, testing_data = split_by_monitoring_period(dataset)
    features_train, target_train = prepare_features(training_data)
    features_test, target_test = prepare_features(testing_data)

    results = create_comparison_table(
        build_models(),
        features_train,
        target_train,
        features_test,
        target_test,
    )
    output_path = save_results(results, project_root)

    print(f"Training records: {len(training_data)}")
    print(f"Testing records: {len(testing_data)}")
    print(results.to_string(index=False))
    print(f"Saved time-based validation results to {output_path}")


if __name__ == "__main__":
    main()