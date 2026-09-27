from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


INPUT_FILE = "ml_dataset.csv"
MODEL_FILE = "gradient_boosting_baseline.joblib"
LOGISTIC_REGRESSION_MODEL_FILE = "logistic_regression_baseline.joblib"
RANDOM_FOREST_MODEL_FILE = "random_forest_baseline.joblib"
TARGET_COLUMN = "high_risk_next_period"
EXCLUDED_COLUMNS = {
    "site_id",
    TARGET_COLUMN,
    "behavior",
    "monitoring_period",
}
RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_dataset(processed_data_dir):
    """Load the processed machine-learning dataset."""
    return pd.read_csv(processed_data_dir / INPUT_FILE)


def prepare_features(dataset):
    """Select numeric current-period features and separate the target."""
    feature_columns = [
        column_name
        for column_name in dataset.select_dtypes(include="number").columns
        if column_name not in EXCLUDED_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns are available for training.")

    features = dataset[feature_columns]
    target = dataset[TARGET_COLUMN]
    return features, target


def split_dataset(features, target):
    """Create the same stratified partitions used by the other baselines."""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def build_model():
    """Build the configured Gradient Boosting classifier."""
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_STATE,
    )


def evaluate_model(model, features_test, target_test):
    """Calculate classification metrics on the held-out test set."""
    predictions = model.predict(features_test)
    probabilities = model.predict_proba(features_test)[:, 1]
    return {
        "Accuracy": accuracy_score(target_test, predictions),
        "Precision": precision_score(target_test, predictions, zero_division=0),
        "Recall": recall_score(target_test, predictions, zero_division=0),
        "F1-score": f1_score(target_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(target_test, probabilities),
        "Confusion matrix": confusion_matrix(target_test, predictions),
    }


def print_metrics(model_name, metrics):
    """Print evaluation metrics in a consistent format."""
    print(f"{model_name} evaluation results:")
    for metric_name, metric_value in metrics.items():
        if metric_name == "Confusion matrix":
            print(f"{metric_name}:\n{metric_value}")
        else:
            print(f"{metric_name}: {metric_value:.4f}")


def print_feature_importances(model, feature_names):
    """Print feature importances from highest to lowest."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    print("Gradient Boosting feature importances (highest to lowest):")
    print(importances.sort_values(ascending=False).to_string())


def load_saved_model(models_dir, filename):
    """Load a saved comparison model when its artifact is available."""
    model_path = models_dir / filename
    if not model_path.is_file():
        return None
    return joblib.load(model_path)


def print_comparison(metrics_by_model):
    """Print metric values for all available models side by side."""
    comparison = pd.DataFrame(
        {
            model_name: {
                metric_name: metric_value
                for metric_name, metric_value in metrics.items()
                if metric_name != "Confusion matrix"
            }
            for model_name, metrics in metrics_by_model.items()
        }
    )
    print("Model comparison:")
    print(comparison)


def save_model(model, models_dir):
    """Create the model directory and save the trained model."""
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / MODEL_FILE
    joblib.dump(model, output_path)
    return output_path


def main():
    """Train, evaluate, compare, and save the Gradient Boosting baseline."""
    project_root = Path(__file__).resolve().parents[2]
    processed_data_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"

    dataset = load_dataset(processed_data_dir)
    features, target = prepare_features(dataset)
    features_train, features_test, target_train, target_test = split_dataset(
        features, target
    )

    model = build_model()
    model.fit(features_train, target_train)
    gradient_boosting_metrics = evaluate_model(
        model,
        features_test,
        target_test,
    )

    print(f"Training records: {len(features_train)}")
    print(f"Testing records: {len(features_test)}")
    print(f"Number of features: {features.shape[1]}")
    print_metrics("Gradient Boosting", gradient_boosting_metrics)
    print_feature_importances(model, features.columns)

    metrics_by_model = {"Gradient Boosting": gradient_boosting_metrics}
    comparison_models = {
        "Logistic Regression": LOGISTIC_REGRESSION_MODEL_FILE,
        "Random Forest": RANDOM_FOREST_MODEL_FILE,
    }
    for model_name, model_filename in comparison_models.items():
        comparison_model = load_saved_model(models_dir, model_filename)
        if comparison_model is None:
            print(
                f"{model_name} comparison unavailable: "
                f"{model_filename} was not found."
            )
            continue
        comparison_metrics = evaluate_model(
            comparison_model,
            features_test,
            target_test,
        )
        print_metrics(f"{model_name} baseline", comparison_metrics)
        metrics_by_model[model_name] = comparison_metrics

    print_comparison(metrics_by_model)
    output_path = save_model(model, models_dir)
    print(f"Saved Gradient Boosting model to {output_path}")


if __name__ == "__main__":
    main()
