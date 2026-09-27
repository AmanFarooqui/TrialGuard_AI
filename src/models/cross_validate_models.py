from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "ml_dataset.csv"
OUTPUT_FILE = "cross_validation_results.csv"
TARGET_COLUMN = "high_risk_next_period"
EXCLUDED_COLUMNS = {
    "site_id",
    TARGET_COLUMN,
    "behavior",
    "monitoring_period",
}
RANDOM_STATE = 42
N_SPLITS = 5
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def load_dataset(project_root):
    """Load the processed machine-learning dataset."""
    return pd.read_csv(project_root / "data" / "processed" / INPUT_FILE)


def prepare_features(dataset):
    """Select numeric features and separate the target column."""
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


def evaluate_model(model, features, target, cross_validator):
    """Calculate cross-validation scores for one model."""
    scores = cross_validate(
        model,
        features,
        target,
        cv=cross_validator,
        scoring=SCORING,
        n_jobs=-1,
    )
    return [
        {
            "mean": scores[f"test_{metric_name}"].mean(),
            "std": scores[f"test_{metric_name}"].std(),
        }
        for metric_name in SCORING
    ]


def create_results(models, features, target):
    """Evaluate all models and return one row per model and metric."""
    cross_validator = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    result_rows = []
    for model_name, model in models.items():
        model_scores = evaluate_model(
            model,
            features,
            target,
            cross_validator,
        )
        for metric_name, metric_scores in zip(SCORING, model_scores):
            result_rows.append(
                {
                    "model": model_name,
                    "metric": metric_name,
                    "mean": metric_scores["mean"],
                    "standard_deviation": metric_scores["std"],
                }
            )
    return pd.DataFrame(result_rows)


def save_results(results, project_root):
    """Create the reports directory and save cross-validation results."""
    reports_dir = project_root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / OUTPUT_FILE
    results.to_csv(output_path, index=False)
    return output_path


def main():
    """Run stratified cross-validation and save the model comparison."""
    project_root = Path(__file__).resolve().parents[2]
    dataset = load_dataset(project_root)
    features, target = prepare_features(dataset)
    results = create_results(build_models(), features, target)
    output_path = save_results(results, project_root)

    print(results.to_string(index=False))
    print(f"Saved cross-validation results to {output_path}")


if __name__ == "__main__":
    main()