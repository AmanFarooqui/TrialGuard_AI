from itertools import product
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


INPUT_FILE = "ml_dataset.csv"
MODEL_FILE = "tuned_random_forest.joblib"
OUTPUT_FILE = "random_forest_tuning_results.csv"
TARGET_COLUMN = "high_risk_next_period"
TIME_COLUMN = "monitoring_period"
EXCLUDED_COLUMNS = {
    "site_id",
    TIME_COLUMN,
    TARGET_COLUMN,
    "behavior",
}
DEVELOPMENT_PERIODS = range(1, 8)
VALIDATION_FOLDS = (
    (range(1, 4), 4),
    (range(1, 5), 5),
    (range(1, 6), 6),
    (range(1, 7), 7),
)
RANDOM_STATE = 42
PARAMETER_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"],
}


def load_dataset(project_root):
    """Load and sort the processed machine-learning dataset."""
    dataset = pd.read_csv(project_root / "data" / "processed" / INPUT_FILE)
    if TIME_COLUMN not in dataset.columns:
        raise ValueError(f"Required time column is missing: {TIME_COLUMN}")
    return dataset.sort_values(TIME_COLUMN).reset_index(drop=True)


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


def build_parameter_combinations():
    """Return every requested Random Forest hyperparameter combination."""
    parameter_names = list(PARAMETER_GRID)
    return [
        {
            "n_estimators": int(values[0]),
            "max_depth": (
                None if values[1] is None else int(values[1])
            ),
            "min_samples_split": int(values[2]),
            "min_samples_leaf": int(values[3]),
            "class_weight": values[4],
        }
        for values in product(*(PARAMETER_GRID[name] for name in parameter_names))
    ]


def build_model(parameters):
    """Build a Random Forest with the supplied hyperparameters."""
    typed_parameters = {
        "n_estimators": int(parameters["n_estimators"]),
        "max_depth": (
            None
            if parameters["max_depth"] is None
            else int(parameters["max_depth"])
        ),
        "min_samples_split": int(parameters["min_samples_split"]),
        "min_samples_leaf": int(parameters["min_samples_leaf"]),
        "class_weight": parameters["class_weight"],
    }
    if typed_parameters["class_weight"] != "balanced":
        raise ValueError("class_weight must remain 'balanced'.")
    return RandomForestClassifier(random_state=RANDOM_STATE, **typed_parameters)


def calculate_fold_scores(parameters, dataset, features, target):
    """Calculate F1-score for each expanding-window validation fold."""
    fold_scores = []
    for training_periods, validation_period in VALIDATION_FOLDS:
        training_mask = dataset[TIME_COLUMN].isin(training_periods)
        validation_mask = dataset[TIME_COLUMN].eq(validation_period)

        model = build_model(parameters)
        model.fit(features.loc[training_mask], target.loc[training_mask])
        predictions = model.predict(features.loc[validation_mask])
        fold_scores.append(
            f1_score(
                target.loc[validation_mask],
                predictions,
                zero_division=0,
            )
        )
    return fold_scores


def tune_model(dataset, features, target):
    """Evaluate the parameter grid and return results plus the best parameters."""
    parameter_combinations = build_parameter_combinations()
    result_rows = []
    mean_scores = []
    for parameters in parameter_combinations:
        fold_scores = calculate_fold_scores(parameters, dataset, features, target)
        mean_validation_f1 = sum(fold_scores) / len(fold_scores)
        result_rows.append(
            {
                **parameters,
                "max_depth": (
                    "None" if parameters["max_depth"] is None else parameters["max_depth"]
                ),
                "fold_1_f1": fold_scores[0],
                "fold_2_f1": fold_scores[1],
                "fold_3_f1": fold_scores[2],
                "fold_4_f1": fold_scores[3],
                "mean_validation_f1": mean_validation_f1,
            }
        )
        mean_scores.append(mean_validation_f1)

    results = pd.DataFrame(result_rows).sort_values(
        "mean_validation_f1",
        ascending=False,
    ).reset_index(drop=True)
    best_index = max(range(len(mean_scores)), key=mean_scores.__getitem__)
    best_parameters = parameter_combinations[best_index].copy()
    return results, best_parameters


def save_results(results, project_root):
    """Create the reports directory and save tuning results."""
    reports_dir = project_root / "outputs" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / OUTPUT_FILE
    results.to_csv(output_path, index=False)
    return output_path


def save_model(model, project_root):
    """Create the models directory and save the tuned model."""
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    output_path = models_dir / MODEL_FILE
    joblib.dump(model, output_path)
    return output_path


def main():
    """Tune on periods 1-7 and save the best model for later evaluation."""
    project_root = Path(__file__).resolve().parents[2]
    dataset = load_dataset(project_root)
    development_data = dataset[dataset[TIME_COLUMN].isin(DEVELOPMENT_PERIODS)].copy()
    if development_data.empty:
        raise ValueError("Development periods 1-7 must contain records.")

    features, target = prepare_features(development_data)
    results, best_parameters = tune_model(
        development_data,
        features,
        target,
    )

    best_model = build_model(best_parameters)
    best_model.fit(features, target)
    model_path = save_model(best_model, project_root)
    results_path = save_results(results, project_root)

    print("Best hyperparameters:")
    print(best_parameters)
    print(
        "Best mean validation F1-score: "
        f"{results.loc[0, 'mean_validation_f1']:.4f}"
    )
    print("Validation results:")
    print(results.to_string(index=False))
    print(f"Saved tuned model to {model_path}")
    print(f"Saved tuning results to {results_path}")


if __name__ == "__main__":
    main()