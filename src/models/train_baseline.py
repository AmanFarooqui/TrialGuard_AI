from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "ml_dataset.csv"
MODEL_FILE = "logistic_regression_baseline.joblib"
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
	"""Create stratified training and testing partitions."""
	return train_test_split(
		features,
		target,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
		stratify=target,
	)


def build_pipeline():
	"""Build the standardized logistic-regression baseline pipeline."""
	return Pipeline(
		steps=[
			("scaler", StandardScaler()),
			("classifier", LogisticRegression(random_state=RANDOM_STATE)),
		]
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


def save_pipeline(model, models_dir):
	"""Create the model directory and save the trained pipeline."""
	models_dir.mkdir(parents=True, exist_ok=True)
	output_path = models_dir / MODEL_FILE
	joblib.dump(model, output_path)
	return output_path


def print_results(metrics, training_records, testing_records, feature_count):
	"""Print dataset sizes, feature count, and evaluation metrics."""
	print(f"Training records: {training_records}")
	print(f"Testing records: {testing_records}")
	print(f"Number of features: {feature_count}")
	print("Evaluation results:")
	for metric_name, metric_value in metrics.items():
		if metric_name == "Confusion matrix":
			print(f"{metric_name}:\n{metric_value}")
		else:
			print(f"{metric_name}: {metric_value:.4f}")


def main():
	"""Train, evaluate, and save the logistic-regression baseline."""
	project_root = Path(__file__).resolve().parents[2]
	processed_data_dir = project_root / "data" / "processed"
	models_dir = project_root / "models"

	dataset = load_dataset(processed_data_dir)
	features, target = prepare_features(dataset)
	features_train, features_test, target_train, target_test = split_dataset(
		features, target
	)

	model = build_pipeline()
	model.fit(features_train, target_train)
	metrics = evaluate_model(model, features_test, target_test)
	print_results(
		metrics,
		len(features_train),
		len(features_test),
		features.shape[1],
	)

	output_path = save_pipeline(model, models_dir)
	print(f"Saved trained pipeline to {output_path}")


if __name__ == "__main__":
	main()
