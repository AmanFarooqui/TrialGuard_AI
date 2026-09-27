from pathlib import Path

import pandas as pd


DATASET_FILES = {
	"trials": "trials.csv",
	"sites": "sites.csv",
	"patients": "patients.csv",
	"visits": "visits.csv",
	"lab_results": "lab_results.csv",
	"adverse_events": "adverse_events.csv",
	"operational_events": "operational_events.csv",
}

IMPORTANT_ID_COLUMNS = {
	"trials": ["trial_id"],
	"sites": ["site_id", "trial_id"],
	"patients": ["patient_id", "site_id"],
	"visits": ["visit_id", "patient_id"],
	"lab_results": ["lab_id", "patient_id", "visit_id"],
	"adverse_events": ["event_id", "patient_id", "site_id"],
	"operational_events": ["event_id", "site_id"],
}

DUPLICATE_ID_COLUMNS = {
	"trial_id": ["trials"],
	"site_id": ["sites"],
	"patient_id": ["patients"],
	"visit_id": ["visits"],
	"lab_id": ["lab_results"],
	"event_id": ["adverse_events", "operational_events"],
}

FOREIGN_KEY_RELATIONSHIPS = [
	("sites", "trial_id", "trials", "trial_id"),
	("patients", "site_id", "sites", "site_id"),
	("visits", "patient_id", "patients", "patient_id"),
	("lab_results", "patient_id", "patients", "patient_id"),
	("lab_results", "visit_id", "visits", "visit_id"),
	("adverse_events", "patient_id", "patients", "patient_id"),
	("adverse_events", "site_id", "sites", "site_id"),
	("operational_events", "site_id", "sites", "site_id"),
]


def check_files_exist(data_dir):
	"""Return whether every expected CSV file exists."""
	print("File existence:")
	all_files_exist = True

	for dataset_name, filename in DATASET_FILES.items():
		file_path = data_dir / filename
		exists = file_path.is_file()
		print(f"  {dataset_name}: {'found' if exists else 'missing'}")
		all_files_exist = all_files_exist and exists

	return all_files_exist


def load_datasets(data_dir):
	"""Load available CSV files and return them by dataset name."""
	datasets = {}
	print("\nDataset loading:")

	for dataset_name, filename in DATASET_FILES.items():
		file_path = data_dir / filename

		try:
			datasets[dataset_name] = pd.read_csv(file_path)
			print(f"  {dataset_name}: loaded successfully")
		except (FileNotFoundError, pd.errors.ParserError, OSError) as error:
			print(f"  {dataset_name}: failed to load ({error})")

	return datasets


def print_dataset_shapes(datasets):
	"""Print the row and column count for each loaded dataset."""
	print("\nDataset shapes:")

	for dataset_name in DATASET_FILES:
		if dataset_name in datasets:
			print(f"  {dataset_name}: {datasets[dataset_name].shape}")


def check_missing_ids(datasets):
	"""Print missing-value counts for important ID columns."""
	print("\nMissing values in important ID columns:")

	for dataset_name, id_columns in IMPORTANT_ID_COLUMNS.items():
		dataframe = datasets.get(dataset_name)
		if dataframe is None:
			continue

		for column_name in id_columns:
			if column_name in dataframe.columns:
				missing_count = dataframe[column_name].isna().sum()
				print(f"  {dataset_name}.{column_name}: {missing_count}")
			else:
				print(f"  {dataset_name}.{column_name}: column missing")


def check_duplicate_ids(datasets):
	"""Print duplicate counts for each requested primary ID column."""
	print("\nDuplicate ID counts:")

	for column_name, dataset_names in DUPLICATE_ID_COLUMNS.items():
		for dataset_name in dataset_names:
			dataframe = datasets.get(dataset_name)
			if dataframe is None:
				continue

			if column_name in dataframe.columns:
				duplicate_count = dataframe[column_name].duplicated().sum()
				print(f"  {dataset_name}.{column_name}: {duplicate_count}")
			else:
				print(f"  {dataset_name}.{column_name}: column missing")


def check_foreign_key(datasets, child_dataset, child_column,
					  parent_dataset, parent_column):
	"""Return the number of child records without a matching parent key."""
	child_dataframe = datasets.get(child_dataset)
	parent_dataframe = datasets.get(parent_dataset)

	if child_dataframe is None or parent_dataframe is None:
		return None
	if child_column not in child_dataframe.columns:
		return None
	if parent_column not in parent_dataframe.columns:
		return None

	parent_keys = parent_dataframe[parent_column]
	return (~child_dataframe[child_column].isin(parent_keys)).sum()


def check_foreign_keys(datasets):
	"""Print orphan counts for all configured foreign-key relationships."""
	print("\nForeign-key orphan records:")

	for child_dataset, child_column, parent_dataset, parent_column in (
		FOREIGN_KEY_RELATIONSHIPS
	):
		orphan_count = check_foreign_key(
			datasets,
			child_dataset,
			child_column,
			parent_dataset,
			parent_column,
		)
		relationship = (
			f"{child_dataset}.{child_column} -> "
			f"{parent_dataset}.{parent_column}"
		)

		if orphan_count is None:
			print(f"  {relationship}: unable to check")
		else:
			print(f"  {relationship}: {orphan_count}")


def check_business_rules(datasets):
	"""Print invalid-record counts for business and data-quality rules."""
	print("\nBusiness and data-quality checks:")

	patients = datasets.get("patients")
	if patients is not None:
		invalid_age = ~patients["age"].between(18, 100)
		invalid_status = ~patients["status"].isin(
			["Active", "Completed", "Dropped Out"]
		)
		print(f"  patients.age outside 18-100: {invalid_age.sum()}")
		print(f"  patients.status has invalid value: {invalid_status.sum()}")

	visits = datasets.get("visits")
	if visits is not None:
		invalid_delay = visits["visit_delay_days"].lt(0)
		actual_dates = pd.to_datetime(visits["actual_date"], errors="coerce")
		scheduled_dates = pd.to_datetime(
			visits["scheduled_date"], errors="coerce"
		)
		missing_actual_date = actual_dates.isna()
		invalid_completed_date = (
			visits["completed"] & missing_actual_date
		)
		invalid_incomplete_date = (
			~visits["completed"] & ~missing_actual_date
		)
		invalid_date_order = (
			actual_dates.notna()
			& scheduled_dates.notna()
			& (actual_dates < scheduled_dates)
		)
		print(f"  visits.visit_delay_days is negative: {invalid_delay.sum()}")
		print(
			"  completed visits missing actual_date: "
			f"{invalid_completed_date.sum()}"
		)
		print(
			"  incomplete visits with actual_date: "
			f"{invalid_incomplete_date.sum()}"
		)
		print(
			"  visits.actual_date before scheduled_date: "
			f"{invalid_date_order.sum()}"
		)

	lab_results = datasets.get("lab_results")
	if lab_results is not None:
		invalid_reference_range = ~(
			lab_results["reference_low"] < lab_results["reference_high"]
		)
		print(
			"  lab_results.reference_low not less than reference_high: "
			f"{invalid_reference_range.sum()}"
		)

	operational_events = datasets.get("operational_events")
	if operational_events is not None:
		invalid_value = operational_events["value"].lt(0)
		invalid_period = ~operational_events["monitoring_period"].between(
			1, 12
		)
		print(f"  operational_events.value is negative: {invalid_value.sum()}")
		print(
			"  operational_events.monitoring_period outside 1-12: "
			f"{invalid_period.sum()}"
		)

	adverse_events = datasets.get("adverse_events")
	if adverse_events is not None:
		invalid_reported_delay = adverse_events["reported_delay_days"].lt(0)
		print(
			"  adverse_events.reported_delay_days is negative: "
			f"{invalid_reported_delay.sum()}"
		)


def main():
	"""Run the initial synthetic-data validation checks."""
	project_root = Path(__file__).resolve().parents[2]
	data_dir = project_root / "data" / "synthetic"

	check_files_exist(data_dir)
	datasets = load_datasets(data_dir)
	print_dataset_shapes(datasets)
	check_missing_ids(datasets)
	check_duplicate_ids(datasets)
	check_foreign_keys(datasets)
	check_business_rules(datasets)


if __name__ == "__main__":
	main()
