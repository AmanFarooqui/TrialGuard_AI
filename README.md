# TrialGuard AI

TrialGuard AI is a technical demonstration of clinical-trial site risk
intelligence built with synthetic data. It combines a supervised ML risk model,
statistical anomaly detection, explainability, and a FastAPI plus Streamlit
application for reviewing site-level signals.

**Important:** TrialGuard AI uses synthetic data, is not clinically validated,
and is **not a clinical decision-support system**. It must not be used to make
patient-care, trial eligibility, regulatory, or real-world trial decisions.

## Problem Statement

Clinical-trial operations generate data across sites, patients, visits,
laboratory results, adverse events, protocol deviations, and operational
activity. TrialGuard AI demonstrates how these signals can be combined to:

- Predict whether a site will become high-risk in the next monitoring period.
- Detect unusual site-period patterns.
- Prioritize records for human investigation.
- Explain the factors contributing to a model prediction.
- Track risk behavior over time.

The system is designed to support analytical review, not replace qualified
clinical, statistical, or operational judgment.

## Key Capabilities

- Synthetic multi-domain clinical-trial data generation.
- Data validation and feature engineering across site monitoring periods.
- Binary classification of `high_risk_next_period`.
- Tuned Random Forest risk probabilities with a 0.40 operating threshold.
- Isolation Forest anomaly detection.
- SHAP explanations for individual site predictions.
- Priority-based site intelligence and temporal trend analysis.
- FastAPI service and Streamlit dashboard.
- Automated API, data-pipeline, and model-artifact tests.

## Architecture

```text
Synthetic Data → Validation → Feature Engineering → ML Risk Model
→ Anomaly Detection → Site Intelligence → FastAPI → Streamlit Dashboard
```

The API loads processed datasets and the persisted tuned model. The dashboard
calls the API rather than loading model artifacts directly.

## Data and Scale

The project uses generated synthetic data representing:

- Trials and sites.
- Patients, visits, enrollment, and dropout.
- Laboratory results.
- Adverse events, including serious and severe events.
- Operational events, data entry delays, queries, and protocol deviations.

The ML dataset contains 250 unique site IDs across repeated monitoring periods.
Derived features describe data quality, operational performance, patient and
visit activity, adverse events, and enrollment deterioration. Synthetic labels
are created for `high_risk_next_period`; they do not represent clinical truth.

## ML Approach

The primary task is binary classification:

- `0`: not high-risk in the next monitoring period.
- `1`: high-risk in the next monitoring period.

The evaluated supervised models are:

- Logistic Regression baseline.
- Random Forest baseline.
- Gradient Boosting baseline.
- Tuned Random Forest, selected as the final model.

The final model uses current-period numeric features and excludes identifiers,
monitoring period, target labels, behavior labels, and the temporary composite
risk score. Risk status is assigned at a probability threshold of **0.40**:
probabilities at or above the threshold are `High Risk`; lower probabilities
are `Low Risk`.

### Final Model Evaluation

Held-out final test metrics at threshold 0.40:

| Metric | Score |
|---|---:|
| Accuracy | 0.744 |
| Precision | 0.622 |
| Recall | 0.647 |
| F1 | 0.634 |
| ROC-AUC | 0.766 |

Evaluation includes stratified cross-validation for model comparison,
time-based validation on later monitoring periods, and final held-out test
evaluation. These scores apply only to the synthetic evaluation data and are
not evidence of real-world clinical performance.

## Anomaly Detection

An Isolation Forest identifies unusual site-period patterns from the processed
feature data. Each record receives an anomaly prediction, anomaly score, and
`Normal` or `Anomaly` status. An anomaly is a statistical deviation from the
synthetic reference distribution, not proof of fraud, misconduct, a protocol
violation, or a safety issue.

## SHAP Explainability

For a selected site’s latest record, the API uses a tree-based SHAP explainer
to show the strongest local contributors to the prediction. The explanation
contains `feature`, `feature_value`, and `shap_value` for:

- `risk_increasing_factors`
- `risk_reducing_factors`

SHAP values explain model behavior and are not causal conclusions.

## Site Intelligence Priority System

Risk status and anomaly status are combined into an investigation priority:

| Risk status | Anomaly status | Priority |
|---|---|---|
| High Risk | Anomaly | Critical |
| High Risk | Normal | High |
| Low Risk | Anomaly | Medium |
| Low Risk | Normal | Low |

This is a triage ordering for the demonstration dashboard, not an automated
decision policy.

## API

The FastAPI application is defined in `src/api/main.py`:

| Endpoint | Purpose |
|---|---|
| `GET /` | Health status |
| `GET /summary` | Aggregate site and risk metrics |
| `GET /sites` | Site-level trend records |
| `GET /sites/{site_id}` | Site trend and monitoring history |
| `GET /sites/{site_id}/explanation` | SHAP explanation for a site |
| `GET /risk` | High-risk records |
| `GET /anomalies` | Anomaly records |
| `GET /critical` | Critical-priority records |

## Streamlit Dashboard

`dashboard/app.py` provides a browser-based interface for:

- Reviewing summary KPIs and risk distributions.
- Filtering critical and high-risk records.
- Inspecting site history and trends.
- Viewing SHAP factors for a selected site.
- Downloading filtered site intelligence data.

The dashboard expects the FastAPI service at `http://127.0.0.1:8000`.

## Project Structure

```text
dashboard/app.py                 Streamlit dashboard
src/api/main.py                  FastAPI application
src/data_generation/             Synthetic data generation
src/validation/                  Data validation
src/features/                    Feature and site-intelligence pipelines
src/models/                      Training and evaluation scripts
src/anomaly/                     Isolation Forest detection
src/explainability/              SHAP explanations and reports
data/synthetic/                  Generated source tables
data/processed/                  Feature and scoring outputs
models/                          Persisted model artifacts
outputs/reports/                 Evaluation reports
tests/                            API and pipeline tests
docs/model_card.md               Detailed model card
```

## Installation and Running

From the project root, create and activate a virtual environment, then install
the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the API:

```powershell
uvicorn src.api.main:app --reload
```

In a second terminal, start the dashboard:

```powershell
streamlit run dashboard/app.py
```

Run the automated tests:

```powershell
python -m pytest -v
```

The repository test suite contains **20 passing tests** covering API contracts,
processed-data schemas, missing values, target classes, model artifacts,
probability generation, anomaly outputs, site intelligence, and trend outputs.

## Responsible AI and Limitations

TrialGuard AI is a portfolio and engineering demonstration. It uses synthetic
data and has no real-world clinical validation, prospective validation, or
evidence of clinical utility. Model predictions are associations, not causes;
probabilities should not be interpreted as calibrated clinical risk without
additional analysis.

Human experts must review source data and context before acting on any signal.
Users should monitor data quality, drift, calibration, subgroup behavior, and
the risk of feedback loops. The 0.40 threshold is specific to this synthetic
demonstration and must not be transferred to another environment without a
new validation and decision analysis.

## Future Improvements

- Validate on appropriately governed real-world or external datasets.
- Add probability calibration and confidence or uncertainty reporting.
- Expand temporal and site-level drift monitoring.
- Add experiment tracking, model versioning, and data lineage.
- Evaluate fairness and performance across relevant site and trial subgroups.
- Add role-based access, audit logging, and production observability.
- Add robust model-serving and dashboard integration tests.
- Compare additional calibrated and time-aware model families.

##  Points

- Framed future-period site risk as a leakage-aware binary classification task.
- Built a reproducible pipeline from synthetic raw data through API delivery.
- Compared Logistic Regression, Random Forest, and Gradient Boosting before
	tuning the final Random Forest.
- Used stratified CV, time-based validation, and a held-out final evaluation.
- Selected a 0.40 threshold to improve recall while reporting precision and F1.
- Combined supervised risk prediction with Isolation Forest anomaly detection.
- Added SHAP explanations so analysts can inspect local model drivers.
- Encoded risk and anomaly signals into a transparent four-level priority rule.
- Added automated tests for API behavior, data contracts, and model inference.
- Communicated clearly that synthetic performance is not clinical validation.
