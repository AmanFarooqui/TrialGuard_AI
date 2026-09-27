# TrialGuard AI Tuned Random Forest Model Card

## Model Summary

TrialGuard AI is a technical demonstration of machine-learning methods for
clinical-trial site risk intelligence. It is **not a clinical decision-support
system**, is not validated for use in real clinical trials, and must not be
used to make clinical, patient-safety, regulatory, or trial-operational
decisions.

The primary model is a tuned Random Forest classifier that estimates whether a
site will become high-risk during the next monitoring period. The binary target
is `high_risk_next_period`:

- `0`: the site is not labeled high-risk in the next monitoring period.
- `1`: the site is labeled high-risk in the next monitoring period.

The model produces a probability that is used to prioritize review in the
demonstration dashboard. It does not prescribe an action or replace expert
judgment.

## Intended Use

The model is intended for:

- Demonstrating an end-to-end clinical-trial risk analytics workflow.
- Exploring site-level operational and data-quality signals.
- Ranking synthetic sites for analyst investigation.
- Demonstrating model evaluation, temporal validation, threshold selection,
	and local explanation techniques.
- Serving as a portfolio artifact for Data Science and AI Engineering work.

## Prohibited Uses

The model must not be used to:

- Make or automate patient-care, medical, or clinical decisions.
- Determine trial eligibility, treatment, diagnosis, or patient safety.
- Replace clinical, statistical, data-management, or regulatory review.
- Make decisions about investigators, sites, participants, or vendors in a
	real trial.
- Support regulatory submissions or claims of clinical effectiveness.
- Treat a high-risk prediction as proof of misconduct, poor performance, or
	data integrity failure.
- Be deployed on real-world data without a new validation, governance review,
	and domain-appropriate authorization.

## Data

### Source and Scope

The model was developed with generated synthetic clinical-trial data. The
synthetic environment represents trial sites observed over repeated monitoring
periods and includes 250 unique site IDs in the machine-learning dataset.
Because the data is simulated, its distributions, relationships, event rates,
and labels should not be assumed to represent any clinical-trial population.

The major data domains represented by the project are:

- Trial and site metadata.
- Patient enrollment and visits.
- Laboratory results.
- Adverse events, including serious and severe events.
- Operational events and data-quality activity.
- Derived site-level monitoring-period features.

The target is generated from the synthetic risk process and shifted to the
next monitoring period. Current-period inputs are used to predict future risk;
the final risk score and target label are not model features.

### Model Features

The tuned model uses 18 numeric current-period features:

- `missing_data_count`
- `protocol_deviation_count`
- `data_entry_delay_avg`
- `query_resolution_avg`
- `total_patients`
- `dropped_out_patients`
- `patient_dropout_rate`
- `total_visits`
- `completed_visits`
- `visit_completion_rate`
- `average_visit_delay_days`
- `total_adverse_events`
- `serious_adverse_events`
- `severe_adverse_events`
- `average_reported_delay_days`
- `serious_event_rate`
- `severe_event_rate`
- `enrollment_deterioration`

Site identifier, monitoring period, the target, behavior labels, and the
temporary composite `risk_score` are not used as model inputs.

## Model and Decision Policy

The model is a tuned `RandomForestClassifier` persisted as
`models/tuned_random_forest.joblib`. Random Forest was selected for its ability
to model nonlinear relationships among heterogeneous operational signals and
to provide probability estimates and feature-level explanation inputs.

The operational risk threshold is **0.40**. A predicted probability at or
above 0.40 is treated as `High Risk` by the demonstration API; lower values
are treated as `Low Risk`. The threshold is a prioritization policy, not a
clinical or regulatory cutoff. It was selected to improve the balance between
precision and recall for this synthetic use case.

## Evaluation

The project evaluates the model through complementary views:

1. **Stratified cross-validation** compares candidate model families while
	 preserving the class distribution in each fold.
2. **Time-based validation** evaluates performance on later monitoring periods
	 to probe temporal generalization and reduce reliance on randomly mixed
	 future observations.
3. **Final test evaluation** reports classification metrics on a held-out test
	 set. The selected operating threshold is 0.40.

### Final Test Metrics at Threshold 0.40

| Metric | Result |
|---|---:|
| Accuracy | 0.744 |
| Precision | 0.622 |
| Recall | 0.647 |
| F1 | 0.634 |
| ROC-AUC | 0.766 |

These results describe performance on the project’s synthetic evaluation data
only. They are not estimates of performance on real clinical-trial data.

## Explainability and Anomaly Detection

### SHAP Explanations

The API uses SHAP with a tree-based explainer to describe individual site
predictions. For a site’s latest monitoring record, the response reports the
largest positive contributors in `risk_increasing_factors` and the largest
negative contributors in `risk_reducing_factors`. Each factor contains:

- `feature`
- `feature_value`
- `shap_value`

SHAP values explain the model output; they do not establish causality and
should be interpreted alongside the underlying data and human review.

### Isolation Forest Anomaly Detection

TrialGuard AI also uses an Isolation Forest workflow to identify unusual
site-period patterns. An anomaly is a statistical signal that a record differs
from patterns learned in the synthetic data. It is not evidence of fraud,
misconduct, a protocol violation, or a clinical safety issue.

## Human Oversight and Responsible AI

Any meaningful use of a risk signal requires qualified human review. Reviewers
should examine source records, data quality, site context, monitoring history,
and alternative explanations before taking action. Predictions and anomaly
flags should be treated as triage signals, with documented decisions and an
ability to override or reject the model output.

Responsible use requires checking for data-quality problems, subgroup or site
type disparities, temporal drift, calibration changes, and feedback loops.
Users should avoid presenting model scores with unwarranted certainty and
should communicate the synthetic and experimental nature of this system.

## Limitations and Risks

- The training and evaluation data is synthetic and may not reflect real
	clinical-trial processes, populations, missingness, or operational behavior.
- There is no real-world clinical validation, prospective validation, or
	evidence of clinical utility.
- Synthetic labels may encode assumptions from the data-generation process,
	so strong test performance may not transfer to another environment.
- Temporal validation reduces but does not eliminate the risk of leakage or
	distribution shift.
- Probability scores are not presented as calibrated clinical probabilities.
- A site-level prediction can hide important patient-level and trial-level
	context.
- Feature relationships are predictive associations, not causal findings.
- Anomaly detection is sensitive to the reference distribution and can flag
	legitimate operational changes.
- The model does not account for every factor that may affect site performance
	and may be affected by missing, delayed, or incorrectly recorded inputs.
- The 0.40 threshold reflects this demonstration’s tradeoff and should not be
	reused without a new decision analysis.

## Reproducibility and Testing

The repository contains scripts for synthetic data generation, feature
construction, target creation, model training, cross-validation, threshold
analysis, time-based validation, final evaluation, anomaly detection, and SHAP
reporting. Persisted model artifacts are stored under `models/`, and evaluation
outputs are stored under `outputs/reports/`.

Automated tests cover:

- API response contracts and endpoint behavior.
- Required processed datasets and schema fields.
- Missing-value checks for model features and target.
- Presence of both target classes.
- Presence of persisted model artifacts.
- Tuned-model probability generation and probability bounds.
- Anomaly, site-intelligence, and site-trend category values.
- Expected dataset site coverage.

Results should be reproduced in the project’s configured Python environment
using the repository’s dependency manifest. Any change to the data-generation
process, feature definitions, target construction, model artifact, or threshold
requires rerunning the validation and tests before the results are considered
comparable.

## Version and Status

This model card describes the tuned Random Forest artifact in the TrialGuard AI
technical demonstration repository as of September 2026. The model is
experimental, portfolio-oriented, and not approved for production or clinical
use.
