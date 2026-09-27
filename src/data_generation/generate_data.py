import pandas as pd
import numpy as np
import pathlib as path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -----------------------------
# Generate Trials
# -----------------------------

trials = pd.DataFrame([
    {
        "trial_id": "TRIAL001",
        "trial_name": "CardioNova",
        "therapeutic_area": "Cardiology",
        "phase": "Phase III",
        "start_date": "2025-01-15",
        "end_date": "2027-06-30",
        "target_enrollment": 5000
    },
    {
        "trial_id": "TRIAL002",
        "trial_name": "OncoAdvance",
        "therapeutic_area": "Oncology",
        "phase": "Phase III",
        "start_date": "2025-03-01",
        "end_date": "2027-12-31",
        "target_enrollment": 4000
    },
    {
        "trial_id": "TRIAL003",
        "trial_name": "ImmunoCare",
        "therapeutic_area": "Immunology",
        "phase": "Phase II",
        "start_date": "2025-05-10",
        "end_date": "2027-05-31",
        "target_enrollment": 2500
    },
    {
        "trial_id": "TRIAL004",
        "trial_name": "NeuroBalance",
        "therapeutic_area": "Neurology",
        "phase": "Phase II",
        "start_date": "2025-07-01",
        "end_date": "2027-08-31",
        "target_enrollment": 2200
    },
    {
        "trial_id": "TRIAL005",
        "trial_name": "RareGen",
        "therapeutic_area": "Rare Disease",
        "phase": "Phase III",
        "start_date": "2025-09-15",
        "end_date": "2028-03-31",
        "target_enrollment": 1800
    }
])
print(trials)
# -----------------------------
# Generate Sites
# -----------------------------

countries = [
    "United States",
    "Germany",
    "India",
    "United Kingdom",
    "Canada",
    "Japan",
    "Australia"
]

cities = {
    "United States": ["Boston", "Chicago", "New York"],
    "Germany": ["Berlin", "Munich", "Hamburg"],
    "India": ["Hyderabad", "Mumbai", "Bangalore"],
    "United Kingdom": ["London", "Manchester", "Birmingham"],
    "Canada": ["Toronto", "Vancouver", "Montreal"],
    "Japan": ["Tokyo", "Osaka", "Kyoto"],
    "Australia": ["Sydney", "Melbourne", "Brisbane"]
}

sites = []

site_counter = 1

for _, trial in trials.iterrows():

    for _ in range(50):

        country = np.random.choice(countries)
        city = np.random.choice(cities[country])

        sites.append({
            "site_id": f"SITE{site_counter:04d}",
            "trial_id": trial["trial_id"],
            "country": country,
            "city": city,
            "site_type": np.random.choice(
                ["Academic", "Hospital", "Research Center"]
            ),
            "investigator_experience": np.random.randint(2, 21),
            "target_patients": np.random.randint(80, 151)
        })

        site_counter += 1

sites = pd.DataFrame(sites)
# -----------------------------
# Assign Site Behavior
# -----------------------------

site_behaviors = np.random.choice(
    [
        "Stable",
        "Deteriorating",
        "High Risk",
        "Anomalous"
    ],
    size=len(sites),
    p=[0.60, 0.20, 0.15, 0.05]
)

sites["behavior"] = site_behaviors

print("\nSite Behavior Distribution:")
print(sites["behavior"].value_counts())

print("\nSites:")
print(sites.head())
print("\nTotal sites:", len(sites))
# -----------------------------
# Generate Patients
# -----------------------------

patients = []

patient_counter = 1

for _, site in sites.iterrows():

    # Number of patients assigned to this site
    num_patients = np.random.randint(80, 151)

    for _ in range(num_patients):

        enrollment_date = pd.Timestamp(
            np.random.choice(
                pd.date_range(
                    start="2025-01-01",
                    end="2026-06-30"
                )
            )
        )

        patients.append({
            "patient_id": f"PATIENT{patient_counter:06d}",
            "site_id": site["site_id"],
            "enrollment_date": enrollment_date,
            "age": np.random.randint(18, 76),
            "sex": np.random.choice(["Male", "Female"]),
            "treatment_group": np.random.choice(
                ["Treatment", "Placebo"]
            ),
            "status": np.random.choice(
                ["Active", "Completed", "Dropped Out"],
                p=[0.70, 0.20, 0.10]
            )
        })

        patient_counter += 1

patients = pd.DataFrame(patients)

print("\nPatients:")
print(patients.head())

print("\nTotal patients:", len(patients))
# -----------------------------
# Generate Visits
# -----------------------------

visit_types = [
    "Screening",
    "Baseline",
    "Week 4",
    "Week 8",
    "Week 12",
    "Week 16"
]

visits = []

visit_counter = 1

for _, patient in patients.iterrows():

    for visit_type in visit_types:

        scheduled_date = patient["enrollment_date"] + pd.Timedelta(
            days=visit_types.index(visit_type) * 28
        )

        # Randomly determine whether the visit was completed
        completed = np.random.choice(
            [True, False],
            p=[0.92, 0.08]
        )

        if completed:
            visit_delay_days = max(
                0,
                int(np.random.normal(loc=2, scale=3))
            )

            actual_date = (
                scheduled_date
                + pd.Timedelta(days=visit_delay_days)
            )

        else:
            visit_delay_days = np.nan
            actual_date = pd.NaT

        visits.append({
            "visit_id": f"VISIT{visit_counter:07d}",
            "patient_id": patient["patient_id"],
            "visit_type": visit_type,
            "scheduled_date": scheduled_date,
            "actual_date": actual_date,
            "visit_delay_days": visit_delay_days,
            "completed": completed
        })

        visit_counter += 1

visits = pd.DataFrame(visits)

print("\nVisits:")
print(visits.head(10))

print("\nTotal visits:", len(visits))
# -----------------------------
# Generate Lab Results
# -----------------------------

lab_tests = {
    "Hemoglobin": (10, 18),
    "White Blood Cell Count": (4, 11),
    "Platelets": (150, 450),
    "ALT": (7, 56),
    "AST": (10, 40),
    "Creatinine": (0.6, 1.3)
}

lab_results = []

lab_counter = 1

completed_visits = visits[visits["completed"] == True]

for _, visit in completed_visits.iterrows():

    # Each completed visit has a few laboratory tests
    selected_tests = np.random.choice(
        list(lab_tests.keys()),
        size=3,
        replace=False
    )

    for test_name in selected_tests:

        reference_low, reference_high = lab_tests[test_name]

        # Generate a realistic value around the reference range
        mean = (reference_low + reference_high) / 2
        std = (reference_high - reference_low) / 6

        result = np.random.normal(mean, std)

        abnormal_flag = (
            result < reference_low
            or result > reference_high
        )

        lab_results.append({
            "lab_id": f"LAB{lab_counter:07d}",
            "patient_id": visit["patient_id"],
            "visit_id": visit["visit_id"],
            "test_name": test_name,
            "result": round(result, 2),
            "reference_low": reference_low,
            "reference_high": reference_high,
            "abnormal_flag": abnormal_flag
        })

        lab_counter += 1

lab_results = pd.DataFrame(lab_results)

print("\nLab Results:")
print(lab_results.head(10))

print("\nTotal lab results:", len(lab_results))
# -----------------------------
# Generate Adverse Events
# -----------------------------

event_types = [
    "Headache",
    "Fatigue",
    "Nausea",
    "Dizziness",
    "Injection Site Reaction",
    "Fever",
    "Abdominal Pain"
]

severity_levels = [
    "Mild",
    "Moderate",
    "Severe"
]

adverse_events = []

ae_counter = 1

for _, patient in patients.iterrows():

    # Not every patient experiences an adverse event
    num_events = np.random.poisson(lam=1.2)

    for _ in range(num_events):

        event_date = patient["enrollment_date"] + pd.Timedelta(
            days=np.random.randint(1, 365)
        )

        severity = np.random.choice(
            severity_levels,
            p=[0.65, 0.30, 0.05]
        )

        serious = severity == "Severe"

        reported_delay_days = max(
            0,
            int(np.random.normal(loc=2, scale=2))
        )

        # Find patient's site
        patient_site = patients.loc[
            patients["patient_id"] == patient["patient_id"],
            "site_id"
        ].iloc[0]

        adverse_events.append({
            "event_id": f"AE{ae_counter:07d}",
            "patient_id": patient["patient_id"],
            "site_id": patient_site,
            "event_date": event_date,
            "event_type": np.random.choice(event_types),
            "severity": severity,
            "serious": serious,
            "reported_delay_days": reported_delay_days
        })

        ae_counter += 1

adverse_events = pd.DataFrame(adverse_events)

print("\nAdverse Events:")
print(adverse_events.head(10))

print("\nTotal adverse events:", len(adverse_events))

# -----------------------------
# Generate Operational Events
# -----------------------------

operational_event_types = [
    "Missing Data",
    "Protocol Deviation",
    "Data Entry Delay",
    "Query Resolution"
]

operational_events = []

operation_counter = 1
monitoring_start = pd.Timestamp("2025-01-01")
monitoring_end = pd.Timestamp("2026-08-31")
monitoring_period_edges = pd.date_range(
    start=monitoring_start,
    end=monitoring_end + pd.Timedelta(days=1),
    periods=13
)

for _, site in sites.iterrows():

    behavior = site["behavior"]

    if behavior == "Stable":
        event_rates = [5] * 12
    elif behavior == "Deteriorating":
        event_rates = np.linspace(3, 18, 12)
    elif behavior == "High Risk":
        event_rates = [24] * 12
    else:  # Anomalous
        event_rates = [11] * 12
        spike_period = np.random.randint(4, 11)
        event_rates[spike_period - 1] = 45

    for period in range(1, 13):
        period_start = monitoring_period_edges[period - 1]
        period_end = monitoring_period_edges[period]
        period_dates = pd.date_range(
            start=period_start,
            end=period_end - pd.Timedelta(days=1)
        )
        num_events = np.random.poisson(lam=event_rates[period - 1])

        for _ in range(num_events):
            event_date = pd.Timestamp(np.random.choice(period_dates))

            event_type = np.random.choice(
                operational_event_types,
                p=[0.35, 0.20, 0.25, 0.20]
            )

            if event_type == "Missing Data":
                value = np.random.randint(1, 6)

            elif event_type == "Protocol Deviation":
                value = 1

            elif event_type == "Data Entry Delay":
                value = np.random.randint(1, 15)

            else:
                value = np.random.randint(1, 21)

            operational_events.append({
                "event_id": f"OP{operation_counter:07d}",
                "site_id": site["site_id"],
                "event_date": event_date,
                "monitoring_period": period,
                "event_type": event_type,
                "value": value
            })

            operation_counter += 1

operational_events = pd.DataFrame(operational_events)

print("\nOperational Events:")
print(operational_events.head(10))

print("\nTotal operational events:", len(operational_events))

# -----------------------------
# Verify Operational Event Patterns
# -----------------------------

site_event_counts = (
    operational_events.groupby("site_id")
    .size()
    .rename("event_count")
    .reset_index()
)

site_event_summary = sites[["site_id", "behavior"]].merge(
    site_event_counts,
    on="site_id",
    how="left"
)
site_event_summary["event_count"] = (
    site_event_summary["event_count"].fillna(0).astype(int)
)

print("\nVerification: Average events per site by behavior:")
print(
    site_event_summary.groupby("behavior")["event_count"]
    .mean()
    .sort_index()
)

highest_event_site = site_event_summary.loc[
    site_event_summary["event_count"].idxmax()
]
print("\nVerification: Site with the highest number of events:")
print(highest_event_site)

anomalous_sites = site_event_summary.loc[
    site_event_summary["behavior"] == "Anomalous", "site_id"
].head(5)
anomalous_period_counts = (
    operational_events[
        operational_events["site_id"].isin(anomalous_sites)
    ]
    .groupby(["site_id", "monitoring_period"])
    .size()
    .unstack(fill_value=0)
    .reindex(columns=range(1, 13), fill_value=0)
)

print("\nVerification: Anomalous site event counts by monitoring period:")
print(anomalous_period_counts)

# -----------------------------
# Save Generated Data
# -----------------------------

synthetic_data_dir = path.Path("data/synthetic")
synthetic_data_dir.mkdir(parents=True, exist_ok=True)

dataframes_to_save = {
    "trials": trials,
    "sites": sites,
    "patients": patients,
    "visits": visits,
    "lab_results": lab_results,
    "adverse_events": adverse_events,
    "operational_events": operational_events
}

for dataframe_name, dataframe in dataframes_to_save.items():
    dataframe.to_csv(
        synthetic_data_dir / f"{dataframe_name}.csv",
        index=False
    )