from typing import Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st


API_BASE_URL = "https://trialguard-ai.onrender.com"
REQUEST_TIMEOUT_SECONDS = 10


st.set_page_config(
    page_title="TrialGuard AI",
    page_icon="TG",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #17202a;
        --muted: #687582;
        --line: #dfe7e8;
        --surface: #ffffff;
        --canvas: #f4f7f6;
        --teal: #087f8c;
        --teal-dark: #07515d;
        --coral: #dc6b52;
        --amber: #d49a36;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: var(--ink);
    }

    .hero {
        background: linear-gradient(120deg, #073c48 0%, #087f8c 66%, #4da6a1 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 28px rgba(7, 81, 93, 0.16);
    }

    .hero h1 {
        color: white;
        font-size: 2.6rem;
        margin: 0;
        letter-spacing: 0;
    }

    .hero p {
        color: #d9f0ef;
        font-size: 1.05rem;
        margin: 0.35rem 0 0;
    }

    .kpi-card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-top: 4px solid var(--teal);
        border-radius: 8px;
        padding: 1rem 1.1rem;
        min-height: 108px;
        box-shadow: 0 4px 14px rgba(23, 32, 42, 0.04);
    }

    .kpi-label {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .kpi-value {
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.45rem;
    }

    .section-heading {
        border-bottom: 1px solid var(--line);
        margin: 2rem 0 1rem;
        padding-bottom: 0.55rem;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin: 1rem 0;
    }

    .metric-item {
        background: #eef5f4;
        border-left: 3px solid var(--teal);
        padding: 0.75rem 0.9rem;
    }

    .metric-item span {
        color: var(--muted);
        display: block;
        font-size: 0.76rem;
        margin-bottom: 0.25rem;
    }

    .metric-item strong {
        color: var(--ink);
        font-size: 1.12rem;
    }

    @media (max-width: 800px) {
        .metric-grid { grid-template-columns: repeat(2, 1fr); }
        .hero { padding: 1.6rem; }
        .hero h1 { font-size: 2.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


class APIUnavailableError(Exception):
    """Raised when the FastAPI backend cannot provide a response."""


def api_get(endpoint: str) -> Any:
    """Fetch and return JSON from the FastAPI backend."""
    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        raise APIUnavailableError(
            "TrialGuard AI backend is unavailable. Start it with "
            "`uvicorn src.api.main:app --reload` and refresh this page."
        ) from error
    except ValueError as error:
        raise APIUnavailableError(
            f"The backend returned an invalid response for {endpoint}."
        ) from error


@st.cache_data(ttl=60, show_spinner=False)
def fetch_summary() -> dict[str, Any]:
    return api_get("/summary")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_sites() -> list[dict[str, Any]]:
    return api_get("/sites")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_critical() -> list[dict[str, Any]]:
    return api_get("/critical")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_risk() -> list[dict[str, Any]]:
    return api_get("/risk")


def fetch_site(site_id: str) -> dict[str, Any]:
    return api_get(f"/sites/{site_id}")


def fetch_site_explanation(site_id: str) -> dict[str, Any]:
    """Fetch the selected site's SHAP explanation from FastAPI."""
    return api_get(f"/sites/{site_id}/explanation")


def render_kpis(summary: dict[str, Any]) -> None:
    """Render summary values as a responsive KPI grid."""
    metrics = [
        ("Total Sites", summary["total_sites"]),
        ("Site-Period Records", summary["total_site_period_records"]),
        ("High-Risk Records", summary["high_risk_records"]),
        ("Critical Records", summary["critical_records"]),
        ("Anomalous Records", summary["anomaly_records"]),
        ("Deteriorating Sites", summary["deteriorating_sites"]),
    ]
    cards = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value:,}</div></div>'
        for label, value in metrics
    )
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)


def render_risk_overview(risk_records: list[dict[str, Any]]) -> None:
    """Render the distribution of high-risk probabilities."""
    risk_data = pd.DataFrame(risk_records)
    st.markdown('<h2 class="section-heading">Risk Overview</h2>', unsafe_allow_html=True)
    if risk_data.empty:
        st.info("No high-risk records are currently available.")
        return

    chart = px.histogram(
        risk_data,
        x="risk_probability",
        nbins=15,
        labels={"risk_probability": "Risk Probability", "count": "Records"},
        color_discrete_sequence=["#087f8c"],
    )
    chart.update_layout(
        height=350,
        bargap=0.08,
        margin=dict(l=10, r=10, t=20, b=10),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    chart.update_xaxes(range=[0, 1], gridcolor="#e5eded")
    chart.update_yaxes(gridcolor="#e5eded")
    st.plotly_chart(chart, use_container_width=True)


def render_priority_overview(critical_records: list[dict[str, Any]]) -> None:
    """Render filtered critical site-period records for operations teams."""
    st.markdown(
        '<h2 class="section-heading">Critical Site Monitoring</h2>',
        unsafe_allow_html=True,
    )
    priority_data = pd.DataFrame(critical_records)
    columns = [
        "site_id",
        "monitoring_period",
        "risk_probability",
        "risk_status",
        "anomaly_status",
        "priority",
        "missing_data_count",
        "protocol_deviation_count",
        "data_entry_delay_avg",
        "query_resolution_avg",
        "patient_dropout_rate",
        "average_visit_delay_days",
        "total_adverse_events",
    ]
    if priority_data.empty:
        st.info("No critical-site records are currently available.")
        return

    filter_columns = st.columns(2)
    with filter_columns[0]:
        selected_priority = st.selectbox(
            "Priority",
            ["All", "Critical", "High"],
            key="critical_priority_filter",
        )
    with filter_columns[1]:
        minimum_risk = st.slider(
            "Minimum risk probability",
            min_value=0.40,
            max_value=1.00,
            value=0.40,
            step=0.01,
            key="critical_minimum_risk",
        )

    filtered_data = priority_data[
        priority_data["risk_probability"].ge(minimum_risk)
    ]
    if selected_priority != "All":
        filtered_data = filtered_data[
            filtered_data["priority"].eq(selected_priority)
        ]

    priority_order = {"Critical": 0, "High": 1}
    filtered_data = filtered_data.assign(
        priority_order=filtered_data["priority"].map(priority_order).fillna(2)
    ).sort_values(
        ["priority_order", "risk_probability"],
        ascending=[True, False],
    )
    filtered_data = filtered_data.drop(columns="priority_order")[columns]

    st.markdown(f"**Showing {len(filtered_data)} records**")
    st.dataframe(
        filtered_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_probability": st.column_config.NumberColumn(
                "Risk Probability", format="%.3f"
            ),
            "monitoring_period": "Period",
            "data_entry_delay_avg": st.column_config.NumberColumn(
                "Data Entry Delay Avg", format="%.2f"
            ),
            "query_resolution_avg": st.column_config.NumberColumn(
                "Query Resolution Avg", format="%.2f"
            ),
            "patient_dropout_rate": st.column_config.NumberColumn(
                "Patient Dropout Rate", format="%.3f"
            ),
            "average_visit_delay_days": st.column_config.NumberColumn(
                "Average Visit Delay Days", format="%.2f"
            ),
        },
    )
    st.download_button(
        "Download filtered critical-site table",
        data=filtered_data.to_csv(index=False).encode("utf-8"),
        file_name="critical_site_monitoring.csv",
        mime="text/csv",
    )


def render_site_detail(
    site_detail: dict[str, Any],
    site_explanation: dict[str, Any],
) -> None:
    """Render selected-site metrics, explanation, and risk trend."""
    trend = site_detail["trend"]
    history = pd.DataFrame(site_detail["records"])

    st.markdown(
        '<h2 class="section-heading">Site Investigation</h2>',
        unsafe_allow_html=True,
    )
    metric_values = [
        ("First Risk Probability", f"{trend['first_risk_probability']:.3f}"),
        ("Latest Risk Probability", f"{trend['latest_risk_probability']:.3f}"),
        ("Risk Probability Change", f"{trend['risk_probability_change']:+.3f}"),
        ("Average Risk Probability", f"{trend['average_risk_probability']:.3f}"),
        ("Maximum Risk Probability", f"{trend['maximum_risk_probability']:.3f}"),
        ("High-Risk Periods", trend["high_risk_periods"]),
        ("Anomaly Periods", trend["anomaly_periods"]),
        ("Critical Periods", trend["critical_periods"]),
        ("Trend Status", trend["trend_status"]),
        ("Priority Status", trend["priority_status"]),
    ]
    metric_markup = "".join(
        f'<div class="metric-item"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in metric_values
    )
    st.markdown(
        f'<div class="metric-grid">{metric_markup}</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<h3>Model Explanation</h3>', unsafe_allow_html=True)
    explanation_columns = st.columns(2)
    with explanation_columns[0]:
        st.metric(
            "Risk Probability",
            f"{site_explanation['risk_probability']:.3f}",
        )
    with explanation_columns[1]:
        st.metric("Risk Status", site_explanation["risk_status"])

    increasing_factors = site_explanation.get(
        "risk_increasing_factors",
        site_explanation.get("increasing_factors", []),
    )
    reducing_factors = site_explanation.get(
        "risk_reducing_factors",
        site_explanation.get("reducing_factors", []),
    )
    factor_columns = {
        "feature": "Feature",
        "feature_value": "Feature Value",
        "shap_value": "SHAP Contribution",
    }
    factor_table_columns = [
        "feature",
        "feature_value",
        "shap_value",
    ]

    explanation_sections = st.columns(2)
    with explanation_sections[0]:
        st.markdown("**Factors Increasing Risk**")
        st.dataframe(
            pd.DataFrame(increasing_factors).reindex(
                columns=factor_table_columns
            ).rename(columns=factor_columns),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature Value": st.column_config.NumberColumn(format="%.3f"),
                "SHAP Contribution": st.column_config.NumberColumn(
                    format="%+.4f"
                ),
            },
        )
    with explanation_sections[1]:
        st.markdown("**Factors Reducing Risk**")
        st.dataframe(
            pd.DataFrame(reducing_factors).reindex(
                columns=factor_table_columns
            ).rename(columns=factor_columns),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Feature Value": st.column_config.NumberColumn(format="%.3f"),
                "SHAP Contribution": st.column_config.NumberColumn(
                    format="%+.4f"
                ),
            },
        )

    st.info(
        "SHAP values explain how individual features contributed to this site's "
        "model prediction. Positive values increase predicted risk; negative "
        "values decrease predicted risk."
    )

    st.markdown('<h3>Site Risk Trend</h3>', unsafe_allow_html=True)
    if history.empty:
        st.info("No monitoring-period history is available for this site.")
        return

    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=history["monitoring_period"],
            y=history["risk_probability"],
            mode="lines+markers",
            name="Risk Probability",
            line=dict(color="#087f8c", width=3),
            marker=dict(color="#dc6b52", size=8),
        )
    )
    chart.add_hline(
        y=0.40,
        line_dash="dash",
        line_color="#dc6b52",
        annotation_text="High Risk Threshold",
        annotation_position="top left",
    )
    chart.update_layout(
        height=380,
        title="Risk Probability Over Monitoring Periods",
        xaxis_title="Monitoring Period",
        yaxis_title="Risk Probability",
        yaxis=dict(range=[0, 1]),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    chart.update_xaxes(dtick=1, gridcolor="#e5eded")
    chart.update_yaxes(gridcolor="#e5eded")
    st.plotly_chart(chart, use_container_width=True)


def render_dashboard() -> None:
    """Fetch API data and render the dashboard sections."""
    st.markdown(
        '<div class="hero"><h1>TrialGuard AI</h1>'
        '<p>Clinical Trial Risk Intelligence &amp; Monitoring Platform</p></div>',
        unsafe_allow_html=True,
    )

    try:
        summary = fetch_summary()
        sites = fetch_sites()
        risk_records = fetch_risk()
        critical_records = fetch_critical()
    except APIUnavailableError as error:
        st.error(str(error))
        st.stop()

    render_kpis(summary)
    render_risk_overview(risk_records)
    render_priority_overview(critical_records)

    st.markdown('<h2 class="section-heading">Site Investigation</h2>', unsafe_allow_html=True)
    site_ids = [site["site_id"] for site in sites]
    if not site_ids:
        st.info("No sites are currently available from the backend.")
        return

    selected_site = st.selectbox("Select a site", site_ids)
    try:
        site_detail = fetch_site(selected_site)
        site_explanation = fetch_site_explanation(selected_site)
    except APIUnavailableError as error:
        st.error(str(error))
        return
    render_site_detail(site_detail, site_explanation)


if __name__ == "__main__":
    render_dashboard()
