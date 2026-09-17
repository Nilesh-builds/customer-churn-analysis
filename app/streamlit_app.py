"""A transparent decision-support dashboard for the churn analysis."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from churn_analysis.data import clean_telco_data, load_telco_data, profile_telco_data
from churn_analysis.modeling import (
    cross_validate_models,
    fit_full_model,
    holdout_predictions,
    threshold_report,
    train_models,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
REPORT_PATH = ROOT / "outputs" / "metrics" / "model_metrics.json"

st.set_page_config(page_title="Customer Churn Decision Support", layout="wide")

st.markdown(
    """
    <style>
    .block-container { max-width: 1180px; padding-top: 2.5rem; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    [data-testid="stMetricLabel"] { color: #8b95a7; }
    .context-card {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 0.75rem;
        padding: 1rem 1.15rem;
        margin: 0.5rem 0 1.25rem;
        background: rgba(128, 128, 128, 0.06);
    }
    .context-card strong { color: #7dd3fc; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_clean_data() -> pd.DataFrame:
    return clean_telco_data(load_telco_data(DATA_PATH))


@st.cache_data
def load_customer_ids() -> pd.Series:
    return load_telco_data(DATA_PATH)["customerID"]


@st.cache_data
def get_profile() -> dict:
    return profile_telco_data(load_telco_data(DATA_PATH))


@st.cache_data
def get_cross_validation() -> dict:
    if REPORT_PATH.exists():
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        return report["cross_validation"]
    # Community Cloud has a small memory budget, so the dashboard uses three
    # sequential folds. The full five-fold report remains in the CLI pipeline.
    return cross_validate_models(load_clean_data(), folds=3, n_jobs=1)


@st.cache_data
def get_holdout_metrics() -> dict:
    return train_models(load_clean_data())[1]


@st.cache_resource
def get_full_model():
    return fit_full_model(load_clean_data())


@st.cache_data
def get_holdout_data():
    _, features, target, probabilities = holdout_predictions(load_clean_data())
    return features, target, probabilities


st.title("Customer Churn Decision Support")
st.caption(
    "A transparent portfolio project: predictions are evidence for a retention "
    "conversation, not automatic customer decisions."
)

with st.sidebar:
    st.markdown("### About this demo")
    st.write(
        "I built this dashboard to show how a churn model can support a human "
        "retention review instead of making an automatic customer decision."
    )
    st.link_button("View source on GitHub", "https://github.com/Nilesh-builds/customer-churn-analysis")
    st.markdown("**Workflow**")
    st.caption("Raw data -> quality checks -> model -> cost scenario -> review list")
    st.markdown("**Important**")
    st.caption("The dataset is a static sample. Cost values are adjustable scenarios, not company facts.")

frame = load_clean_data()
profile = get_profile()
features, target, probabilities = get_holdout_data()

metric_one, metric_two, metric_three, metric_four = st.columns(4)
metric_one.metric("Customers", f"{len(frame):,}")
metric_two.metric("Observed churn", f"{frame['Churn'].mean():.1%}")
metric_three.metric("Data quality", profile["status"].title())
metric_four.metric("Holdout rows", f"{len(target):,}")

tab_overview, tab_model, tab_customers, tab_data = st.tabs(
    ["Overview", "Model quality", "Customer risk", "Data quality"]
)

with tab_overview:
    st.subheader("Where churn is concentrated")
    contract_summary = (
        frame.groupby("Contract", as_index=False)
        .agg(customers=("Churn", "size"), churn_rate=("Churn", "mean"))
    )
    contract_summary["churn_rate_percent"] = contract_summary.pop("churn_rate") * 100
    st.bar_chart(contract_summary.set_index("Contract")["churn_rate_percent"])
    display_summary = contract_summary.rename(
        columns={
            "customers": "Customers",
            "churn_rate_percent": "Churn rate (%)",
        }
    )
    st.dataframe(
        display_summary.style.format({"Churn rate (%)": "{:.1f}%"}),
        use_container_width=True,
        hide_index=False,
    )
    st.markdown(
        '<div class="context-card"><strong>Reading this:</strong> '
        "Month-to-month customers are a priority segment for investigation, "
        "not proof that contract type alone causes churn.</div>",
        unsafe_allow_html=True,
    )
    st.info(
        "The contract pattern is an association in this sample. It tells us "
        "where to investigate and test retention ideas; it does not prove why "
        "a customer leaves."
    )

with tab_model:
    st.subheader("Cross-validated model quality")
    try:
        cv_frame = pd.DataFrame(get_cross_validation()).T
        cv_frame.index = cv_frame.index.map(
            {
                "logistic_regression": "Logistic regression",
                "random_forest": "Random forest",
                "balanced_random_forest": "Balanced random forest",
            }
        )
        st.dataframe(
            cv_frame.style.format({column: "{:.1%}" for column in cv_frame.columns}),
            use_container_width=True,
        )
    except Exception:
        # The hosted runtime can have stricter limits than local Python. The
        # dashboard should remain useful while the full report stays in CI.
        st.warning(
            "Cross-validation is unavailable in this hosted session. "
            "Showing the verified holdout comparison instead."
        )
        holdout_frame = pd.DataFrame(get_holdout_metrics()).T
        st.dataframe(
            holdout_frame.style.format({column: "{:.1%}" for column in holdout_frame.columns}),
            use_container_width=True,
        )

    st.subheader("Retention threshold scenarios")
    contact_cost = st.number_input("Contact cost", min_value=0.0, value=5.0, step=1.0)
    missed_cost = st.number_input(
        "Scenario cost of a missed churner", min_value=0.0, value=100.0, step=10.0
    )
    scenarios = pd.DataFrame(
        threshold_report(
            probabilities,
            target,
            contact_cost=contact_cost,
            missed_churn_cost=missed_cost,
        )
    )
    best = scenarios.loc[scenarios["scenario_cost"].idxmin()]
    st.success(
        f"Lowest-cost scenario under these assumptions: contact customers at "
        f"probability >= {best['threshold']:.2f} ({int(best['contacts'])} contacts)."
    )
    st.bar_chart(scenarios.set_index("threshold")["scenario_cost"])
    scenario_display = scenarios.rename(
        columns={
            "threshold": "Risk threshold",
            "contacts": "Contacts",
            "precision": "Precision",
            "recall": "Recall",
            "scenario_cost": "Scenario cost",
        }
    )[["Risk threshold", "Contacts", "Precision", "Recall", "Scenario cost"]]
    st.dataframe(
        scenario_display.style.format(
            {
                "Risk threshold": "{:.0%}",
                "Precision": "{:.1%}",
                "Recall": "{:.1%}",
                "Scenario cost": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("These costs are scenario inputs, not measured company costs.")

with tab_customers:
    st.subheader("Customers to review")
    st.caption("Use the threshold to decide how broad the human review queue should be.")
    threshold = st.slider("Risk threshold", 0.05, 0.95, 0.50, 0.05)
    model = get_full_model()
    customer_features = frame.drop(columns=["Churn"])
    scores = model.predict_proba(customer_features)[:, 1]
    risk = frame.copy()
    risk.insert(0, "customerID", load_customer_ids().to_numpy())
    risk["churn_probability"] = scores
    risk["review_recommended"] = risk["churn_probability"] >= threshold
    risk["risk_band"] = pd.cut(
        risk["churn_probability"],
        bins=[-0.01, 0.33, 0.66, 1.0],
        labels=["Lower", "Medium", "Higher"],
    )
    risk = risk.sort_values("churn_probability", ascending=False)
    review_count = int(risk["review_recommended"].sum())
    risk_metric, top_risk_metric = st.columns(2)
    risk_metric.metric("Review queue", f"{review_count:,} customers")
    top_risk_metric.metric("Highest predicted risk", f"{risk['churn_probability'].max():.1%}")
    display_columns = [
        "customerID",
        "tenure",
        "Contract",
        "InternetService",
        "MonthlyCharges",
        "churn_probability",
        "risk_band",
        "review_recommended",
    ]
    risk_display = risk[display_columns].head(100)
    st.dataframe(
        risk_display.style.format(
            {
                "MonthlyCharges": "${:,.2f}",
                "churn_probability": "{:.1%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download risk review list",
        risk[display_columns].to_csv(index=False),
        "churn_risk_review.csv",
        "text/csv",
    )

with tab_data:
    st.subheader("Data-quality checks")
    quality_one, quality_two, quality_three = st.columns(3)
    quality_one.metric("Status", profile["status"].title())
    quality_two.metric("Rows checked", f"{profile['rows']:,}")
    quality_three.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
    with st.expander("View complete quality report"):
        st.json(profile)
    st.write("The raw dataset is kept separate from generated model outputs.")
