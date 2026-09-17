"""A transparent decision-support dashboard for the churn analysis."""

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

st.set_page_config(page_title="Customer Churn Decision Support", layout="wide")


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
    contract_summary["churn_rate"] = contract_summary["churn_rate"] * 100
    st.bar_chart(contract_summary.set_index("Contract")["churn_rate"])
    st.dataframe(contract_summary, use_container_width=True, hide_index=True)
    st.info(
        "The contract pattern is an association in this sample. It tells us "
        "where to investigate and test retention ideas; it does not prove why "
        "a customer leaves."
    )

with tab_model:
    st.subheader("Cross-validated model quality")
    try:
        cv_frame = pd.DataFrame(get_cross_validation()).T
        st.dataframe(cv_frame, use_container_width=True)
    except Exception:
        # The hosted runtime can have stricter limits than local Python. The
        # dashboard should remain useful while the full report stays in CI.
        st.warning(
            "Cross-validation is unavailable in this hosted session. "
            "Showing the verified holdout comparison instead."
        )
        st.dataframe(
            pd.DataFrame(get_holdout_metrics()).T,
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
    st.line_chart(scenarios.set_index("threshold")[["scenario_cost", "contacts"]])
    st.dataframe(scenarios, use_container_width=True, hide_index=True)
    st.caption("These costs are scenario inputs, not measured company costs.")

with tab_customers:
    st.subheader("Customers to review")
    threshold = st.slider("Risk threshold", 0.05, 0.95, 0.50, 0.05)
    model = get_full_model()
    customer_features = frame.drop(columns=["Churn"])
    scores = model.predict_proba(customer_features)[:, 1]
    risk = frame.copy()
    risk.insert(0, "customerID", load_customer_ids().to_numpy())
    risk["churn_probability"] = scores
    risk["review_recommended"] = risk["churn_probability"] >= threshold
    risk = risk.sort_values("churn_probability", ascending=False)
    st.write(f"{int(risk['review_recommended'].sum()):,} customers meet the review threshold.")
    display_columns = [
        "customerID",
        "tenure",
        "Contract",
        "InternetService",
        "MonthlyCharges",
        "churn_probability",
        "review_recommended",
    ]
    st.dataframe(risk[display_columns].head(100), use_container_width=True, hide_index=True)
    st.download_button(
        "Download risk review list",
        risk[display_columns].to_csv(index=False),
        "churn_risk_review.csv",
        "text/csv",
    )

with tab_data:
    st.subheader("Data-quality checks")
    st.json(profile)
    st.write("The raw dataset is kept separate from generated model outputs.")
