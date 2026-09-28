"""Customer churn decision-support dashboard.

Same data + modeling pipeline as before. Only the presentation layer was redesigned:
Plotly charts, a dark themed layout, plain-English captions and guided tabs.
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
GITHUB_URL = "https://github.com/Nilesh-builds/customer-churn-analysis"

# ---- Design tokens ---------------------------------------------------------
CYAN, VIOLET, PINK = "#22d3ee", "#a78bfa", "#f472b6"
AMBER, GREEN, SLATE = "#fbbf24", "#34d399", "#64748b"
TEXT, MUTED = "#cbd5e1", "#94a3b8"
GRID = "rgba(148,163,184,0.12)"
MODEL_COLORS = [CYAN, VIOLET, PINK, AMBER]
MODEL_NAMES = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "balanced_random_forest": "Balanced random forest",
}

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp { font-family: 'Inter', 'Segoe UI', sans-serif; }
.block-container { max-width: 1240px; padding-top: 1.6rem; padding-bottom: 3rem; }
footer { visibility: hidden; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* Hero */
.hero {
    position: relative; overflow: hidden;
    padding: 2.1rem 2.3rem; margin-bottom: 1.2rem;
    border-radius: 22px; border: 1px solid rgba(148,163,184,0.18);
    background:
        radial-gradient(600px 220px at 8% 0%, rgba(34,211,238,0.22), transparent 70%),
        radial-gradient(520px 240px at 100% 100%, rgba(244,114,182,0.20), transparent 70%),
        linear-gradient(135deg, #0b1020 0%, #15103a 60%, #062a3a 100%);
}
.hero-badge {
    display: inline-block; padding: 4px 12px; margin-bottom: 0.9rem;
    border-radius: 999px; font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #67e8f9; background: rgba(255,255,255,0.06);
    border: 1px solid rgba(34,211,238,0.45);
}
.hero h1 { margin: 0; padding: 0; font-size: 2.6rem; font-weight: 800; line-height: 1.1; color: #f8fafc; }
.hero h1 span {
    background: linear-gradient(90deg, #22d3ee, #a78bfa 55%, #f472b6);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero p { margin: 0.8rem 0 0; max-width: 720px; font-size: 1.02rem; color: #cbd5e1; }

/* KPI cards (st.metric) */
[data-testid="stMetric"] {
    padding: 1rem 1.15rem; border-radius: 16px;
    border: 1px solid rgba(148,163,184,0.18);
    background: linear-gradient(145deg, rgba(34,211,238,0.08), rgba(167,139,250,0.06));
}
[data-testid="stMetricLabel"] p {
    color: #94a3b8; font-size: 0.74rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.4rem; }
.stTabs [data-baseweb="tab"] { padding: 0.55rem 1.1rem; border-radius: 10px 10px 0 0; font-weight: 600; }

/* Guide + insight cards */
.step-card {
    height: 100%; padding: 0.95rem 1.1rem; border-radius: 14px;
    border: 1px solid rgba(148,163,184,0.16); background: rgba(148,163,184,0.05);
}
.step-num { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.14em; color: #22d3ee; }
.step-title { margin: 0.15rem 0 0.2rem; font-weight: 700; color: #f1f5f9; }
.step-text { font-size: 0.86rem; color: #94a3b8; }

.insight {
    display: flex; gap: 0.8rem; height: 100%; padding: 1rem 1.1rem;
    border-radius: 14px; border: 1px solid rgba(148,163,184,0.16);
    border-left: 4px solid #22d3ee; background: rgba(34,211,238,0.05);
}
.insight-icon { font-size: 1.5rem; line-height: 1.2; }
.insight-title { margin-bottom: 0.15rem; font-weight: 700; color: #f1f5f9; }
.insight-body { font-size: 0.9rem; color: #cbd5e1; }

.section-title { margin: 1.1rem 0 0.1rem; font-size: 1.35rem; font-weight: 700; color: #f8fafc; }
.section-sub { margin-bottom: 0.8rem; font-size: 0.93rem; color: #94a3b8; }

.gloss {
    padding: 0.8rem 1rem; border-radius: 12px;
    border: 1px dashed rgba(148,163,184,0.3); height: 100%;
}
.gloss b { color: #f1f5f9; }
.gloss span { display: block; margin-top: 0.15rem; font-size: 0.84rem; color: #94a3b8; }
</style>
"""


# ---- Small UI helpers ------------------------------------------------------
def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380, legend: bool = False) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=TEXT, size=13),
        title=dict(font=dict(size=16, color="#f1f5f9"), x=0.0, xanchor="left"),
        margin=dict(l=8, r=8, t=56, b=8),
        height=height,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text=""),
        hoverlabel=dict(bgcolor="#0f172a", font_size=13),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


def humanize(name: str) -> str:
    special = {"f1": "F1", "roc_auc": "ROC AUC", "auc": "AUC"}
    return special.get(name.lower(), name.replace("_", " ").title())


# ---- Cached data + model helpers (unchanged logic) -------------------------
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


# ---- Chart builders --------------------------------------------------------
def donut_chart(frame: pd.DataFrame) -> go.Figure:
    churned = int(frame["Churn"].sum())
    retained = len(frame) - churned
    fig = go.Figure(
        go.Pie(
            labels=["Retained", "Churned"],
            values=[retained, churned],
            hole=0.72,
            sort=False,
            marker=dict(colors=[CYAN, PINK], line=dict(color="#0b1020", width=3)),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,} customers (%{percent})<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"<b>{frame['Churn'].mean():.1%}</b><br><span style='font-size:12px;color:{MUTED}'>churned</span>",
        showarrow=False,
        font=dict(size=30, color="#f8fafc"),
    )
    fig.update_layout(title="Who stays vs. who leaves")
    return style_fig(fig, height=360, legend=True)


def contract_chart(frame: pd.DataFrame) -> go.Figure:
    df = (
        frame.groupby("Contract")
        .agg(customers=("Churn", "size"), churn_rate=("Churn", "mean"))
        .reset_index()
        .sort_values("churn_rate")
    )
    rates = df["churn_rate"] * 100
    fig = go.Figure(
        go.Bar(
            x=rates,
            y=df["Contract"],
            orientation="h",
            marker=dict(
                color=rates,
                colorscale=[[0, CYAN], [0.5, VIOLET], [1, PINK]],
                showscale=False,
                line=dict(width=0),
            ),
            text=[f"{v:.1f}%" for v in rates],
            textposition="outside",
            cliponaxis=False,
            customdata=df["customers"],
            hovertemplate="<b>%{y}</b><br>Churn rate: %{x:.1f}%<br>Customers: %{customdata:,}<extra></extra>",
        )
    )
    overall = frame["Churn"].mean() * 100
    fig.add_vline(
        x=overall,
        line_dash="dash",
        line_color=AMBER,
        annotation_text=f"Average {overall:.1f}%",
        annotation_position="top",
        annotation_font_color=AMBER,
    )
    fig.update_xaxes(title="Churn rate (%)", range=[0, max(rates.max() * 1.2, overall * 1.3)])
    fig.update_yaxes(title="")
    fig.update_layout(title="Churn rate by contract type")
    return style_fig(fig, height=360)


def tenure_chart(frame: pd.DataFrame) -> go.Figure:
    buckets = pd.cut(
        frame["tenure"],
        bins=[-1, 3, 12, 24, 48, 10_000],
        labels=["0–3 mo", "4–12 mo", "13–24 mo", "25–48 mo", "49+ mo"],
    )
    df = frame.groupby(buckets, observed=True)["Churn"].agg(["mean", "size"]).reset_index()
    df.columns = ["bucket", "rate", "customers"]
    rates = df["rate"] * 100
    fig = go.Figure(
        go.Scatter(
            x=df["bucket"].astype(str),
            y=rates,
            mode="lines+markers+text",
            line=dict(color=CYAN, width=3, shape="spline"),
            marker=dict(size=11, color=PINK, line=dict(color="#0b1020", width=2)),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.12)",
            text=[f"{v:.0f}%" for v in rates],
            textposition="top center",
            customdata=df["customers"],
            hovertemplate="<b>%{x}</b><br>Churn rate: %{y:.1f}%<br>Customers: %{customdata:,}<extra></extra>",
        )
    )
    fig.update_yaxes(title="Churn rate (%)", range=[0, rates.max() * 1.25])
    fig.update_xaxes(title="How long they've been a customer")
    fig.update_layout(title="Churn falls as customers stay longer")
    return style_fig(fig, height=360)


def heatmap_chart(frame: pd.DataFrame) -> go.Figure:
    pivot = (
        frame.pivot_table(index="InternetService", columns="Contract", values="Churn", aggfunc="mean") * 100
    )
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=[[0, "#0f172a"], [0.5, "#7c3aed"], [1, PINK]],
            xgap=4,
            ygap=4,
            texttemplate="%{z:.1f}%",
            textfont=dict(size=15, color="white"),
            colorbar=dict(title="Churn %", thickness=12),
            hovertemplate="<b>%{y}</b> · %{x}<br>Churn rate: %{z:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(title="Hot spots: internet service × contract")
    return style_fig(fig, height=360)


def charges_chart(frame: pd.DataFrame) -> go.Figure:
    plot_frame = frame.assign(Status=frame["Churn"].map({1: "Churned", 0: "Retained"}))
    fig = px.histogram(
        plot_frame,
        x="MonthlyCharges",
        color="Status",
        barmode="overlay",
        nbins=40,
        opacity=0.7,
        histnorm="percent",
        color_discrete_map={"Churned": PINK, "Retained": CYAN},
    )
    fig.update_xaxes(title="Monthly charge ($)")
    fig.update_yaxes(title="Share of group (%)")
    fig.update_layout(title="Monthly charges: churned vs. retained")
    return style_fig(fig, height=360, legend=True)


def build_insights(frame: pd.DataFrame) -> list:
    insights = []
    overall = frame["Churn"].mean()

    by_contract = frame.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
    if len(by_contract) >= 2 and by_contract.iloc[-1] > 0:
        ratio = by_contract.iloc[0] / by_contract.iloc[-1]
        insights.append(
            (
                "📄",
                "Contract type stands out",
                f"<b>{by_contract.index[0]}</b> customers churn at <b>{by_contract.iloc[0]:.1%}</b>, "
                f"about <b>{ratio:.0f}×</b> the rate of <b>{by_contract.index[-1]}</b> ({by_contract.iloc[-1]:.1%}).",
            )
        )

    early = frame.loc[frame["tenure"] <= 3, "Churn"]
    if len(early):
        insights.append(
            (
                "⏳",
                "The first 3 months are risky",
                f"New customers (0–3 months) churn at <b>{early.mean():.1%}</b> versus <b>{overall:.1%}</b> overall.",
            )
        )

    by_service = frame.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)
    if len(by_service):
        insights.append(
            (
                "🌐",
                "One internet plan leads",
                f"<b>{by_service.index[0]}</b> customers churn at <b>{by_service.iloc[0]:.1%}</b>, "
                "the highest of any internet service.",
            )
        )
    return insights


# ---- Page: hero, sidebar, KPIs --------------------------------------------
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">Decision support · Portfolio project</div>
        <h1>Customer Churn <span>Dashboard</span></h1>
        <p>See who is likely to leave, why, and where a retention team should look first.
        Predictions are evidence for a human conversation, not automatic customer decisions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### 📉 About this demo")
    st.write(
        "I built this dashboard to show how a churn model can support a human "
        "retention review instead of making an automatic customer decision."
    )
    st.link_button("View source on GitHub", GITHUB_URL)
    st.markdown("**Workflow**")
    st.caption("Raw data → quality checks → model → cost scenario → review list")
    st.markdown("**How to read it**")
    st.caption(
        "Cyan means healthy or retained. Pink means high churn or high risk. "
        "Every chart has a one-line explanation underneath."
    )
    st.markdown("**Important**")
    st.caption("The dataset is a static sample. Cost values are adjustable scenarios, not company facts.")

frame = load_clean_data()
profile = get_profile()
features, target, probabilities = get_holdout_data()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Customers", f"{len(frame):,}")
k2.metric("Churn rate", f"{frame['Churn'].mean():.1%}", help="Share of customers who left in this dataset.")
k3.metric("Customers lost", f"{int(frame['Churn'].sum()):,}")
k4.metric("Avg. tenure", f"{frame['tenure'].mean():.0f} mo")
k5.metric("Data quality", str(profile["status"]).title())

st.write("")
g1, g2, g3 = st.columns(3)
g1.markdown(
    '<div class="step-card"><div class="step-num">STEP 1 · OVERVIEW</div>'
    '<div class="step-title">Where does churn happen?</div>'
    '<div class="step-text">Charts that show which customers leave and when.</div></div>',
    unsafe_allow_html=True,
)
g2.markdown(
    '<div class="step-card"><div class="step-num">STEP 2 · MODEL QUALITY</div>'
    '<div class="step-title">Can we trust the predictions?</div>'
    '<div class="step-text">How well each model performs, and what acting on it costs.</div></div>',
    unsafe_allow_html=True,
)
g3.markdown(
    '<div class="step-card"><div class="step-num">STEP 3 · CUSTOMER RISK</div>'
    '<div class="step-title">Who should we contact?</div>'
    '<div class="step-text">A ranked review list you can filter and download.</div></div>',
    unsafe_allow_html=True,
)
st.write("")

tab_overview, tab_model, tab_customers, tab_data = st.tabs(
    ["📊 Overview", "🧪 Model quality", "🎯 Customer risk", "🛡️ Data quality"]
)

# ---- Tab 1: Overview -------------------------------------------------------
with tab_overview:
    section("Where is churn concentrated?", "The key patterns in this dataset, in plain language.")

    insights = build_insights(frame)
    if insights:
        for column, (icon, title, body) in zip(st.columns(len(insights)), insights):
            column.markdown(
                f'<div class="insight"><div class="insight-icon">{icon}</div><div>'
                f'<div class="insight-title">{title}</div><div class="insight-body">{body}</div></div></div>',
                unsafe_allow_html=True,
            )
    st.write("")

    left, right = st.columns([1, 2])
    with left:
        with st.container(border=True):
            show(donut_chart(frame))
            st.caption("💡 The centre number is the share of customers who left.")
    with right:
        with st.container(border=True):
            show(contract_chart(frame))
            st.caption("💡 Longer bar = more customers leaving. The dashed line is the overall average.")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            show(tenure_chart(frame))
            st.caption("💡 Churn is highest for brand-new customers and drops the longer they stay.")
    with right:
        with st.container(border=True):
            show(heatmap_chart(frame))
            st.caption("💡 Brighter pink cells are the riskiest combinations of plan and contract.")

    with st.container(border=True):
        show(charges_chart(frame))
        st.caption("💡 If the pink shape sits further right, churned customers tend to pay more per month.")

    st.info(
        "These patterns are associations in this sample. They tell us where to investigate "
        "and test retention ideas; they do not prove why a customer leaves."
    )

# ---- Tab 2: Model quality --------------------------------------------------
with tab_model:
    section(
        "Can we trust the predictions?",
        "We compare three models on data they were not trained on. Taller bars are better.",
    )

    try:
        metrics_frame = pd.DataFrame(get_cross_validation()).T
        source_note = "Cross-validated scores (average across several train/test splits)."
    except Exception:
        # The hosted runtime can have stricter limits than local Python. The
        # dashboard should remain useful while the full report stays in CI.
        st.warning(
            "Cross-validation is unavailable in this hosted session. "
            "Showing the verified holdout comparison instead."
        )
        metrics_frame = pd.DataFrame(get_holdout_metrics()).T
        source_note = f"Holdout scores ({len(target):,} customers the models never saw during training)."

    metrics_frame.index = [MODEL_NAMES.get(name, humanize(name)) for name in metrics_frame.index]
    metrics_frame.index.name = "Model"
    metrics_frame.columns = [humanize(str(column)) for column in metrics_frame.columns]

    long_frame = metrics_frame.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig_models = px.bar(
        long_frame,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text_auto=".0%",
        color_discrete_sequence=MODEL_COLORS,
    )
    fig_models.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
    fig_models.update_yaxes(tickformat=".0%", range=[0, 1.08], title="")
    fig_models.update_xaxes(title="")
    fig_models.update_layout(title="Model scorecard")
    with st.container(border=True):
        show(style_fig(fig_models, height=420, legend=True))
        st.caption(f"💡 {source_note}")

    recall_columns = [column for column in metrics_frame.columns if "recall" in column.lower()]
    if recall_columns:
        best_model = metrics_frame[recall_columns[0]].idxmax()
        best_value = metrics_frame[recall_columns[0]].max()
        st.success(
            f"**{best_model}** catches the most real churners ({best_value:.1%} recall). "
            "Catching churners early matters more here than avoiding a few unnecessary offers."
        )

    st.markdown("**What do these scores mean?**")
    glossary = [
        ("Accuracy", "Share of all predictions that were correct."),
        ("Precision", "Of the customers we flag, how many really leave."),
        ("Recall", "Of the customers who really leave, how many we catch."),
        ("F1", "One score that balances precision and recall."),
    ]
    for column, (name, meaning) in zip(st.columns(4), glossary):
        column.markdown(
            f'<div class="gloss"><b>{name}</b><span>{meaning}</span></div>', unsafe_allow_html=True
        )

    with st.expander("See the exact numbers"):
        st.dataframe(
            metrics_frame.style.format({column: "{:.1%}" for column in metrics_frame.columns}),
            use_container_width=True,
        )

    section(
        "What does it cost to act?",
        "Choose how risky a customer must look before your team contacts them. "
        "Costs are adjustable what-if inputs.",
    )
    c1, c2 = st.columns(2)
    contact_cost = c1.number_input("Cost of contacting one customer ($)", min_value=0.0, value=5.0, step=1.0)
    missed_cost = c2.number_input(
        "Scenario cost of a missed churner ($)", min_value=0.0, value=100.0, step=10.0
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

    left, right = st.columns(2)
    with left:
        fig_cost = go.Figure()
        fig_cost.add_trace(
            go.Scatter(
                x=scenarios["threshold"],
                y=scenarios["scenario_cost"],
                mode="lines+markers",
                line=dict(color=CYAN, width=3, shape="spline"),
                marker=dict(size=7),
                fill="tozeroy",
                fillcolor="rgba(34,211,238,0.10)",
                hovertemplate="Threshold %{x:.0%}<br>Cost $%{y:,.0f}<extra></extra>",
            )
        )
        fig_cost.add_trace(
            go.Scatter(
                x=[best["threshold"]],
                y=[best["scenario_cost"]],
                mode="markers+text",
                text=["Lowest cost"],
                textposition="top center",
                marker=dict(size=16, color=PINK, line=dict(color="white", width=2)),
                hoverinfo="skip",
            )
        )
        fig_cost.update_xaxes(title="Risk threshold", tickformat=".0%")
        fig_cost.update_yaxes(title="Scenario cost ($)", rangemode="tozero")
        fig_cost.update_layout(title="Total cost by risk threshold")
        with st.container(border=True):
            show(style_fig(fig_cost, height=380))
            st.caption("💡 The pink dot is the cheapest place to draw the line.")
    with right:
        fig_tradeoff = go.Figure()
        fig_tradeoff.add_trace(
            go.Scatter(
                x=scenarios["threshold"],
                y=scenarios["recall"],
                name="Recall (churners caught)",
                mode="lines+markers",
                line=dict(color=PINK, width=3),
            )
        )
        fig_tradeoff.add_trace(
            go.Scatter(
                x=scenarios["threshold"],
                y=scenarios["precision"],
                name="Precision (flags that are right)",
                mode="lines+markers",
                line=dict(color=CYAN, width=3),
            )
        )
        fig_tradeoff.update_xaxes(title="Risk threshold", tickformat=".0%")
        fig_tradeoff.update_yaxes(title="", tickformat=".0%", range=[0, 1.05])
        fig_tradeoff.update_layout(title="The trade-off: catch more vs. be more precise")
        with st.container(border=True):
            show(style_fig(fig_tradeoff, height=380, legend=True))
            st.caption("💡 A lower threshold catches more churners but contacts more people by mistake.")

    with st.expander("See the scenario table"):
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

# ---- Tab 3: Customer risk --------------------------------------------------
with tab_customers:
    section(
        "Who should we contact first?",
        "Move the slider to decide how broad the human review queue should be.",
    )
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
    q1, q2, q3 = st.columns(3)
    q1.metric("Review queue", f"{review_count:,} customers")
    q2.metric("Share of all customers", f"{review_count / len(risk):.1%}")
    q3.metric("Highest predicted risk", f"{risk['churn_probability'].max():.1%}")

    left, right = st.columns([2, 1])
    with left:
        fig_hist = go.Figure(
            go.Histogram(
                x=risk["churn_probability"] * 100,
                nbinsx=25,
                marker=dict(color=VIOLET, line=dict(color="#0b1020", width=1)),
                opacity=0.9,
                hovertemplate="Risk %{x:.0f}%<br>%{y} customers<extra></extra>",
            )
        )
        fig_hist.add_vline(
            x=threshold * 100,
            line_dash="dash",
            line_color=AMBER,
            annotation_text="Your threshold",
            annotation_font_color=AMBER,
        )
        fig_hist.update_xaxes(title="Predicted churn risk (%)")
        fig_hist.update_yaxes(title="Customers")
        fig_hist.update_layout(title="How risky are our customers?")
        with st.container(border=True):
            show(style_fig(fig_hist, height=340))
            st.caption("💡 Customers to the right of the dashed line land in the review queue.")
    with right:
        band_counts = risk["risk_band"].value_counts().reindex(["Lower", "Medium", "Higher"])
        fig_bands = go.Figure(
            go.Bar(
                x=band_counts.index.astype(str),
                y=band_counts.values,
                marker=dict(color=[GREEN, AMBER, PINK], line=dict(width=0)),
                text=[f"{v:,}" for v in band_counts.values],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{x}: %{y:,} customers<extra></extra>",
            )
        )
        fig_bands.update_yaxes(title="", range=[0, band_counts.max() * 1.2])
        fig_bands.update_xaxes(title="")
        fig_bands.update_layout(title="Customers by risk band")
        with st.container(border=True):
            show(style_fig(fig_bands, height=340))
            st.caption("💡 Lower < 33% · Medium 33–66% · Higher > 66%.")

    st.markdown("**Top 100 customers by predicted risk**")
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
    risk_display = risk[display_columns].head(100).copy()
    risk_display["churn_probability"] = risk_display["churn_probability"] * 100
    st.dataframe(
        risk_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "customerID": st.column_config.TextColumn("Customer ID"),
            "tenure": st.column_config.NumberColumn("Tenure (months)", format="%d"),
            "Contract": st.column_config.TextColumn("Contract"),
            "InternetService": st.column_config.TextColumn("Internet"),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly charge", format="$%.2f"),
            "churn_probability": st.column_config.ProgressColumn(
                "Churn risk", format="%.1f%%", min_value=0, max_value=100
            ),
            "risk_band": st.column_config.TextColumn("Risk band"),
            "review_recommended": st.column_config.CheckboxColumn("Review?"),
        },
    )
    st.download_button(
        "⬇️ Download risk review list",
        risk[display_columns].to_csv(index=False),
        "churn_risk_review.csv",
        "text/csv",
    )
    st.caption(
        "Scores come from a model trained on this same sample, so treat them as a demo "
        "of the workflow rather than a production forecast."
    )

# ---- Tab 4: Data quality ---------------------------------------------------
with tab_data:
    section("Is the data trustworthy?", "Basic checks run on the raw file before any modelling.")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Status", str(profile["status"]).title())
    d2.metric("Rows checked", f"{profile['rows']:,}")
    d3.metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
    d4.metric("Missing values (cleaned)", f"{int(frame.isna().sum().sum()):,}")
    with st.expander("View complete quality report"):
        st.json(profile)
    st.caption("The raw dataset is kept separate from generated model outputs.")

st.divider()
st.caption("Built by Nilesh Singh · Streamlit + Plotly · Telco Customer Churn dataset (IBM sample via Kaggle)")
