import pandas as pd

from churn_analysis.data import profile_telco_data
from churn_analysis.database import build_sqlite_database, query_view
from churn_analysis.modeling import (
    calibration_report,
    select_cost_optimal_threshold,
    threshold_report,
)


def test_profile_reports_blank_total_charges_without_treating_it_as_an_error():
    frame = pd.DataFrame(
        {
            "customerID": ["a"],
            "tenure": [0],
            "MonthlyCharges": [30.0],
            "TotalCharges": [""],
            "Churn": ["No"],
            "Contract": ["Month-to-month"],
        }
    )

    report = profile_telco_data(frame)

    assert report["status"] == "pass"
    assert report["blank_total_charges"] == 1


def test_threshold_report_makes_cost_assumptions_visible():
    probabilities = pd.Series([0.9, 0.7, 0.4, 0.1])
    target = pd.Series([1, 0, 1, 0])

    report = threshold_report(
        probabilities,
        target,
        thresholds=[0.5],
        contact_cost=5,
        missed_churn_cost=100,
    )

    assert report[0]["contacts"] == 2
    assert report[0]["false_negative"] == 1
    assert report[0]["scenario_cost"] == 110
    assert select_cost_optimal_threshold(report)["threshold"] == 0.5


def test_sqlite_database_exposes_business_views(tmp_path):
    frame = pd.DataFrame(
        {
            "Contract": ["Month-to-month", "Two year"],
            "tenure": [2, 30],
            "MonthlyCharges": [70.0, 50.0],
            "Churn": [1, 0],
        }
    )

    database_path = build_sqlite_database(frame, tmp_path / "churn.db")
    result = query_view(database_path, "churn_by_contract")

    assert set(result["contract"]) == {"Month-to-month", "Two year"}
    assert result["customers"].sum() == 2


def test_calibration_report_returns_observed_and_predicted_rates():
    report = calibration_report(
        pd.Series([0.05, 0.15, 0.85, 0.95]),
        pd.Series([0, 0, 1, 1]),
        bins=2,
    )

    assert len(report) == 2
    assert set(report[0]) == {
        "mean_predicted_probability",
        "observed_churn_rate",
    }
