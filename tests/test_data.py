import pandas as pd
import pytest

from churn_analysis.data import clean_telco_data, load_telco_data


def test_cleaning_converts_churn_and_fills_blank_total_charges(tmp_path):
    path = tmp_path / "telco.csv"
    pd.DataFrame(
        {
            "customerID": ["a", "b"],
            "tenure": [0, 3],
            "MonthlyCharges": [30.0, 50.0],
            "TotalCharges": ["", "150.0"],
            "Churn": ["No", "Yes"],
        }
    ).to_csv(path, index=False)

    cleaned = clean_telco_data(load_telco_data(path))

    assert "customerID" not in cleaned.columns
    assert cleaned["TotalCharges"].tolist() == [0.0, 150.0]
    assert cleaned["Churn"].tolist() == [0, 1]


def test_loader_rejects_an_unexpected_schema(tmp_path):
    path = tmp_path / "invalid.csv"
    pd.DataFrame({"Churn": ["Yes"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_telco_data(path)
