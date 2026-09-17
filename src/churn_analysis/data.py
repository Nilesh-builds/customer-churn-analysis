"""Data loading and cleaning decisions for the Telco churn dataset."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "customerID",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}

EXPECTED_CATEGORICAL_VALUES = {
    "Churn": {"Yes", "No"},
    "Contract": {"Month-to-month", "One year", "Two year"},
}


def load_telco_data(path: str | Path) -> pd.DataFrame:
    """Load the source CSV and fail early when its schema is unexpected."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"The churn dataset is missing columns: {missing_names}")
    return frame


def clean_telco_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning choices used in the original notebook.

    Blank TotalCharges values belong to brand-new customers. Treating those
    blanks as zero is more faithful to the meaning of the field than silently
    dropping those customers from the analysis.
    """
    cleaned = frame.copy()
    cleaned["TotalCharges"] = pd.to_numeric(
        cleaned["TotalCharges"], errors="coerce"
    ).fillna(0)

    if cleaned["Churn"].dtype == "object":
        churn_values = cleaned["Churn"].str.strip().map({"Yes": 1, "No": 0})
        if churn_values.isna().any():
            raise ValueError("Churn contains values other than Yes and No")
        cleaned["Churn"] = churn_values.astype(int)

    return cleaned.drop(columns=["customerID"], errors="ignore")


def profile_telco_data(frame: pd.DataFrame) -> dict:
    """Create a small, serializable data-quality report before modeling."""
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    invalid_categories = {}
    for column, expected in EXPECTED_CATEGORICAL_VALUES.items():
        if column in frame:
            observed = set(frame[column].dropna().astype(str).str.strip())
            unexpected = sorted(observed - expected)
            if unexpected:
                invalid_categories[column] = unexpected

    def numeric_column(name: str) -> pd.Series:
        if name not in frame:
            return pd.Series(index=frame.index, dtype="float64")
        return pd.to_numeric(frame[name], errors="coerce")

    numeric_total_charges = numeric_column("TotalCharges")
    numeric_tenure = numeric_column("tenure")
    numeric_monthly_charges = numeric_column("MonthlyCharges")
    invalid_ranges = {
        "tenure": int((numeric_tenure < 0).sum()),
        "MonthlyCharges": int((numeric_monthly_charges < 0).sum()),
        "TotalCharges": int((numeric_total_charges < 0).sum()),
    }
    invalid_ranges = {key: value for key, value in invalid_ranges.items() if value}
    missing_values = {
        str(column): int(value)
        for column, value in frame.isna().sum().items()
        if value
    }
    return {
        "status": "pass"
        if not missing_columns
        and not invalid_categories
        and not invalid_ranges
        and not missing_values
        and not frame.duplicated().any()
        else "fail",
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": missing_values,
        "blank_total_charges": int(numeric_total_charges.isna().sum()),
        "missing_required_columns": missing_columns,
        "unexpected_categories": invalid_categories,
        "invalid_numeric_ranges": invalid_ranges,
    }
