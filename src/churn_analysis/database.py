"""Build a small local analytics database from the cleaned dataset."""

import sqlite3
from pathlib import Path

import pandas as pd


def build_sqlite_database(frame: pd.DataFrame, database_path: str | Path) -> Path:
    """Write cleaned customers and reusable analytics views to SQLite."""
    destination = Path(database_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(destination) as connection:
        frame.to_sql("customers", connection, if_exists="replace", index=False)
        connection.executescript(
            """
            DROP VIEW IF EXISTS churn_by_contract;
            CREATE VIEW churn_by_contract AS
            SELECT Contract AS contract,
                   COUNT(*) AS customers,
                   ROUND(AVG(Churn) * 100, 2) AS churn_rate_percent,
                   ROUND(AVG(MonthlyCharges), 2) AS average_monthly_charges
            FROM customers
            GROUP BY Contract;

            DROP VIEW IF EXISTS churn_by_tenure_band;
            CREATE VIEW churn_by_tenure_band AS
            SELECT CASE
                     WHEN tenure <= 3 THEN '0-3 months'
                     WHEN tenure <= 12 THEN '4-12 months'
                     WHEN tenure <= 24 THEN '13-24 months'
                     ELSE '25+ months'
                   END AS tenure_band,
                   COUNT(*) AS customers,
                   ROUND(AVG(Churn) * 100, 2) AS churn_rate_percent
            FROM customers
            GROUP BY tenure_band;

            DROP VIEW IF EXISTS retention_opportunities;
            CREATE VIEW retention_opportunities AS
            SELECT Contract AS contract,
                   COUNT(*) AS customers,
                   SUM(CASE WHEN Churn = 1 THEN 1 ELSE 0 END) AS observed_churners,
                   ROUND(SUM(MonthlyCharges), 2) AS monthly_revenue_exposure
            FROM customers
            GROUP BY Contract;
            """
        )
    return destination


def query_view(database_path: str | Path, view_name: str) -> pd.DataFrame:
    """Read one of the known analytics views."""
    allowed_views = {
        "churn_by_contract",
        "churn_by_tenure_band",
        "retention_opportunities",
    }
    if view_name not in allowed_views:
        raise ValueError(f"Unknown analytics view: {view_name}")
    with sqlite3.connect(database_path) as connection:
        return pd.read_sql_query(f"SELECT * FROM {view_name}", connection)
