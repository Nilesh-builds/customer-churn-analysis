import pandas as pd

from churn_analysis.data import clean_telco_data
from churn_analysis.modeling import train_models


def test_all_baseline_models_return_imbalanced_classification_metrics():
    frame = pd.DataFrame(
        {
            "customerID": [str(i) for i in range(12)],
            "tenure": [1, 2, 3, 4, 5, 6, 12, 18, 24, 30, 36, 48],
            "MonthlyCharges": [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85],
            "TotalCharges": [30, 70, 120, 180, 250, 330, 720, 1170, 1680, 2250, 2880, 4080],
            "Contract": ["Month-to-month"] * 6 + ["One year"] * 3 + ["Two year"] * 3,
            "InternetService": ["DSL", "DSL", "Fiber optic"] * 4,
            "Churn": ["Yes", "No", "Yes", "No", "Yes", "No", "No", "No", "No", "No", "No", "No"],
        }
    )

    _, metrics = train_models(clean_telco_data(frame), test_size=0.25)

    assert set(metrics) == {
        "logistic_regression",
        "random_forest",
        "balanced_random_forest",
    }
    for result in metrics.values():
        assert set(result) == {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "pr_auc",
            "brier_score",
        }
        assert all(0 <= value <= 1 for value in result.values())
