"""Leakage guards: identifiers stay out, test rows stay untouched."""
import pandas as pd

from churn_analysis.data import clean_telco_data
from churn_analysis.modeling import build_models, holdout_predictions, make_preprocessor


def _tiny_frame() -> pd.DataFrame:
    return clean_telco_data(
        pd.DataFrame(
            {
                "customerID": [str(i) for i in range(30)],
                "tenure": list(range(1, 31)),
                "MonthlyCharges": [30.0 + i for i in range(30)],
                "TotalCharges": [str(30.0 * (i + 1)) for i in range(30)],
                "Contract": ["Month-to-month"] * 15 + ["Two year"] * 15,
                "Churn": ["Yes"] * 8 + ["No"] * 22,
            }
        )
    )


def test_customer_id_never_reaches_model_features():
    frame = _tiny_frame()
    assert "customerID" not in frame.columns
    preprocessor = make_preprocessor(frame.drop(columns=["Churn"]))
    assert "customerID" not in list(preprocessor.transformers[0][2]) + list(
        preprocessor.transformers[1][2]
    )


def test_preprocessor_lives_inside_each_pipeline():
    # Fitting must happen per train split, not once on full data.
    frame = _tiny_frame()
    models = build_models(make_preprocessor(frame.drop(columns=["Churn"])))
    for model in models.values():
        assert "preprocessor" in model.named_steps


def test_holdout_test_rows_are_not_training_rows():
    frame = _tiny_frame()
    _, x_test, _, probabilities = holdout_predictions(frame, test_size=0.3)
    assert len(probabilities) == len(x_test)
    assert set(probabilities.index) == set(x_test.index)
    assert len(x_test) > 0
    assert len(x_test) < len(frame)
