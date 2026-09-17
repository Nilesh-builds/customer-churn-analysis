"""Model training and evaluation for the churn classification task."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Create one preprocessing path shared by every candidate model."""
    categorical = features.select_dtypes(include=["object", "category"]).columns
    numeric = features.select_dtypes(exclude=["object", "category"]).columns
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(numeric)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(categorical),
            ),
        ]
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Return the three models compared in the original analysis."""
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(n_estimators=100, random_state=42),
                ),
            ]
        ),
        "balanced_random_forest": Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=100,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, target: pd.Series
) -> dict[str, Any]:
    """Evaluate a fitted model using metrics suited to churn work."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "accuracy": round(accuracy_score(target, predictions), 4),
        "precision": round(precision_score(target, predictions, zero_division=0), 4),
        "recall": round(recall_score(target, predictions, zero_division=0), 4),
        "f1": round(f1_score(target, predictions, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(target, probabilities), 4),
        "pr_auc": round(average_precision_score(target, probabilities), 4),
        "brier_score": round(brier_score_loss(target, probabilities), 4),
    }


def train_models(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]]]:
    """Train candidate models on a fixed stratified holdout split."""
    features = frame.drop(columns=["Churn"])
    target = frame["Churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    models = build_models(make_preprocessor(x_train))
    metrics: dict[str, dict[str, Any]] = {}
    fitted_models: dict[str, Pipeline] = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        metrics[name] = evaluate_model(model, x_test, y_test)
        fitted_models[name] = model
    return fitted_models, metrics


def cross_validate_models(
    frame: pd.DataFrame, folds: int = 5, random_state: int = 42
) -> dict[str, dict[str, float]]:
    """Estimate model stability across stratified folds."""
    features = frame.drop(columns=["Churn"])
    target = frame["Churn"]
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    results: dict[str, dict[str, float]] = {}
    for name, model in build_models(make_preprocessor(features)).items():
        scores = cross_validate(
            model,
            features,
            target,
            cv=splitter,
            scoring=scoring,
            n_jobs=-1,
        )
        results[name] = {
            metric: round(float(scores[f"test_{metric}"].mean()), 4)
            for metric in scoring
        }
    return results


def threshold_report(
    probabilities: pd.Series,
    target: pd.Series,
    thresholds: list[float] | None = None,
    contact_cost: float = 5.0,
    missed_churn_cost: float = 100.0,
) -> list[dict[str, float]]:
    """Compare outreach thresholds using explicit, adjustable costs.

    The costs are scenario inputs, not measured company costs. Keeping them
    visible makes the business assumption reviewable rather than hiding it in
    a model choice.
    """
    thresholds = thresholds or [round(value / 20, 2) for value in range(1, 20)]
    actual = target.to_numpy()
    scores = probabilities.to_numpy()
    report = []
    for threshold in thresholds:
        predicted = (scores >= threshold).astype(int)
        true_positive = int(((predicted == 1) & (actual == 1)).sum())
        false_positive = int(((predicted == 1) & (actual == 0)).sum())
        false_negative = int(((predicted == 0) & (actual == 1)).sum())
        true_negative = int(((predicted == 0) & (actual == 0)).sum())
        contacts = true_positive + false_positive
        report.append(
            {
                "threshold": threshold,
                "contacts": contacts,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "true_negative": true_negative,
                "precision": round(true_positive / contacts, 4) if contacts else 0.0,
                "recall": round(
                    true_positive / (true_positive + false_negative), 4
                )
                if true_positive + false_negative
                else 0.0,
                "scenario_cost": round(
                    contacts * contact_cost + false_negative * missed_churn_cost, 2
                ),
            }
        )
    return report


def select_cost_optimal_threshold(report: list[dict[str, float]]) -> dict[str, float]:
    """Return the lowest-cost scenario, preserving all its assumptions."""
    if not report:
        raise ValueError("At least one threshold scenario is required")
    return min(report, key=lambda row: (row["scenario_cost"], row["threshold"]))


def calibration_report(
    probabilities: pd.Series, target: pd.Series, bins: int = 10
) -> list[dict[str, float]]:
    """Compare predicted risk with observed churn in probability bins."""
    observed, predicted = calibration_curve(
        target, probabilities, n_bins=bins, strategy="uniform"
    )
    return [
        {
            "mean_predicted_probability": round(float(predicted_value), 4),
            "observed_churn_rate": round(float(observed_value), 4),
        }
        for observed_value, predicted_value in zip(observed, predicted)
    ]


def fit_full_model(frame: pd.DataFrame, model_name: str = "balanced_random_forest") -> Pipeline:
    """Fit a named model on all available rows for dashboard scoring."""
    features = frame.drop(columns=["Churn"])
    model = build_models(make_preprocessor(features))[model_name]
    model.fit(features, frame["Churn"])
    return model


def holdout_predictions(
    frame: pd.DataFrame,
    model_name: str = "balanced_random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, pd.DataFrame, pd.Series, pd.Series]:
    """Return a fitted holdout model and untouched test probabilities."""
    features = frame.drop(columns=["Churn"])
    target = frame["Churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
    model = build_models(make_preprocessor(x_train))[model_name]
    model.fit(x_train, y_train)
    probabilities = pd.Series(
        model.predict_proba(x_test)[:, 1], index=x_test.index, name="churn_probability"
    )
    return model, x_test, y_test, probabilities
