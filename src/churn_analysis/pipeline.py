"""Run the reproducible baseline analysis from the command line."""

import argparse
import json
from pathlib import Path

from .data import clean_telco_data, load_telco_data, profile_telco_data
from .database import build_sqlite_database
from .modeling import (
    cross_validate_models,
    calibration_report,
    holdout_predictions,
    select_cost_optimal_threshold,
    threshold_report,
    train_models,
)


def run(input_path: Path, output_path: Path, database_path: Path | None = None) -> dict:
    """Run the reproducible analysis and save quality, model, and cost results."""
    raw_frame = load_telco_data(input_path)
    frame = clean_telco_data(raw_frame)
    _, metrics = train_models(frame)
    _, _, test_target, probabilities = holdout_predictions(frame)
    threshold_scenarios = threshold_report(probabilities, test_target)
    report = {
        "data_profile": profile_telco_data(raw_frame),
        "holdout_metrics": metrics["balanced_random_forest"],
        "model_metrics": metrics,
        "cross_validation": cross_validate_models(frame),
        "threshold_assumptions": {
            "contact_cost": 5.0,
            "missed_churn_cost": 100.0,
        },
        "lowest_cost_threshold": select_cost_optimal_threshold(threshold_scenarios),
        "calibration": calibration_report(probabilities, test_target),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if database_path is not None:
        build_sqlite_database(frame, database_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/metrics/model_metrics.json"),
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/churn_analysis.db"),
    )
    args = parser.parse_args()
    report = run(args.input, args.output, args.database)
    for model_name, values in report["model_metrics"].items():
        print(f"{model_name}: recall={values['recall']:.3f}, f1={values['f1']:.3f}")
    print(
        "lowest-cost threshold: "
        f"{report['lowest_cost_threshold']['threshold']:.2f}"
    )


if __name__ == "__main__":
    main()
