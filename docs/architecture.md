# Architecture

The project has two paths that serve different purposes:

```text
raw CSV -> quality profile -> cleaning -> SQLite views -> SQL/reporting
                              |
                              +-> leakage-safe preprocessing -> models
                                                               |
                                                               +-> evaluation
                                                               +-> cost scenarios
                                                               +-> Streamlit review tool
```

The notebook remains the readable exploration record. The `src/churn_analysis`
package is the repeatable path used by tests, the command-line pipeline, and
the dashboard.

## Decisions

- `customerID` is removed before modeling because it identifies a row rather
  than describing customer behavior.
- Blank `TotalCharges` values are converted to zero because they represent
  customers who have not accumulated charges yet.
- Numeric scaling and categorical encoding happen inside a scikit-learn
  pipeline so test data cannot influence preprocessing.
- Recall, PR-AUC, calibration, and business cost are reported together. No
  single metric is treated as the whole decision.
- The dashboard recommends a human review list. It does not approve,
  reject, or contact customers automatically.

## Reproducibility

The command-line pipeline records data-quality results, holdout metrics,
cross-validation averages, calibration bins, and threshold assumptions in
`outputs/metrics/model_metrics.json`. The verified metrics snapshot is kept in
Git so the hosted dashboard does not need to retrain during startup. It also
creates a local SQLite database with reusable views; that database is ignored
by Git and can be rebuilt from the source CSV.
