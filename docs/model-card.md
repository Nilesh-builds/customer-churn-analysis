# Model Card

## Intended use

This project is an educational and portfolio demonstration of churn-risk
analysis. It can help a retention analyst decide which customers to review
first in a hypothetical telco setting.

## Not intended for

- Automatic cancellation, rejection, or pricing decisions
- Decisions about real customers without governance and validation
- Claims about causal reasons for churn
- Measuring actual financial impact without an intervention experiment

## Data

The model uses the IBM Telco Customer Churn sample dataset distributed through
Kaggle. It contains 7,043 customer records and a binary churn outcome. It is a
static sample rather than a current production feed.

## Evaluation

The repository reports a stratified holdout evaluation and five-fold
cross-validation. Metrics include accuracy, precision, recall, F1, ROC-AUC,
PR-AUC, and Brier score. Threshold reports make contact cost and missed-churn
cost explicit assumptions.

## Limitations

- The data has no event timestamp, so temporal drift cannot be measured.
- The sample may not represent a real telco customer population.
- Observational relationships are not causal findings.
- The class-balanced model improves recall by accepting more false positives.
- Cost values in the dashboard are scenarios, not measured business costs.

## Human oversight

Any real deployment would require a retention specialist to review the reason
for a risk score, confirm that the intervention is appropriate, and monitor
outcomes. The dashboard intentionally presents a review queue instead of an
automatic action.
