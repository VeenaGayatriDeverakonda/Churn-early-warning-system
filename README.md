# Customer Churn Early-Warning System

## Problem

Churn is the silent revenue leak in any subscription business. Company-wide churn
numbers hide where the bleeding actually happens, and retention teams waste effort
on customers who were never leaving while the real flight risks get no attention.

## Approach

1. **Diagnose**: explored 7,043 telecom customer records to find where churn
   concentrates across contract type, tenure, internet service, and add-on services.
2. **Predict**: trained logistic regression and XGBoost classifiers to score every
   customer's churn probability, optimized for recall (catching churners matters
   more than avoiding false alarms).
3. **Prioritize**: ranked customers by churn probability x monthly charges to
   surface dollars of revenue at risk, and wrote segment-specific save plays.

## Impact (measured on a held-out 20% test set, n=1,409)

- Model: logistic regression, ROC-AUC **0.84**; XGBoost, ROC-AUC **0.84**
  (average precision 0.65). The simple baseline matched the boosted trees,
  so the honest model choice is the interpretable one.
- **Top-decile lift 2.9x**: the 10% of customers scored riskiest contain 107 of
  the 374 actual churners, a 76% hit rate versus 26.5% baseline.
- **$38,522/month in revenue at risk** across the test set ($462K annualized).
  The top 10% riskiest customers hold $10,192 of it (26%); the top 25% hold
  $22,365 (58%).
- Recommendation: target contract-upgrade and support add-on offers at
  month-to-month fiber customers in their first year, where churn is
  front-loaded and most concentrated.

## Key findings

- Month-to-month customers churn at **42.7%**, versus 11.3% on one-year and
  **2.8%** on two-year contracts. Contract length is the single strongest lever.
- Fiber-optic customers churn at **41.9%**, more than double DSL at 19.0%.
  The premium product has the worst retention.
- Customers paying by electronic check churn at **45.3%**, versus about 16% for
  automatic bank transfer or credit card. Autopay is a retention feature.
- **48.3%** of customers with under 12 months tenure churn. Churn is
  front-loaded: survival past year one cuts the risk in half.
- Customers without tech support churn at 41.6% versus 15.2% with it.
  Support add-ons protect revenue, not just satisfaction scores.

## Save plays

1. **New month-to-month fiber customers (tenure under 12 months)**: proactive
   outreach in the first 90 days with an annual-contract discount plus a free
   3-month tech-support trial. This is the highest-risk, highest-value segment.
2. **Electronic-check payers**: nudge to autopay with a one-time bill credit.
   Payment method is a behavioral signal the billing team can act on directly.
3. **No-add-on customers approaching month 6**: save-desk call offering
   OnlineSecurity or TechSupport at a trial price. Unprotected accounts churn
   at nearly 3x the rate of protected ones.

## Dataset

IBM Telco Customer Churn, 7,043 rows x 21 columns. Fetch it with:

```bash
curl -L -o data/telco_churn.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

Known quirk: 11 rows have blank `TotalCharges`, all with tenure 0. Treated as 0
(the honest value for customers with no billing history yet).

## Repo structure

```
README.md
requirements.txt
data/
  telco_churn.csv            # downloaded, gitignored
notebooks/
  01_churn_early_warning.ipynb
src/
  clean.py
  train.py
```

## How to run

```bash
pip install -r requirements.txt
# fetch the data (see Dataset above), then:
jupyter notebook notebooks/01_churn_early_warning.ipynb
# or run the pipeline scripts:
python src/clean.py && python src/train.py
```

## What I would do next

- Calibrate probabilities and validate on a time-based split.
- A/B test save plays on the top-risk segment to measure real retention lift.
- Ship the ranked at-risk table as a weekly dashboard for the retention team.
