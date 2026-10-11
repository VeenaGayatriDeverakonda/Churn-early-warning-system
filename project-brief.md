# Churn Early-Warning System: project brief

Your resume standout #1. Build this first.

## Why this one

Every subscription business obsesses over churn. This project says: "I can find the
customers about to leave, rank them by the revenue at stake, and tell the business
what to do about it." That is a business problem, not a coding exercise.

## What you will build

1. Clean the IBM Telco Customer Churn dataset (7,043 customers, 21 columns).
2. Explore where churn concentrates (contract type, tenure, internet service, add-ons).
3. Train an XGBoost churn model, tuned for recall. Missing a churner costs more
   than a false alarm.
4. Evaluate with ROC-AUC, precision/recall, and top-decile lift.
5. Translate to dollars: rank customers by churn probability x monthly charges to get
   revenue at risk, and write save plays for the top segment.
6. Package it: a README that reads like a business case and (optional) a Streamlit
   dashboard where a retention manager can filter at-risk accounts.

## Repo layout

```
churn-early-warning-system/
  README.md                  <- business-case write-up (fill in results as you go)
  requirements.txt
  data/
    README.md                <- how to fetch the dataset (one command)
  notebooks/
    01_churn_early_warning.ipynb  <- the full build, step by step
  src/
    clean.py                 <- reusable cleaning logic
    train.py                 <- reusable training + evaluation logic
```

## Build order (roughly 3-4 hours together)

- Hour 1: data + EDA. Run the notebook sections 0-3. The story usually jumps out
  fast: month-to-month fiber customers churn at 2x the average.
- Hour 2: modeling. Sections 4-6. Baseline logistic regression first, then XGBoost.
  Expect ROC-AUC around 0.84.
- Hour 3: business translation. Sections 7-8. Revenue at risk, save plays, README.
- Stretch: Streamlit dashboard with a ranked at-risk table.

## Resume bullet (use once built, with your real numbers)

"Built churn early-warning system on 7K telecom customers (XGBoost, AUC 0.84),
flagging $X in at-risk ARR with save-play recommendations for the retention team."

## Interview talking points

- Why recall over accuracy: a false alarm costs a retention call; a missed churner
  costs the whole account.
- The blank TotalCharges quirk: 11 customers with tenure 0 had blank totals. How you
  handled it shows data judgment.
- Feature story: contract type and tenure dominate. Fiber-optic month-to-month
  customers are the danger segment. Model features should echo that.
- What you would do next with real data: time-based validation, calibration,
  A/B test on save plays.

## Known data quirks (handled in src/clean.py)

- `TotalCharges` has 11 blank values, all with tenure 0. Convert to numeric and
  treat as 0.
- `customerID` is an identifier, never a model feature.
- Target `Churn` is Yes/No. Map to 1/0.
- Churn rate is about 26.5 percent, so use stratified splits and class weights.
