"""Cleaning logic for the IBM Telco Customer Churn dataset.

Handles the known quirks so the notebook and train script share one source
of truth:
- TotalCharges has 11 blank values, all with tenure 0 -> treated as 0
- customerID is an identifier, never a model feature
- Churn (Yes/No) -> 1/0 target
"""

import pandas as pd


TARGET = "Churn"
ID_COL = "customerID"


def load_raw(path="data/telco_churn.csv"):
    df = pd.read_csv(path)
    return df


def clean(df):
    df = df.copy()

    # TotalCharges arrives as object because of blank strings.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # The blanks all belong to tenure-0 customers; 0 is the honest value.
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def feature_lists(df):
    numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    categorical = [c for c in df.columns if c not in numeric + [TARGET, ID_COL]]
    return numeric, categorical


def prepare(path="data/telco_churn.csv"):
    """Load, clean, and return (X, y) with one-hot encoded features."""
    df = clean(load_raw(path))
    numeric, categorical = feature_lists(df)

    X = pd.get_dummies(df[numeric + categorical], columns=categorical, drop_first=True)
    y = df[TARGET]

    # Keep MonthlyCharges on the side for the revenue-at-risk ranking.
    monthly = df["MonthlyCharges"]
    return X, y, monthly


if __name__ == "__main__":
    X, y, _ = prepare()
    print(f"rows={X.shape[0]} features={X.shape[1]} churn_rate={y.mean():.3f}")
