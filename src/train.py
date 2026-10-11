"""Train and evaluate the churn early-warning model.

Baseline logistic regression first, then XGBoost tuned for recall.
Reports ROC-AUC, precision/recall, and top-decile lift, then ranks
customers by revenue at risk (churn probability x monthly charges).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

from clean import prepare


def top_decile_lift(y_true, proba):
    """How much more churn the top-risk decile captures vs random."""
    df = pd.DataFrame({"y": y_true, "p": proba}).sort_values("p", ascending=False)
    n = max(1, len(df) // 10)
    return df["y"].iloc[:n].mean() / df["y"].mean()


def main():
    X, y, monthly = prepare()

    X_train, X_test, y_train, y_test, m_train, m_test = train_test_split(
        X, y, monthly, test_size=0.2, random_state=42, stratify=y
    )

    # Baseline: balanced logistic regression.
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="roc_auc")
    print(f"baseline CV ROC-AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")

    # Main model: XGBoost. scale_pos_weight counters the ~26% churn rate.
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=neg / pos,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    print(f"xgboost test ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, proba > 0.5))
    print(f"top-decile lift: {top_decile_lift(y_test, proba):.2f}x")

    # Revenue at risk: probability-weighted monthly charges on the test set.
    risk = pd.DataFrame(
        {"churn_probability": proba, "monthly_charges": m_test.values}
    )
    risk["revenue_at_risk"] = risk["churn_probability"] * risk["monthly_charges"]
    top = risk.sort_values("revenue_at_risk", ascending=False).head(10)
    print("\nTop 10 accounts by revenue at risk:")
    print(top.round(2).to_string())
    print(
        f"\nTotal monthly revenue at risk in test set: "
        f"${risk['revenue_at_risk'].sum():,.0f}"
    )


if __name__ == "__main__":
    main()
