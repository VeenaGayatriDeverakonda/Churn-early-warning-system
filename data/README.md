# Data

The dataset is not committed to the repo. Download it with one command:

```bash
curl -L -o data/telco_churn.csv \
  https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

- Source: IBM's public Telco Customer Churn mirror on GitHub.
- 7,043 customer rows, 21 columns. Target column: `Churn` (Yes/No).
- Churn rate is about 26.5 percent.
- Known quirk: 11 rows have a blank `TotalCharges`, all with tenure 0.
  `src/clean.py` converts these to 0.
