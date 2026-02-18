# Census Income — Classification + Marketing Segmentation

This repo contains a take-home style analysis on the UCI/US Census Bureau–style income dataset.

## Contents

- `EDA_classification.ipynb`
  - Binary classification: predict whether income is `> 50k`.
  - Business framing: marketing targeting with an imbalanced positive rate (~6%).
  - Metrics: PR-AUC, ROC-AUC, lift/precision at top-K.
  - Models: Logistic Regression baseline, isotonic calibration, HistGradientBoosting benchmark.
  - Evaluation protocol: **holdout TEST once + CV/tuning on DEV** (cleaner, avoids test leakage).

- `EDA_segmentation.ipynb`
  - Unsupervised segmentation: cluster prospects into interpretable groups.
  - Model: MiniBatchKMeans on a mixed numeric + categorical feature pipeline.
  - Outputs: segment shares, segment profiles, and `segmentation_assignments.csv`.
  - Notes: `y` (income>50k) is used **only for interpretation** of segment value, not for training clusters.

- Data
  - `census-bureau.data` — raw comma-separated data
  - `census-bureau.columns` — one column name per line

## Environment / Requirements

This project is notebook-first and uses standard Python ML tooling.

Recommended:
- Python 3.8+ (this repo was tested in a Python 3.8 environment)
- scikit-learn 1.3+ (the notebooks use `OneHotEncoder(..., sparse_output=True)`)

Typical packages:
- `pandas`, `numpy`
- `scikit-learn`
- `matplotlib`, `seaborn`

### (Optional) Create a virtual environment

If you want an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib seaborn nbformat
```

Or, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If you are using conda, just ensure the packages above are available.

## How to run

### Run via VS Code / Jupyter UI (recommended)

1. Open the repo in VS Code.
2. Open either notebook:
   - `EDA_classification.ipynb`
   - `EDA_segmentation.ipynb`
3. Select your Python kernel and run cells from top to bottom.


## Key outputs

### Classification notebook
- Reported TEST metrics:
  - ROC-AUC
  - PR-AUC
  - Precision/Recall/Lift at top K% targeted
- Model comparison summary table.

### Segmentation notebook
- Segment population share + profiles.
- Optional business interpretation: `income_rate` (mean of `y`) per segment.
- Export file:
  - `segmentation_assignments.csv` (row-level segment IDs)

## Notes / Caveats

- Variables like `sex`, `race`, and citizenship-related fields may raise fairness/compliance concerns. Use them only if your organization allows it and you have a documented policy.
- For real-world activation, segments should be validated with downstream KPIs (conversion rate, AOV, churn/LTV) via A/B tests.

## Repository structure

```text
.
├── EDA_classification.ipynb
├── EDA_segmentation.ipynb
├── census-bureau.data
├── census-bureau.columns
├── segmentation_assignments.csv
└── README.md
```