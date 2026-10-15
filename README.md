# MLE-AI-201 Machine Learning — Final Project

This repository implements a decision tree and a Pegasos SVM in NumPy for
**bina.az** sale listings, then compares them with scikit-learn baselines.
The full assignment is in `PROJECT_STATEMENT.pdf`.

- **Released:** Sun 04 Oct 2026
- **Due:** **Sun 18 Oct 2026, 23:59 (Baku)** — two weeks, no late submissions.
- **Demo:** defended in the session after the deadline.

## What you submit (on Moodle)
Exactly **three** things; **one member submits** for the whole team:
1. **GitHub link** to your repo (tagged `v1.0-final`).
2. **Report PDF** (IEEE two-column — use `report/report.tex`).
3. **Slides PDF**.

Keep `contribution_report.md` in the repo root (checked at the demo).

## The two tasks
- **Task A — Regression:** fit a tree to `log1p(price)` and convert predictions
  back to AZN for RMSE, MAE and R².
- **Task B — Classification:** predict a derived **price tier**
  (premium vs standard, median split on the training set).

## Golden rule
Your **decision tree** and your **SVM** must be **your own NumPy code**.
scikit-learn is for the comparison baselines (and the optional bonuses)
only. For the SVM you do **not** need a QP/SMO solver — use the
**Pegasos** sub-gradient method (see the statement and `src/svm.py`).

## Repo layout
```
.
├── README.md               # this file
├── requirements.txt        # pinned environment used for the results
├── .gitignore
├── contribution_report.md  # who did what (keep in repo root)
├── data/                   # dataset lives here (not committed — see data/README.md)
├── src/                    # YOUR from-scratch code
│   ├── data_prep.py        #   loading, cleaning, split, tier label
│   ├── decision_tree.py    #   DecisionTree (Gini/entropy/MSE)
│   ├── svm.py              #   PegasosSVM (hinge-loss sub-gradient)
│   ├── baselines.py        #   scikit-learn reference models
│   ├── evaluate.py         #   NumPy metrics + comparison table
│   └── run_all.py          #   ONE command reproduces every number/figure
├── tests/                  # metrics and integration checks
├── report/                 # IEEE report (self-contained, no IEEEtran.cls)
│   ├── report.tex
│   └── figures/
└── presentation/           # your slides PDF
```

## Reproduce the results

Use Python 3.9–3.12. Download Kaggle dataset version 1 and extract
`house_sale.csv` to `data/` as described in [data/README.md](data/README.md).
The CSV and archive are ignored by Git; do not commit either.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.run_all
```

The last command performs cleaning, a seeded 70/15/15 split, validation-only
model selection, three-fold cross-validation on the training rows, and one
final evaluation on the held-out test rows. It writes full metrics and
selected settings to `results/summary.json` and five figures to
`report/figures/`. Both locations are generated locally and ignored by Git.
Use `python -m src.run_all --quick` for a 3,000-row smoke test; it writes to
`results/summary_quick.json` and `report/figures/quick/` so it cannot replace
the full results. Run
`python -m unittest discover -s tests -v` for the integration checks.

The premium threshold is the training-price median; validation and test
labels use that fixed threshold. The holdout rows are stratified by this
label. Missing feature values use training medians, and the NumPy SVM uses
training means and standard deviations. Each CV fold refits those statistics
on its own training portion. Regression uses log prices to reduce the effect
of the heavy price tail; no test prices are clipped or removed.

## Current full-dataset result

The 15 October 2026 run used 64,454 deduplicated listings and five numeric
features (`area_m2`, `floor_number`, `total_floors`, `lat`, `lng`). The split
has 45,118 training, 9,667 validation and 9,669 test rows. The premium
threshold is 210,000 AZN. All settings were selected by validation F1 or
log-RMSE before test evaluation.

| Model | Held-out test metric | Three-fold training CV |
|---|---:|---:|
| Our tree regressor | MAE 105,951 AZN; RMSE 753,420 AZN; R² 0.238 | log-RMSE 0.4871 ± 0.0141 |
| scikit-learn tree regressor | MAE 105,931 AZN; RMSE 753,424 AZN; R² 0.238 | log-RMSE 0.4867 ± 0.0139 |
| Our tree classifier | F1 0.8978; ROC-AUC 0.9627 | F1 0.8892 ± 0.0008 |
| scikit-learn tree classifier | F1 0.8979; ROC-AUC 0.9630 | F1 0.8890 ± 0.0009 |
| Our linear Pegasos | F1 0.6989; ROC-AUC 0.6946 | F1 0.7025 ± 0.0066 |
| scikit-learn SGD hinge | F1 0.6701; ROC-AUC 0.6968 | F1 0.6891 ± 0.0220 |
| Our RFF Pegasos | F1 0.7352; ROC-AUC 0.7908 | F1 0.7454 ± 0.0035 |

The 85 test listings priced above the training 99th percentile (2.5 million
AZN) have an average absolute error of about 3.65 million AZN; this tail
drives much of the price RMSE. Fit and predict times are saved in the JSON
because they depend on the machine.
