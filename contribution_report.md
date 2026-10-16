# Contribution report

This record describes work visible in Git history as of 16 October 2026.
The team should confirm members' full names and agree on approximate effort
percentages before the final submission. Commit counts alone are not effort.

| Member / Git author | GitHub handle | Main responsibilities evidenced in the repo |
|---|---|---|
| nhesen (full name to confirm) | [nhesen](https://github.com/nhesen) | Data loading, cleaning, feature engineering, EDA and data-quality analysis |
| Bulqeyis Mammadova (name spelling to confirm) | [BulqeyisMammadova](https://github.com/BulqeyisMammadova) | Decision tree, tree experiments and from-scratch random forest |
| Ulvi (full name to confirm) | [Ulvi022](https://github.com/Ulvi022) | Pegasos SVM, random Fourier features, SVM experiments and tests |
| Bakir Gahramanov | [daduu16](https://github.com/daduu16) | Integration runner, evaluation metrics, sklearn baselines, cross-validation, results, report synthesis and results slides |

## Work summaries

**nhesen:** Implemented data preparation in `src/data_prep.py`, including
listing deduplication, area and floor parsing, train-only preprocessing,
property-type features and tier labels. Added EDA scripts and documented
data-quality and outlier investigations in `data_quality_notes.md`.

**Bulqeyis Mammadova:** Implemented the NumPy decision tree in
`src/decision_tree.py` and its real-data experiments in `src/exp_tree.py`.
Added feature importance and a from-scratch random forest bonus in
`src/random_forest.py`.

**Ulvi:** Implemented the linear Pegasos and RFF extension in `src/svm.py`.
Added `src/exp_svm.py`, SVM figures and `tests/test_svm.py` for convergence,
margin, error analysis and sklearn comparison.

**Bakir Gahramanov:** Implemented `src/evaluate.py`, `src/baselines.py` and
`src/run_all.py`, including validation-only tuning, training-fold CV,
held-out comparison, timing and generated figures. Added integration tests,
reconciled the latest feature pipeline with training-only categorical
encoding, documented the final benchmark in `README.md` and
`report/report.tex`, and prepared two editable results slides with a PDF
export in `presentation/`.

## Approximate effort split

Pending team agreement. Enter four percentages that sum to 100% after
reviewing the work and discussing time spent; do not derive effort from
commit counts or invent values.
