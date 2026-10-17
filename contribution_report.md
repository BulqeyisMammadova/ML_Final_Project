# Contribution report

MLE-AI-201 Final Project · bina.az price models · final version, 17 October 2026.
Every responsibility below can be checked in the Git history of this repository.

## Team

| Member | Name | GitHub | Main responsibility |
|---|---|---|---|
| 1 | Hasan Nurmammadov | [nhesen](https://github.com/nhesen) | Data loading, cleaning, feature engineering, split, EDA, data-quality notes, geospatial bonus |
| 2 | Bulqeyis Mammadova | [BulqeyisMammadova](https://github.com/BulqeyisMammadova) | From-scratch decision tree, tree tests and experiments, random forest bonus, final report revision |
| 3 | Ulvi Ismayilzade | [Ulvi022](https://github.com/Ulvi022) | From-scratch Pegasos SVM, random Fourier features, SVM experiments, figures and tests |
| 4 | Bakir Gahramanov | [daduu16](https://github.com/daduu16) | Integration runner, metrics, scikit-learn baselines, cross-validation, integration tests, README, report and results slides |

## Work by member

**Member 1 · Hasan Nurmammadov**
- `src/data_prep.py`: loading, deduplication by listing URL (100,775 → 64,454 rows),
  area and floor parsing (including *sot* → m²), removal of leakage columns,
  train-only medians, one-hot property type, premium label and the shared split.
- EDA scripts (`eda_price.py`, `eda_area.py`, `eda_property_type.py`),
  `compare_features.py`, `test_split_manual.py` and `data_quality_notes.md`.
- Bonus (Problem 10): `geospatial_bonus.py`, k-means clusters on training
  coordinates and `geospatial_bonus_report.md`.

**Member 2 · Bulqeyis Mammadova**
- `src/decision_tree.py`: NumPy decision tree for both tasks (Gini, entropy, MSE),
  midpoint split search, fast sort + cumulative-sum split (~180× faster),
  four stopping hyperparameters, `predict_proba`, feature importance and the
  floating-point threshold fix.
- `src/test_tree.py`: 9 tests, including split-by-split equality with scikit-learn.
- `src/exp_tree.py`: depth / leaf-size / criterion studies, validation-only grid
  search, scikit-learn comparison, error analysis, one-time test evaluation and
  the tree figures in `report/figures/`.
- Bonus (Problem 9): `src/random_forest.py`, a 50-tree random forest built from
  our own trees (test R² 0.781 → 0.832, F1 0.908 → 0.934).
- Final report revision: `src/report_extras.py` and the added report sections
  and figures.

**Member 3 · Ulvi Ismayilzade**
- `src/svm.py`: linear soft-margin SVM trained with Pegasos (`PegasosSVM`) and the
  random Fourier feature extension (`RFFPegasosSVM`).
- `src/exp_svm.py`, `run_svm_validation.py`, `run_svm_figures.py`: convergence,
  λ / margin, learning-rate schedule and linear vs. RFF experiments, SVM figures.
- `tests/test_svm.py`.

**Member 4 · Bakir Gahramanov**
- `src/run_all.py`: one-command reproduction of all headline numbers and figures,
  validation-only tuning, stratified holdout, 3-fold training CV and timings.
- `src/evaluate.py` (NumPy metrics) and `src/baselines.py` (scikit-learn baselines).
- Integration tests in `tests/`, `README.md`, `report/report.tex` structure and
  results section, and the presentation slides in `presentation/`.

## Git evidence (all branches)

| Member | Commits | Main files |
|---|---|---|
| 1 · Hasan Nurmammadov | 27 | `src/data_prep.py`, EDA scripts, `geospatial_bonus.py` |
| 2 · Bulqeyis Mammadova | 30 | `src/decision_tree.py`, `src/exp_tree.py`, `src/test_tree.py`, `src/random_forest.py` |
| 3 · Ulvi Ismayilzade | 16 | `src/svm.py`, `src/exp_svm.py`, `tests/test_svm.py` |
| 4 · Bakir Gahramanov | 9 | `src/run_all.py`, `src/evaluate.py`, `src/baselines.py`, `tests/`, `README.md` |

Commit counts depend on how often each person committed, so they are not a
measure of effort on their own.

## Effort split (agreed by the team)

| Member | Effort |
|---|---|
| 1 · Hasan Nurmammadov | 25% |
| 2 · Bulqeyis Mammadova | 25% |
| 3 · Ulvi Ismayilzade | 25% |
| 4 · Bakir Gahramanov | 25% |

## Use of AI assistants

Each member used an AI assistant as a tutor or coding helper and can explain
every line they submitted. Member 2 used it to explain the tree concepts and to
propose code step by step; she typed, ran, tested and debugged the code herself.
Member 4 used it to draft and test the integration runner, metrics, baselines,
README and results text.
<!-- TODO before submission: Member 1 and Member 3 add one sentence each on how they used AI. -->
