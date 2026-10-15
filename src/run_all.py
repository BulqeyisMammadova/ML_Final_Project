"""Reproduce the validation search, independent test, CV and figures.

Run from the repository root with ``python -m src.run_all``. The raw dataset
is local and ignored by Git; see data/README.md for its source.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from . import baselines, data_prep, evaluate
from .decision_tree import DecisionTree
from .svm import PegasosSVM, RFFPegasosSVM

SEED = data_prep.SEED
ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "report" / "figures"
RESULTS = ROOT / "results"


def _split(raw_features, price):
    """Reserve 70% for training, then stratify the two holdout halves.

    The premium threshold is calculated only after the training portion has
    been fixed. All models subsequently use these same row indices.
    """
    train_x, train_y, val_x, val_y, test_x, test_y = data_prep.train_val_test_split(
        raw_features, price, seed=SEED
    )
    hold_x = pd.concat([val_x, test_x])
    hold_y = pd.concat([val_y, test_y])
    threshold = float(np.median(train_y))
    hold_tier = (hold_y.to_numpy() > threshold).astype(int)
    rng = np.random.default_rng(SEED)
    val_positions = []
    test_positions = []
    for tier in (0, 1):
        positions = np.flatnonzero(hold_tier == tier)
        rng.shuffle(positions)
        midpoint = len(positions) // 2
        val_positions.extend(positions[:midpoint])
        test_positions.extend(positions[midpoint:])
    val_positions = np.sort(val_positions)
    test_positions = np.sort(test_positions)
    return (train_x, train_y, hold_x.iloc[val_positions], hold_y.iloc[val_positions],
            hold_x.iloc[test_positions], hold_y.iloc[test_positions], threshold)


def _impute(train, *others):
    """Use the fit split's medians; never learn preprocessing from evaluation rows."""
    medians = train.median(numeric_only=True).fillna(0.0)
    return tuple(frame.fillna(medians).to_numpy(dtype=float) for frame in (train, *others))


def _score(model, features):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features), dtype=float)
    probabilities = model.predict_proba(features)
    classes = np.asarray(model.classes_)
    return np.asarray(probabilities[:, np.flatnonzero(classes == 1)[0]], dtype=float)


def _regression_metrics(actual_price, predicted_log):
    predicted_price = np.maximum(0.0, np.expm1(predicted_log))
    return {
        "RMSE_AZN": evaluate.rmse(actual_price, predicted_price),
        "MAE_AZN": evaluate.mae(actual_price, predicted_price),
        "R2": evaluate.r2(actual_price, predicted_price),
        "log_RMSE": evaluate.rmse(np.log1p(actual_price), predicted_log),
    }


def _classification_metrics(actual_tier, predicted_tier, scores):
    precision, recall, f1 = evaluate.precision_recall_f1(actual_tier, predicted_tier)
    return {
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "ROC_AUC": evaluate.roc_auc(actual_tier, scores),
    }


def _fit_and_measure(spec, train_x, train_price, eval_x, eval_price, threshold):
    if spec["scaled"]:
        train_x, eval_x = data_prep.standardize(train_x, eval_x)
    if spec["task"] == "regression":
        train_y = np.log1p(train_price)
    else:
        train_y = (train_price > threshold).astype(int)

    model = spec["factory"]()
    start = perf_counter()
    model.fit(train_x, train_y)
    fit_seconds = perf_counter() - start
    start = perf_counter()
    predictions = model.predict(eval_x)
    scores = _score(model, eval_x) if spec["task"] == "classification" else None
    predict_seconds = perf_counter() - start
    if spec["task"] == "regression":
        metrics = _regression_metrics(eval_price, predictions)
    else:
        actual_tier = (eval_price > threshold).astype(int)
        metrics = _classification_metrics(actual_tier, predictions, scores)
        metrics["confusion_matrix"] = evaluate.confusion_matrix(
            actual_tier, predictions
        ).tolist()
    metrics["fit_seconds"] = fit_seconds
    metrics["predict_seconds"] = predict_seconds
    return model, metrics, predictions


def _cross_validate(spec, raw_train_x, train_price, folds=3):
    """Estimate variation on training rows only, refitting imputer and scaler."""
    price = np.asarray(train_price, dtype=float)
    outer_tier = (price > np.median(price)).astype(int)
    if spec["task"] == "classification":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        indices = splitter.split(raw_train_x, outer_tier)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=SEED)
        indices = splitter.split(raw_train_x)

    fold_metrics = []
    for fit_index, check_index in indices:
        fold_train_x, fold_check_x = _impute(
            raw_train_x.iloc[fit_index], raw_train_x.iloc[check_index]
        )
        fold_train_y, fold_check_y = price[fit_index], price[check_index]
        fold_threshold = float(np.median(fold_train_y))
        _, metrics, _ = _fit_and_measure(
            spec, fold_train_x, fold_train_y, fold_check_x, fold_check_y, fold_threshold
        )
        metrics.pop("confusion_matrix", None)
        fold_metrics.append(metrics)
    return {
        key: {"mean": float(np.mean([row[key] for row in fold_metrics])),
              "std": float(np.std([row[key] for row in fold_metrics], ddof=1))}
        for key in fold_metrics[0]
    }


def _validation_search(train_x, train_price, val_x, val_price, threshold, quick):
    """Select configurations using validation only; return factories and curves."""
    log_train = np.log1p(train_price)
    log_val = np.log1p(val_price)
    tier_train = (train_price > threshold).astype(int)
    tier_val = (val_price > threshold).astype(int)
    scaled_train, scaled_val = data_prep.standardize(train_x, val_x)
    leaf_sizes = (20, 100) if not quick else (20,)
    depths = (4, 6, 8, 10, 12) if not quick else (4, 6)
    reg_tree_curve = []
    class_tree_curve = []

    best_reg = None
    best_cls = None
    for leaves in leaf_sizes:
        for depth in depths:
            params = {"max_depth": depth, "min_samples_leaf": leaves,
                      "random_state": SEED}
            reg = DecisionTree(task="regression", criterion="mse", **params)
            reg.fit(train_x, log_train)
            log_error = evaluate.rmse(log_val, reg.predict(val_x))
            reg_tree_curve.append({"max_depth": depth, "min_samples_leaf": leaves,
                                   "log_RMSE": log_error})
            if best_reg is None or log_error < best_reg[0]:
                best_reg = (log_error, params)
            for criterion in ("gini", "entropy"):
                cls = DecisionTree(task="classification", criterion=criterion, **params)
                cls.fit(train_x, tier_train)
                f1 = evaluate.precision_recall_f1(tier_val, cls.predict(val_x))[2]
                class_tree_curve.append({"max_depth": depth, "min_samples_leaf": leaves,
                                         "criterion": criterion, "F1": f1})
                if best_cls is None or f1 > best_cls[0]:
                    best_cls = (f1, {**params, "criterion": criterion})

    n_iters = 4000 if quick else 20000
    lambdas = (1e-3, 1e-4) if quick else (1e-2, 1e-3, 1e-4)
    linear_curve = []
    best_linear = None
    for lam in lambdas:
        svm = PegasosSVM(lambda_=lam, n_iters=n_iters, random_state=SEED)
        svm.fit(scaled_train, tier_train)
        f1 = evaluate.precision_recall_f1(tier_val, svm.predict(scaled_val))[2]
        linear_curve.append({"lambda": lam, "F1": f1,
                             "margin_width": float(2 / np.linalg.norm(svm.w))})
        if best_linear is None or f1 > best_linear[0]:
            best_linear = (f1, lam)

    rff_curve = []
    best_rff = None
    n_components = 64 if quick else 256
    rff_iters = 3000 if quick else 12000
    for gamma in (0.1, 1.0):
        for lam in lambdas:
            svm = RFFPegasosSVM(
                gamma=gamma, n_components=n_components, lambda_=lam,
                n_iters=rff_iters, random_state=SEED,
            )
            svm.fit(scaled_train, tier_train)
            f1 = evaluate.precision_recall_f1(tier_val, svm.predict(scaled_val))[2]
            rff_curve.append({"gamma": gamma, "lambda": lam, "F1": f1})
            if best_rff is None or f1 > best_rff[0]:
                best_rff = (f1, gamma, lam)

    reg_params = best_reg[1]
    cls_params = best_cls[1]
    linear_lambda = best_linear[1]
    rff_gamma, rff_lambda = best_rff[1:]
    specs = {
        "Ours tree regression": {
            "task": "regression", "scaled": False,
            "factory": lambda: DecisionTree(task="regression", criterion="mse", **reg_params),
            "config": reg_params,
        },
        "sklearn tree regression": {
            "task": "regression", "scaled": False,
            "factory": lambda: baselines.tree_regressor(**reg_params),
            "config": reg_params,
        },
        "Ours tree classification": {
            "task": "classification", "scaled": False,
            "factory": lambda: DecisionTree(task="classification", **cls_params),
            "config": cls_params,
        },
        "sklearn tree classification": {
            "task": "classification", "scaled": False,
            "factory": lambda: baselines.tree_classifier(**cls_params),
            "config": cls_params,
        },
        "Ours Pegasos linear": {
            "task": "classification", "scaled": True,
            "factory": lambda: PegasosSVM(lambda_=linear_lambda, n_iters=n_iters,
                                          random_state=SEED),
            "config": {"lambda": linear_lambda, "n_iters": n_iters},
        },
        "sklearn SGD hinge": {
            "task": "classification", "scaled": False,
            "factory": lambda: baselines.linear_svm(lambda_=linear_lambda, random_state=SEED),
            "config": {"alpha": linear_lambda},
        },
        "Ours Pegasos RFF": {
            "task": "classification", "scaled": True,
            "factory": lambda: RFFPegasosSVM(
                gamma=rff_gamma, n_components=n_components, lambda_=rff_lambda,
                n_iters=rff_iters, random_state=SEED,
            ),
            "config": {"gamma": rff_gamma, "lambda": rff_lambda,
                       "n_components": n_components, "n_iters": rff_iters},
        },
    }
    return specs, {"regression_tree": reg_tree_curve,
                   "classification_tree": class_tree_curve,
                   "linear_svm": linear_curve,
                   "rff_svm": rff_curve,
                   "selection_metric": {"regression": "validation log_RMSE (min)",
                                        "classification": "validation F1 (max)"}}


def _save_figures(tuning, test_results, regression_predictions, test_price,
                  objective_curve, figure_dir):
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for leaf in sorted({row["min_samples_leaf"] for row in tuning["regression_tree"]}):
        rows = [row for row in tuning["regression_tree"] if row["min_samples_leaf"] == leaf]
        axes[0].plot([row["max_depth"] for row in rows],
                     [row["log_RMSE"] for row in rows], marker="o", label=f"leaf={leaf}")
        for criterion in ("gini", "entropy"):
            rows = [row for row in tuning["classification_tree"]
                    if row["min_samples_leaf"] == leaf and row["criterion"] == criterion]
            axes[1].plot([row["max_depth"] for row in rows],
                         [row["F1"] for row in rows], marker="o",
                         label=f"{criterion}, leaf={leaf}")
    axes[0].set(title="Regression validation", xlabel="max_depth", ylabel="log RMSE")
    axes[1].set(title="Tier validation", xlabel="max_depth", ylabel="F1")
    axes[0].legend()
    axes[1].legend()
    axes[2].semilogx([row["lambda"] for row in tuning["linear_svm"]],
                     [row["F1"] for row in tuning["linear_svm"]], marker="o")
    axes[2].set(title="Linear Pegasos validation", xlabel="lambda", ylabel="F1")
    fig.tight_layout()
    fig.savefig(figure_dir / "validation_curves.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(3.6, 4.8))
    for leaf in sorted({row["min_samples_leaf"] for row in tuning["regression_tree"]}):
        rows = [row for row in tuning["regression_tree"] if row["min_samples_leaf"] == leaf]
        axes[0].plot([row["max_depth"] for row in rows],
                     [row["log_RMSE"] for row in rows], marker="o", label=f"leaf={leaf}")
        for criterion in ("gini", "entropy"):
            rows = [row for row in tuning["classification_tree"]
                    if row["min_samples_leaf"] == leaf and row["criterion"] == criterion]
            axes[1].plot([row["max_depth"] for row in rows],
                         [row["F1"] for row in rows], marker="o",
                         label=f"{criterion}, leaf={leaf}")
    axes[0].set(title="Regression validation", xlabel="max_depth", ylabel="log RMSE")
    axes[1].set(title="Tier validation", xlabel="max_depth", ylabel="F1")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_dir / "tree_validation_report.png", dpi=180)
    plt.close(fig)

    names = [name for name, row in test_results.items() if "confusion_matrix" in row]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    for ax, name in zip(axes.flat, names):
        matrix = np.asarray(test_results[name]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center")
        ax.set(title=name, xlabel="Predicted", ylabel="Actual",
               xticks=(0, 1), yticks=(0, 1))
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_dir / "test_confusion_matrices.png", dpi=160)
    plt.close(fig)

    predicted_price = np.maximum(1.0, np.expm1(regression_predictions))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(test_price, predicted_price, s=7, alpha=0.3)
    bounds = [max(1.0, float(np.min(test_price))), float(np.max(test_price))]
    ax.plot(bounds, bounds, color="black", linestyle="--", linewidth=1)
    ax.set(xscale="log", yscale="log", xlabel="Actual price (AZN)",
           ylabel="Predicted price (AZN)", title="Held-out regression errors")
    fig.tight_layout()
    fig.savefig(figure_dir / "regression_errors.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.semilogx([row["n_iters"] for row in objective_curve],
                [row["objective"] for row in objective_curve], marker="o")
    ax.set(xlabel="Pegasos updates", ylabel="Regularised hinge objective",
           title="Linear SVM training objective")
    fig.tight_layout()
    fig.savefig(figure_dir / "svm_objective.png", dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / data_prep.DATA_PATH),
                        help="Path to the Kaggle CSV")
    parser.add_argument("--quick", action="store_true",
                        help="Use 3,000 deterministic rows and a smaller search for a smoke test")
    args = parser.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.is_file():
        parser.error(f"Dataset missing: {data_path}. Follow data/README.md.")
    raw = data_prep.load_raw(str(data_path))
    currencies = sorted(set(raw["currency_x"].dropna()) | set(raw["currency_y"].dropna()))
    if currencies != ["AZN"]:
        raise ValueError(f"Expected AZN prices; found currencies: {currencies}")
    cleaned = data_prep.clean(raw)
    if args.quick:
        cleaned = cleaned.sample(n=min(3000, len(cleaned)), random_state=SEED)
    features, price, feature_names = data_prep.make_features(cleaned)
    valid_price = np.isfinite(price.to_numpy(dtype=float)) & (price.to_numpy(dtype=float) > 0)
    features, price = features.loc[valid_price], price.loc[valid_price]
    train_x_raw, train_y, val_x_raw, val_y, test_x_raw, test_y, threshold = _split(
        features, price
    )
    train_x, val_x, test_x = _impute(train_x_raw, val_x_raw, test_x_raw)
    train_price = train_y.to_numpy(dtype=float)
    val_price = val_y.to_numpy(dtype=float)
    test_price = test_y.to_numpy(dtype=float)

    specs, tuning = _validation_search(
        train_x, train_price, val_x, val_price, threshold, args.quick
    )
    test_results = {}
    cv_results = {}
    fitted = {}
    predictions = {}
    for name, spec in specs.items():
        model, metrics, predicted = _fit_and_measure(
            spec, train_x, train_price, test_x, test_price, threshold
        )
        fitted[name] = model
        predictions[name] = predicted
        test_results[name] = metrics
        cv_results[name] = _cross_validate(spec, train_x_raw, train_price)

    linear_spec = specs["Ours Pegasos linear"]
    scaled_train = data_prep.standardize(train_x)[0]
    train_tier = (train_price > threshold).astype(int)
    objective_steps = (100, 500, 1000, 2000, 4000) if args.quick else (
        100, 500, 1000, 2000, 5000, 10000, 20000
    )
    objective_curve = []
    for steps in objective_steps:
        checkpoint = PegasosSVM(
            lambda_=linear_spec["config"]["lambda"], n_iters=steps, random_state=SEED
        ).fit(scaled_train, train_tier)
        objective_curve.append({"n_iters": steps,
                                "objective": checkpoint.objective(scaled_train, train_tier)})
    linear_model = fitted["Ours Pegasos linear"]
    signed_labels = np.where(train_tier == 1, 1, -1)
    signed_margins = signed_labels * linear_model.decision_function(scaled_train)
    margin_width = float(2 / np.linalg.norm(linear_model.w))
    support_count = int(np.count_nonzero(signed_margins <= 1))

    price_p99 = float(np.quantile(train_price, 0.99))
    regression_price = np.maximum(0.0, np.expm1(predictions["Ours tree regression"]))
    absolute_error = np.abs(test_price - regression_price)
    central = test_price <= price_p99
    tail = ~central
    summary = {
        "seed": SEED,
        "quick": args.quick,
        "data_rows_raw": len(raw),
        "data_rows_after_cleaning": len(cleaned),
        "features": feature_names,
        "split_rows": {"train": len(train_y), "validation": len(val_y), "test": len(test_y)},
        "train_price_median_AZN": threshold,
        "train_price_p99_AZN": price_p99,
        "test_above_train_p99": int(np.count_nonzero(test_price > price_p99)),
        "test_price_max_AZN": float(np.max(test_price)),
        "tier_fraction": {
            "train": float(np.mean(train_price > threshold)),
            "validation": float(np.mean(val_price > threshold)),
            "test": float(np.mean(test_price > threshold)),
        },
        "tuning": tuning,
        "selected_configs": {name: spec["config"] for name, spec in specs.items()},
        "test": test_results,
        "train_cv_3fold": cv_results,
        "linear_svm_diagnostics": {
            "weight_norm": float(np.linalg.norm(linear_model.w)),
            "margin_width": margin_width,
            "support_vectors": support_count,
            "support_fraction": support_count / len(train_tier),
            "objective_curve": objective_curve,
        },
        "regression_error_analysis": {
            "test_median_absolute_error_AZN": float(np.median(absolute_error)),
            "test_at_or_below_train_p99_count": int(np.count_nonzero(central)),
            "test_at_or_below_train_p99_MAE_AZN": float(np.mean(absolute_error[central])),
            "test_above_train_p99_count": int(np.count_nonzero(tail)),
            "test_above_train_p99_MAE_AZN": float(np.mean(absolute_error[tail])),
            "test_above_train_p99_underpredicted_fraction": float(
                np.mean(regression_price[tail] < test_price[tail])
            ),
        },
        "tree_regression_importance": dict(zip(
            feature_names, fitted["Ours tree regression"].feature_importances_.tolist()
        )),
        "tree_classification_importance": dict(zip(
            feature_names, fitted["Ours tree classification"].feature_importances_.tolist()
        )),
        "linear_svm_weights": dict(zip(
            feature_names, fitted["Ours Pegasos linear"].w.tolist()
        )),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS / ("summary_quick.json" if args.quick else "summary.json")
    figure_dir = FIGURES / "quick" if args.quick else FIGURES
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
    _save_figures(tuning, test_results, predictions["Ours tree regression"],
                  test_price, objective_curve, figure_dir)

    print(f"Rows: train={len(train_y)}, validation={len(val_y)}, test={len(test_y)}")
    print(f"Price-tier threshold from training median: {threshold:,.0f} AZN")
    print("\nRegression: held-out test metrics and 3-fold CV on training rows")
    evaluate.compare({
        name: {"Test RMSE": row["RMSE_AZN"], "Test MAE": row["MAE_AZN"],
               "Test R2": row["R2"],
               "CV log RMSE": (cv_results[name]["log_RMSE"]["mean"],
                               cv_results[name]["log_RMSE"]["std"]),
               "Fit s": row["fit_seconds"]}
        for name, row in test_results.items() if specs[name]["task"] == "regression"
    })
    print("\nClassification: held-out test metrics and 3-fold CV on training rows")
    evaluate.compare({
        name: {"Test F1": row["F1"], "Test AUC": row["ROC_AUC"],
               "CV F1": (cv_results[name]["F1"]["mean"],
                         cv_results[name]["F1"]["std"]),
               "Fit s": row["fit_seconds"]}
        for name, row in test_results.items() if specs[name]["task"] == "classification"
    })
    print(f"\nSaved details: {result_path}")
    print(f"Saved figures: {figure_dir}")
    return summary


if __name__ == "__main__":
    main()
