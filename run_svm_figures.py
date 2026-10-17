"""Regenerate the six SVM figures using the current train/validation pipeline."""

import os
from time import perf_counter

import pandas as pd

# Avoid interactive windows by default; respect an explicitly selected backend.
os.environ.setdefault("MPLBACKEND", "Agg")

from src import data_prep
from src.exp_svm import (
    analyze_convergence,
    compare_lambda_margin_performance,
    compare_linear_rff_performance,
    plot_convergence,
)


FIGURE_PATHS = (
    "report/figures/svm_convergence.png",
    "report/figures/svm_margin_lambda.png",
    "report/figures/svm_linear_vs_rff.png",
    "report/figures/svm_rff_confusion_matrix.png",
    "report/figures/svm_linear_confusion_matrix.png",
    "report/figures/svm_error_comparison.png",
)


def run_timed_stage(name, details, action):
    print("\n[START] {}: {}".format(name, details), flush=True)
    started = perf_counter()
    try:
        result = action()
    except BaseException as error:
        elapsed = perf_counter() - started
        print(
            "[STOP] {} after {:.1f} seconds ({})".format(
                name, elapsed, type(error).__name__
            ),
            flush=True,
        )
        raise
    elapsed = perf_counter() - started
    print("[DONE] {} in {:.1f} seconds.".format(name, elapsed), flush=True)
    return result


def main():
    total_started = perf_counter()
    cleaned = data_prep.clean(data_prep.load_raw())
    X, y_price, _ = data_prep.make_features(cleaned)
    (
        X_train,
        y_train_price,
        X_val,
        y_val_price,
        _,
        _,
    ) = data_prep.train_val_test_split(X, y_price)

    X_train = pd.get_dummies(X_train, columns=["property_type"], dtype=int)
    X_val = pd.get_dummies(
        X_val, columns=["property_type"], dtype=int
    ).reindex(columns=X_train.columns, fill_value=0)

    X_train["land_area_m2"] = X_train["land_area_m2"].fillna(0)
    X_val["land_area_m2"] = X_val["land_area_m2"].fillna(0)
    train_medians = X_train.median()
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)

    y_train, threshold = data_prep.make_tier_label(y_train_price)
    y_val, _ = data_prep.make_tier_label(y_val_price, threshold)
    X_train_scaled, X_val_scaled = data_prep.standardize(X_train, X_val)
    feature_names = X_train.columns.tolist()

    print("Train samples: {}".format(len(X_train_scaled)))
    print("Validation samples: {}".format(len(X_val_scaled)))
    print("Training-price median threshold: {:.6f}".format(threshold))
    print("Feature count: {}".format(len(feature_names)))
    print("Features: {}".format(feature_names))
    print("Test partition is not used for fitting, selection, or figures.")
    print("Preprocessing complete in {:.1f} seconds.".format(
        perf_counter() - total_started
    ), flush=True)

    def generate_convergence_figure():
        iterations, objectives = analyze_convergence(
            X_train_scaled,
            y_train,
            lambda_=0.01,
            max_iters=20000,
            step=1000,
        )
        plot_convergence(iterations, objectives)

    run_timed_stage(
        "Convergence figure",
        "1 Pegasos fit, lambda=0.01, 20,000 updates; 21 full-training objective checkpoints",
        generate_convergence_figure,
    )

    lambda_values = [0.01, 0.001, 0.0001]
    lambda_updates_each = 3 * len(X_train_scaled)
    total_updates = (
        20000
        + len(lambda_values) * lambda_updates_each
        + lambda_updates_each
        + 20000
    )
    print(
        "Planned Pegasos update loops across all stages: {:,}. "
        "The lambda=0.01 linear model is fit again in the comparison helper.".format(
            total_updates
        ),
        flush=True,
    )
    run_timed_stage(
        "Lambda/margin figure",
        "3 Pegasos fits for {}; {:,} updates each, {:,} total".format(
            lambda_values, lambda_updates_each,
            len(lambda_values) * lambda_updates_each,
        ),
        lambda: compare_lambda_margin_performance(
            X_train_scaled, y_train, X_val_scaled, y_val
        ),
    )

    results = run_timed_stage(
        "Linear/RFF comparison and three figures",
        "Linear Pegasos: {:,} updates; RFF: 20,000 updates over 500 components".format(
            lambda_updates_each
        ),
        lambda: compare_linear_rff_performance(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            X_val,
            feature_names,
        ),
    )
    print("\nSelected-model validation results:")
    for model_name, metrics in results.items():
        print(
            "{}: accuracy={:.6f}, F1={:.6f}".format(
                model_name, metrics["accuracy"], metrics["f1"]
            )
        )

    print("\nGenerated SVM figures:")
    for figure_path in FIGURE_PATHS:
        print(figure_path)
    print("\nAll stages completed in {:.1f} seconds.".format(
        perf_counter() - total_started
    ), flush=True)


if __name__ == "__main__":
    main()
