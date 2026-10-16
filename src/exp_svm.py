
import numpy as np
from pathlib import Path

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.svm import PegasosSVM, RFFPegasosSVM

import matplotlib.pyplot as plt


def evaluate_linear_svm(X_train, y_train, X_val, y_val):
    model = PegasosSVM(
        lambda_=0.01,
        n_iters=3 * len(X_train),
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    results = {
        "accuracy": accuracy,
        "f1": f1,
        "objective": model.objective(X_train, y_train)
    }

    return model, results


def analyze_linear_features(model, feature_names):
    """Rank linear weights for a model fitted on standardized features."""
    if model.w is None:
        raise ValueError("Call fit() before analyzing model coefficients")

    coefficients = np.asarray(model.w)
    if coefficients.ndim != 1 or len(feature_names) != coefficients.size:
        raise ValueError("feature_names must match the model coefficient count")

    return sorted(
        [
            {
                "feature": feature,
                "coefficient": float(coefficient),
                "abs_coefficient": float(abs(coefficient)),
            }
            for feature, coefficient in zip(feature_names, coefficients)
        ],
        key=lambda row: row["abs_coefficient"],
        reverse=True,
    )


def analyze_validation_errors(
    model, X_val_scaled, y_val, X_val_features, feature_names
):
    """Summarize validation errors; inputs must preserve the same row order."""
    feature_index = getattr(X_val_features, "index", None)
    for values in (X_val_scaled, y_val):
        values_index = getattr(values, "index", None)
        if (
            feature_index is not None
            and values_index is not None
            and not values_index.equals(feature_index)
        ):
            raise ValueError("Validation inputs must have matching sample order")

    X_scaled = np.asarray(X_val_scaled)
    y_true = np.asarray(y_val)
    if X_scaled.ndim != 2 or y_true.ndim != 1:
        raise ValueError("Validation features must be 2D and labels must be 1D")
    if X_scaled.shape[0] != len(y_true) or X_scaled.shape[0] != len(X_val_features):
        raise ValueError("Validation features and labels must have matching rows")

    y_pred = np.asarray(model.predict(X_scaled))
    decision_scores = np.asarray(model.decision_function(X_scaled))

    rows = X_val_features.loc[:, feature_names].copy()
    rows["actual"] = y_true
    rows["predicted"] = y_pred
    rows["decision_score"] = decision_scores

    false_positive_mask = (y_true == 0) & (y_pred == 1)
    false_negative_mask = (y_true == 1) & (y_pred == 0)
    correct_standard_mask = (y_true == 0) & (y_pred == 0)
    correct_premium_mask = (y_true == 1) & (y_pred == 1)
    false_positives = rows.iloc[np.flatnonzero(false_positive_mask)]
    false_negatives = rows.iloc[np.flatnonzero(false_negative_mask)]

    return {
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "feature_medians": {
            "false_positives": false_positives[feature_names].median().to_dict(),
            "correct_standard": rows.loc[
                correct_standard_mask, feature_names
            ].median().to_dict(),
            "false_negatives": false_negatives[feature_names].median().to_dict(),
            "correct_premium": rows.loc[
                correct_premium_mask, feature_names
            ].median().to_dict(),
        },
    }


def compare_linear_rff_performance(
    X_train,
    y_train,
    X_val_scaled,
    y_val,
    X_val_features,
    feature_names,
):
    linear_model, linear_metrics = evaluate_linear_svm(
        X_train, y_train, X_val_scaled, y_val
    )
    linear_errors = analyze_validation_errors(
        linear_model, X_val_scaled, y_val, X_val_features, feature_names
    )

    rff_model = RFFPegasosSVM(
        gamma=1,
        lambda_=0.0001,
        n_components=500,
        n_iters=20000,
        random_state=42,
    )
    rff_model.fit(X_train, y_train)
    rff_predictions = rff_model.predict(X_val_scaled)
    rff_errors = analyze_validation_errors(
        rff_model, X_val_scaled, y_val, X_val_features, feature_names
    )
    rff_metrics = {
        "accuracy": accuracy_score(y_val, rff_predictions),
        "f1": f1_score(y_val, rff_predictions),
    }

    linear_color = "tab:blue"
    rff_color = "tab:orange"
    figure_dir = Path("report/figures")
    figure_dir.mkdir(parents=True, exist_ok=True)

    metric_names = ["Accuracy", "F1"]
    positions = np.arange(len(metric_names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8, 5))
    linear_bars = axis.bar(
        positions - width / 2,
        [linear_metrics["accuracy"], linear_metrics["f1"]],
        width,
        color=linear_color,
        label="Linear Pegasos",
    )
    rff_bars = axis.bar(
        positions + width / 2,
        [rff_metrics["accuracy"], rff_metrics["f1"]],
        width,
        color=rff_color,
        label="RFF Pegasos",
    )
    axis.bar_label(linear_bars, fmt="%.3f", padding=3)
    axis.bar_label(rff_bars, fmt="%.3f", padding=3)
    axis.set_xticks(positions, metric_names)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Validation score")
    axis.set_title("Linear Pegasos vs. RFF Pegasos")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figure_dir / "svm_linear_vs_rff.png", dpi=150)
    plt.show()
    plt.close(figure)

    confusion = rff_errors["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion, cmap="Oranges")
    axis.set_xticks([0, 1], ["Standard (0)", "Premium (1)"])
    axis.set_yticks([0, 1], ["Standard (0)", "Premium (1)"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title("RFF Pegasos Validation Confusion Matrix")
    threshold = confusion.max() / 2
    for actual in range(2):
        for predicted in range(2):
            axis.text(
                predicted,
                actual,
                str(confusion[actual, predicted]),
                ha="center",
                va="center",
                color="white" if confusion[actual, predicted] > threshold else "black",
            )
    figure.colorbar(image, ax=axis, label="Validation sample count")
    figure.tight_layout()
    figure.savefig(figure_dir / "svm_rff_confusion_matrix.png", dpi=150)
    plt.show()
    plt.close(figure)

    confusion = linear_errors["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion, cmap="Blues")
    axis.set_xticks([0, 1], ["Standard (0)", "Premium (1)"])
    axis.set_yticks([0, 1], ["Standard (0)", "Premium (1)"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    axis.set_title("Linear Pegasos Validation Confusion Matrix")
    threshold = confusion.max() / 2
    for actual in range(2):
        for predicted in range(2):
            axis.text(
                predicted,
                actual,
                str(confusion[actual, predicted]),
                ha="center",
                va="center",
                color="white" if confusion[actual, predicted] > threshold else "black",
            )
    figure.colorbar(image, ax=axis, label="Validation sample count")
    figure.tight_layout()
    figure.savefig(figure_dir / "svm_linear_confusion_matrix.png", dpi=150)
    plt.show()
    plt.close(figure)

    error_names = ["False positives", "False negatives"]
    positions = np.arange(len(error_names))
    linear_error_counts = [
        len(linear_errors["false_positives"]),
        len(linear_errors["false_negatives"]),
    ]
    rff_error_counts = [
        len(rff_errors["false_positives"]),
        len(rff_errors["false_negatives"]),
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    linear_bars = axis.bar(
        positions - width / 2,
        linear_error_counts,
        width,
        color=linear_color,
        label="Linear Pegasos",
    )
    rff_bars = axis.bar(
        positions + width / 2,
        rff_error_counts,
        width,
        color=rff_color,
        label="RFF Pegasos",
    )
    axis.bar_label(linear_bars, fmt="%d", padding=3)
    axis.bar_label(rff_bars, fmt="%d", padding=3)
    axis.set_xticks(positions, error_names)
    axis.set_ylabel("Validation sample count")
    axis.set_title("Validation Errors by Model")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "svm_error_comparison.png", dpi=150)
    plt.show()
    plt.close(figure)

    return {
        "linear_pegasos": {
            "accuracy": float(linear_metrics["accuracy"]),
            "f1": float(linear_metrics["f1"]),
            "confusion_matrix": linear_errors["confusion_matrix"].tolist(),
            "false_positives": len(linear_errors["false_positives"]),
            "false_negatives": len(linear_errors["false_negatives"]),
        },
        "rff_pegasos": {
            "accuracy": float(rff_metrics["accuracy"]),
            "f1": float(rff_metrics["f1"]),
            "confusion_matrix": rff_errors["confusion_matrix"].tolist(),
            "false_positives": len(rff_errors["false_positives"]),
            "false_negatives": len(rff_errors["false_negatives"]),
        },
    }


def evaluate_sklearn_baselines(X_train, y_train, X_val, y_val):
    models = {
        "linear_svc": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="linear", random_state=42)),
        ]),
        "sgd_hinge": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SGDClassifier(loss="hinge", random_state=42)),
        ]),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        results[name] = {
            "model": model,
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
        }

    return results


def compare_hyperparameters(X_train, y_train, X_val, y_val):
    results = []

    lambdas = [1e-2, 1e-3, 1e-4]
    iterations = [1000, 5000, 20000]
    learning_rates = ["pegasos", "sqrt"]

    for lam in lambdas:
        for n_iter in iterations:
            for lr in learning_rates:
                model = PegasosSVM(
                    lambda_=lam,
                    n_iters=n_iter,
                    random_state=42,
                    learning_rate=lr
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                margin = analyze_margin(model, X_train, y_train)

                results.append({
                    "lambda": lam,
                    "n_iters": n_iter,
                    "learning_rate": lr,
                    "accuracy": accuracy_score(y_val, y_pred),
                    "f1": f1_score(y_val, y_pred),
                    "objective": model.objective(X_train, y_train),
                    "weight_norm": margin["weight_norm"],
                    "margin_width": margin["margin_width"],
                })

    return results


def analyze_margin(model, X_train, y_train):
    y_pm1 = np.where(np.asarray(y_train) == 0, -1, 1)

    weight_norm = np.linalg.norm(model.w)

    margin_width = (
        2.0 / weight_norm
        if weight_norm > 0
        else float("inf")
    )

    scores = model.decision_function(X_train)
    margins = y_pm1 * scores

    support_indices = np.where(margins <= 1)[0]

    return {
        "weight_norm": weight_norm,
        "margin_width": margin_width,
        "support_indices": support_indices,
        "n_support_vectors": len(support_indices)
    }


def compare_lambda_margin_performance(X_train, y_train, X_val, y_val):
    lambdas = [0.01, 0.001, 0.0001]
    results = []

    for lam in lambdas:
        model = PegasosSVM(
            lambda_=lam,
            n_iters=3 * len(X_train),
            learning_rate="pegasos",
            random_state=42
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        margin = analyze_margin(model, X_train, y_train)

        results.append({
            "lambda": lam,
            "margin_width": margin["margin_width"],
            "accuracy": accuracy_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
        })

    margin_ax = plt.figure(figsize=(8, 5)).subplots()
    accuracy_ax = margin_ax.twinx()

    margin_ax.plot(
        [result["lambda"] for result in results],
        [result["margin_width"] for result in results],
        marker="o",
        label="Margin width"
    )
    accuracy_ax.plot(
        [result["lambda"] for result in results],
        [result["accuracy"] for result in results],
        marker="s",
        color="tab:orange",
        label="Validation accuracy"
    )

    margin_ax.set_xscale("log")
    margin_ax.invert_xaxis()
    margin_ax.set_xlabel("Lambda")
    margin_ax.set_ylabel("Margin width")
    accuracy_ax.set_ylabel("Validation accuracy")
    margin_ax.set_title("Lambda vs. Margin Width and Validation Accuracy")
    margin_ax.grid(True)
    margin_ax.legend(loc="upper left")
    accuracy_ax.legend(loc="upper right")

    margin_ax.figure.tight_layout()
    margin_ax.figure.savefig("report/figures/svm_margin_lambda.png", dpi=150)
    plt.show()

    return results


def analyze_convergence(
    X_train,
    y_train,
    lambda_=1e-4,
    max_iters=20000,
    step=1000
):
    model = PegasosSVM(
        lambda_=lambda_,
        n_iters=max_iters,
        random_state=42,
        learning_rate="pegasos"
    )
    model.fit(X_train, y_train, history_interval=step)

    iterations, objectives = zip(*model.loss_history_)
    return list(iterations), list(objectives)


def plot_convergence(iterations, objectives):
    plt.figure(figsize=(8, 5))

    plt.plot(iterations, objectives, marker="o")

    plt.xlabel("Number of iterations")
    plt.ylabel("Training objective J(w, b)")
    plt.title("Pegasos SVM Convergence")

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("report/figures/svm_convergence.png", dpi=150)
    plt.show()


def compare_rff_hyperparameters(X_train, y_train, X_val, y_val):
    results = []

    gammas = [0.01, 0.1, 1.0]
    lambdas = [1e-2, 1e-3, 1e-4]

    for gamma in gammas:
        for lam in lambdas:
            model = RFFPegasosSVM(
                gamma=gamma,
                n_components=500,
                lambda_=lam,
                n_iters=20000,
                random_state=42
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            results.append({
                "gamma": gamma,
                "lambda": lam,
                "accuracy": accuracy_score(y_val, y_pred),
                "f1": f1_score(y_val, y_pred),
                "objective": model.objective(X_train, y_train)
            })

    return results
