
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
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
    iterations = list(range(step, max_iters + 1, step))
    objectives = []

    for n_iter in iterations:
        model = PegasosSVM(
            lambda_=lambda_,
            n_iters=n_iter,
            random_state=42,
            learning_rate="pegasos"
        )

        model.fit(X_train, y_train)

        loss = model.objective(X_train, y_train)
        objectives.append(loss)

    return iterations, objectives


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
