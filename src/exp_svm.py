
import numpy as np

from sklearn.metrics import accuracy_score, f1_score

from src.svm import PegasosSVM


def evaluate_linear_svm(X_train, y_train, X_val, y_val):
    model = PegasosSVM(
        lambda_=1e-4,
        n_iters=20_000,
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

                results.append({
                    "lambda": lam,
                    "n_iters": n_iter,
                    "learning_rate": lr,
                    "accuracy": accuracy_score(y_val, y_pred),
                    "f1": f1_score(y_val, y_pred),
                    "objective": model.objective(X_train, y_train)
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
