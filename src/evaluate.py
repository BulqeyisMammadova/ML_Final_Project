"""NumPy metrics for the price and price-tier tasks.

The positive class is premium (1). scikit-learn is only used as a reference
in the tests and for the comparison models.
"""

from __future__ import annotations

import numpy as np


def _paired_arrays(y_true, values, *, numeric: bool = False):
    actual = np.asarray(y_true)
    predicted = np.asarray(values)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Inputs must be one-dimensional")
    if actual.size == 0 or actual.size != predicted.size:
        raise ValueError("Inputs must have the same nonzero length")
    if numeric:
        actual = actual.astype(float)
        predicted = predicted.astype(float)
        if not (np.isfinite(actual).all() and np.isfinite(predicted).all()):
            raise ValueError("Inputs must contain only finite numbers")
    return actual, predicted


def _binary_labels(y):
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Price-tier labels must be 0 (standard) or 1 (premium)")


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    actual, predicted = _paired_arrays(y_true, y_pred, numeric=True)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(y_true, y_pred) -> float:
    actual, predicted = _paired_arrays(y_true, y_pred, numeric=True)
    return float(np.mean(np.abs(actual - predicted)))


def r2(y_true, y_pred) -> float:
    actual, predicted = _paired_arrays(y_true, y_pred, numeric=True)
    if actual.size < 2:
        return float("nan")  # R² is undefined with one observation.
    residual = float(np.sum((actual - predicted) ** 2))
    total = float(np.sum((actual - np.mean(actual)) ** 2))
    if total == 0:
        return 1.0 if residual == 0 else 0.0
    return 1.0 - residual / total


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return [[TN, FP], [FN, TP]], including absent classes as zero rows."""
    actual, predicted = _paired_arrays(y_true, y_pred)
    _binary_labels(actual)
    _binary_labels(predicted)
    counts = np.bincount(2 * actual.astype(int) + predicted.astype(int), minlength=4)
    return counts.reshape(2, 2)


def precision_recall_f1(y_true, y_pred):
    """Return precision, recall and F1 for premium; zero for undefined ratios."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def roc_auc(y_true, scores) -> float:
    """Compute ROC-AUC from scores using average ranks for ties."""
    actual, score = _paired_arrays(y_true, scores, numeric=True)
    _binary_labels(actual)
    positives = int(np.count_nonzero(actual == 1))
    negatives = actual.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both price-tier classes")

    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    sorted_labels = actual[order]
    positive_rank_sum = 0.0
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0  # 1-based ranks
        positive_rank_sum += average_rank * np.count_nonzero(sorted_labels[start:stop] == 1)
        start = stop
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


# --- Comparison helper ------------------------------------------------------
def compare(name_to_metrics: dict) -> None:
    """Print a compact comparison table; (mean, std) values show uncertainty."""
    if not name_to_metrics:
        raise ValueError("At least one model result is required")
    columns = list(dict.fromkeys(key for row in name_to_metrics.values() for key in row))

    def display(value):
        if value is None:
            return "n/a"
        if isinstance(value, tuple) and len(value) == 2:
            return f"{value[0]:.4f} ± {value[1]:.4f}"
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        return str(value)

    rows = [[name] + [display(metrics.get(column)) for column in columns]
            for name, metrics in name_to_metrics.items()]
    headings = ["Model", *columns]
    widths = [max(len(str(cell)) for cell in column) for column in zip(headings, *rows)]
    print("  ".join(str(cell).ljust(width) for cell, width in zip(headings, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)))
