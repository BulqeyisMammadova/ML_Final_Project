"""
decision_tree.py — a decision tree built FROM SCRATCH (NumPy only).

Must support BOTH tasks:
  - classification (criterion = "gini" or "entropy")  -> price tier
  - regression     (criterion = "mse")                -> price

scikit-learn is NOT allowed here. This is your own code.

Suggested design:
  - a small Node (feature index, threshold, left, right, value/leaf_value),
  - recursive _grow with a best-split search over features & thresholds,
  - impurity functions (gini / entropy / mse) and the impurity-decrease
    objective used to pick splits,
  - hyperparameters that control overfitting (below).
"""

from __future__ import annotations

import numpy as np

def gini(y) -> float:
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities ** 2)

def entropy(y) -> float:
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities + 1e-9))  

def mse(y) -> float:
    if len(y) == 0:
        return 0.0
    y = np.asarray(y, dtype=float)
    return np.mean((y - y.mean()) ** 2)


class Node:
    """One node of the tree. Fill in the fields you need."""
    def __init__(self):
        self.feature = None      # int: feature index to split on (internal)
        self.threshold = None    # float: split threshold (internal)
        self.left = None         # Node
        self.right = None        # Node
        self.value = None        # leaf prediction (class label/proba or mean)

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


class DecisionTree:
    def __init__(
        self,
        task: str = "classification",   # "classification" | "regression"
        criterion: str = "gini",        # "gini" | "entropy" | "mse"
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.root: Node | None = None

    # --- impurity -----------------------------------------------------------
    def _impurity(self, y) -> float:
        if self.criterion == "gini":
            return gini(y)
        if self.criterion == "entropy":
            return entropy(y)
        if self.criterion == "mse":
            return mse(y)
        raise ValueError(f"Unknown criterion: {self.criterion}")

    
    def _impurity_decrease(self, y, y_left, y_right) -> float:  
        n = len(y)
        n_left = len(y_left)
        n_right = len(y_right)
        child_impurity = (n_left / n) * self._impurity(y_left) \
                       + (n_right / n) * self._impurity(y_right)

        return self._impurity(y) - child_impurity

    
    def _best_split(self, X, y):
        """
        Search features & candidate thresholds; return the split that
        maximizes impurity decrease (feature, threshold, decrease), or
        None if no valid split improves things.
        """
        raise NotImplementedError("TODO: best split search")

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        """Recursively build the tree, respecting the stopping rules."""
        raise NotImplementedError("TODO: recursive grow")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.root = self._grow(X, y, depth=0)
        return self

    def _predict_one(self, x, node: Node):
        raise NotImplementedError("TODO: traverse to a leaf")

    def predict(self, X):
        raise NotImplementedError("TODO: predict")

    def predict_proba(self, X):
        """Classification only: return class probabilities from leaf counts."""
        raise NotImplementedError("TODO: predict_proba (classification)")
