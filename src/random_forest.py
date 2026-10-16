import numpy as np
from src.decision_tree import DecisionTree


class RandomForest:
    def __init__(self, task="regression", n_trees=50, max_features=2,
                 max_depth=None, min_samples_leaf=5, random_state=42):
        self.task = task
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n = len(y)
        rng = np.random.default_rng(self.random_state)
        if self.task == "classification":
            self.classes_ = np.unique(y)
        criterion = "mse" if self.task == "regression" else "gini"

        self.trees = []
        for t in range(self.n_trees):
            idx = rng.integers(0, n, size=n)          
            tree = DecisionTree(task=self.task, criterion=criterion,
                                max_depth=self.max_depth,
                                min_samples_leaf=self.min_samples_leaf,
                                max_features=self.max_features,
                                random_state=self.random_state + t)
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
        return self

    def predict(self, X):
        if self.task == "regression":
            all_preds = np.array([tree.predict(X) for tree in self.trees])
            return all_preds.mean(axis=0)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        all_proba = np.array([tree.predict_proba(X) for tree in self.trees])
        return all_proba.mean(axis=0)