"""
svm.py — soft-margin SVM from scratch using NumPy and Pegasos.

Minimise J(w, b) = (lambda/2) * ||w||^2
                   + mean(max(0, 1 - y * (X @ w + b))).

The linear model uses stochastic sub-gradient updates. Bias is updated
separately and is not regularised. Non-linear extension: Random Fourier
Features (RFF), approximating the RBF kernel before linear Pegasos.

Standardise features using training-set statistics before calling fit().
scikit-learn SVC belongs in the separate baseline experiments only.
"""

from __future__ import annotations

import numpy as np


class PegasosSVM:
    """Linear soft-margin SVM trained using stochastic Pegasos updates."""

    def __init__(
        self,
        lambda_: float = 1e-4,
        n_iters: int = 20_000,
        random_state: int = 42,
        learning_rate: str = "pegasos",
    ):
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.w = None
        self.b = 0.0
        self.loss_history_ = []

    @staticmethod
    def _to_pm1(y):
        """Map labels {0, 1} to {-1, +1}."""
        y = np.asarray(y)
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("Labels must be 0 or 1")
        return np.where(y == 0, -1.0, 1.0)

    def _eta(self, t):
        if self.learning_rate == "pegasos":
            return 1.0 / (self.lambda_ * t)
        if self.learning_rate == "sqrt":
            return 1.0 / (self.lambda_ * np.sqrt(t))
        raise ValueError("learning_rate must be 'pegasos' or 'sqrt'")

    def fit(self, X, y, history_interval=None):
        self.loss_history_ = []
        if history_interval is not None and (
            not isinstance(history_interval, (int, np.integer))
            or isinstance(history_interval, (bool, np.bool_))
            or history_interval <= 0
        ):
            raise ValueError("history_interval must be a positive integer or None")

        X = np.asarray(X, dtype=float)
        y_original = np.asarray(y)
        y = self._to_pm1(y_original)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[0] != len(y):
            raise ValueError("X must be a nonempty 2D array matching y")
        if self.lambda_ <= 0 or self.n_iters <= 0:
            raise ValueError("lambda_ and n_iters must be positive")
        if self.learning_rate not in ("pegasos", "sqrt"):
            raise ValueError("learning_rate must be 'pegasos' or 'sqrt'")

        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        if history_interval is not None:
            self.loss_history_.append((0, self.objective(X, y_original)))
        rng = np.random.default_rng(self.random_state)

        for t in range(1, self.n_iters + 1):
            i = rng.integers(0, n_samples)
            xi, yi = X[i], y[i]
            eta = self._eta(t)
            margin = yi * (np.dot(self.w, xi) + self.b)

            self.w *= 1.0 - eta * self.lambda_
            if margin < 1.0:
                self.w += eta * yi * xi
                self.b += eta * yi
            if history_interval is not None and (
                t % history_interval == 0 or t == self.n_iters
            ):
                self.loss_history_.append((t, self.objective(X, y_original)))
        return self

    def decision_function(self, X):
        if self.w is None:
            raise ValueError("Call fit() before decision_function()")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X @ self.w + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, 0)

    def objective(self, X, y):
        """Regularised hinge-loss objective J(w, b)."""
        y_pm1 = self._to_pm1(y)
        scores = self.decision_function(X)
        if scores.shape[0] != len(y_pm1):
            raise ValueError("X and y must have the same number of samples")
        hinge = np.maximum(0.0, 1.0 - y_pm1 * scores)
        return float(0.5 * self.lambda_ * (self.w @ self.w) + np.mean(hinge))


class RFFPegasosSVM:
    """Approximate RBF-kernel SVM via random Fourier features + Pegasos."""

    def __init__(
        self,
        gamma: float = 0.1,
        n_components: int = 500,
        lambda_: float = 1e-4,
        n_iters: int = 20_000,
        random_state: int = 42,
        learning_rate: str = "pegasos",
    ):
        self.gamma = gamma
        self.n_components = n_components
        self.lambda_ = lambda_
        self.n_iters = n_iters
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.W_ = None
        self.c_ = None
        self.linear_ = None

    def _transform(self, X):
        if self.W_ is None:
            raise ValueError("Call fit() before transforming features")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.sqrt(2.0 / self.n_components) * np.cos(X @ self.W_ + self.c_)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a nonempty 2D array")
        if self.gamma <= 0 or self.n_components <= 0:
            raise ValueError("gamma and n_components must be positive")
        rng = np.random.default_rng(self.random_state)
        self.W_ = rng.normal(
            0.0, np.sqrt(2.0 * self.gamma),
            size=(X.shape[1], self.n_components),
        )
        self.c_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        self.linear_ = PegasosSVM(
            lambda_=self.lambda_,
            n_iters=self.n_iters,
            random_state=self.random_state,
            learning_rate=self.learning_rate,
        ).fit(self._transform(X), y)
        return self

    def decision_function(self, X):
        if self.linear_ is None:
            raise ValueError("Call fit() before decision_function()")
        return self.linear_.decision_function(self._transform(X))

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, 0)

    def objective(self, X, y):
        if self.linear_ is None:
            raise ValueError("Call fit() before objective()")
        return self.linear_.objective(self._transform(X), y)
