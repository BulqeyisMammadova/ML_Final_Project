"""scikit-learn comparison models only; never the team's core models."""

from __future__ import annotations

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def tree_regressor(*, max_depth=None, min_samples_split=2,
                   min_samples_leaf=1, min_impurity_decrease=0.0, random_state=42):
    """Mirror the from-scratch regression tree's main settings."""
    return DecisionTreeRegressor(
        criterion="squared_error", max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease, random_state=random_state,
    )


def tree_classifier(*, criterion="gini", max_depth=None, min_samples_split=2,
                    min_samples_leaf=1, min_impurity_decrease=0.0, random_state=42):
    """Mirror the from-scratch classification tree's main settings."""
    return DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth,
        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
        min_impurity_decrease=min_impurity_decrease, random_state=random_state,
    )


def linear_svm(*, lambda_=1e-4, random_state=42):
    """Scaled hinge-loss baseline; alpha has the role of Pegasos lambda.

    SGDClassifier uses a different optimizer, so matching alpha does not
    imply identical predictions or training schedules.
    """
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(loss="hinge", alpha=lambda_, max_iter=2000, tol=1e-4,
                      random_state=random_state),
    )


def rbf_svm(*, C=1.0, gamma="scale"):
    """Scaled nonlinear reference for the team's random Fourier features."""
    return make_pipeline(StandardScaler(), SVC(C=C, gamma=gamma, kernel="rbf"))
