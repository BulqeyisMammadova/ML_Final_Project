"""Check our NumPy metrics against independent scikit-learn references."""

import unittest

import numpy as np
from sklearn import metrics

from src import evaluate


class EvaluateTests(unittest.TestCase):
    def test_regression_matches_sklearn(self):
        actual = np.array([10.0, 25.0, 40.0, 60.0])
        predicted = np.array([12.0, 20.0, 38.0, 67.0])
        self.assertAlmostEqual(evaluate.rmse(actual, predicted),
                               np.sqrt(metrics.mean_squared_error(actual, predicted)))
        self.assertAlmostEqual(evaluate.mae(actual, predicted),
                               metrics.mean_absolute_error(actual, predicted))
        self.assertAlmostEqual(evaluate.r2(actual, predicted),
                               metrics.r2_score(actual, predicted))

    def test_classification_matches_sklearn_with_tied_scores(self):
        actual = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        predicted = np.array([0, 1, 0, 1, 1, 0, 0, 0])
        scores = np.array([0.2, 0.8, 0.5, 0.5, 0.9, 0.2, 0.5, 0.1])
        np.testing.assert_array_equal(evaluate.confusion_matrix(actual, predicted),
                                      metrics.confusion_matrix(actual, predicted, labels=[0, 1]))
        ours = evaluate.precision_recall_f1(actual, predicted)
        reference = (metrics.precision_score(actual, predicted),
                     metrics.recall_score(actual, predicted),
                     metrics.f1_score(actual, predicted))
        for computed, expected in zip(ours, reference):
            self.assertAlmostEqual(computed, expected)
        self.assertAlmostEqual(evaluate.roc_auc(actual, scores),
                               metrics.roc_auc_score(actual, scores))

    def test_zero_division_and_constant_target(self):
        self.assertEqual(evaluate.precision_recall_f1([0, 0], [0, 0]), (0.0, 0.0, 0.0))
        self.assertEqual(evaluate.r2([5, 5], [5, 5]), 1.0)
        self.assertEqual(evaluate.r2([5, 5], [5, 4]), 0.0)
        with self.assertRaises(ValueError):
            evaluate.roc_auc([1, 1], [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
