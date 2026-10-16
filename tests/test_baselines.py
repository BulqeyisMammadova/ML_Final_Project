"""Smoke tests for the comparison models and their scaling pipelines."""

import unittest

import numpy as np
from sklearn.pipeline import Pipeline

from src import baselines


class BaselineTests(unittest.TestCase):
    def test_models_fit_on_small_dataset(self):
        x = np.array([[1, 1], [2, 1], [3, 2], [7, 6], [8, 7], [9, 8]], dtype=float)
        price = np.array([10, 12, 15, 50, 55, 60], dtype=float)
        tier = np.array([0, 0, 0, 1, 1, 1])
        self.assertEqual(baselines.tree_regressor(max_depth=2).fit(x, price).predict(x).shape,
                         price.shape)
        self.assertEqual(baselines.tree_classifier(max_depth=2).fit(x, tier).predict(x).shape,
                         tier.shape)
        for model in (baselines.linear_svm(), baselines.rbf_svm()):
            self.assertIsInstance(model, Pipeline)
            self.assertEqual(model.fit(x, tier).predict(x).shape, tier.shape)


if __name__ == "__main__":
    unittest.main()
