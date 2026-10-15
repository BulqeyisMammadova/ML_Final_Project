"""Checks for the split and preprocessing boundaries used by the runner."""

import unittest

import numpy as np
import pandas as pd

from src.run_all import _impute, _split


class RunnerBoundaryTests(unittest.TestCase):
    def test_holdouts_are_disjoint_and_stratified_after_train_median(self):
        features = pd.DataFrame({"area_m2": np.arange(200, dtype=float)})
        price = pd.Series(np.arange(1, 201, dtype=float) * 1000)
        train_x, train_y, val_x, val_y, test_x, test_y, threshold = _split(
            features, price
        )
        self.assertEqual((len(train_y), len(val_y) + len(test_y)), (140, 60))
        self.assertEqual(threshold, float(train_y.median()))
        self.assertFalse(set(train_x.index) & set(val_x.index))
        self.assertFalse(set(train_x.index) & set(test_x.index))
        self.assertFalse(set(val_x.index) & set(test_x.index))
        for tier in (0, 1):
            val_count = int(((val_y > threshold) == tier).sum())
            test_count = int(((test_y > threshold) == tier).sum())
            self.assertLessEqual(abs(val_count - test_count), 1)

    def test_holdout_missing_value_uses_training_median(self):
        train = pd.DataFrame({"area_m2": [10.0, 30.0]})
        holdout = pd.DataFrame({"area_m2": [np.nan, 1000.0]})
        scaled_train, scaled_holdout = _impute(train, holdout)
        np.testing.assert_array_equal(scaled_train[:, 0], [10.0, 30.0])
        np.testing.assert_array_equal(scaled_holdout[:, 0], [20.0, 1000.0])


if __name__ == "__main__":
    unittest.main()
