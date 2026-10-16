"""Train-only scaling checks for the integrated pipeline."""

import unittest

import numpy as np

from src.data_prep import standardize


class StandardizeTests(unittest.TestCase):
    def test_holdout_does_not_change_train_statistics(self):
        train = np.array([[1.0, 4.0], [3.0, 4.0], [5.0, 4.0]])
        holdout = np.array([[1000.0, 4.0]])
        scaled_train, scaled_holdout = standardize(train, holdout)
        np.testing.assert_allclose(scaled_train.mean(axis=0), [0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(scaled_train.std(axis=0), [1.0, 0.0], atol=1e-12)
        self.assertGreater(scaled_holdout[0, 0], 100)
        self.assertEqual(scaled_holdout[0, 1], 0.0)


if __name__ == "__main__":
    unittest.main()
