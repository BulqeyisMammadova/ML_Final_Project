import unittest

import numpy as np

from src.svm import PegasosSVM, RFFPegasosSVM


class TestPegasosSVM(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [-2.0, -1.0],
            [-1.0, -2.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ])
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_fit_predict_and_binary_output(self):
        model = PegasosSVM(lambda_=0.01, n_iters=300, random_state=42)

        self.assertIs(model.fit(self.X, self.y), model)
        predictions = model.predict(self.X)

        self.assertEqual(predictions.shape, self.y.shape)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))

    def test_decision_function_shapes(self):
        model = PegasosSVM(lambda_=0.01, n_iters=100, random_state=42)
        model.fit(self.X, self.y)

        self.assertEqual(model.decision_function(self.X).shape, (len(self.X),))
        self.assertEqual(model.decision_function(self.X[0]).shape, (1,))

    def test_objective_matches_definition(self):
        model = PegasosSVM(lambda_=0.01, n_iters=100, random_state=42)
        model.fit(self.X, self.y)

        y_pm1 = np.where(self.y == 0, -1.0, 1.0)
        scores = self.X @ model.w + model.b
        hinge = np.maximum(0.0, 1.0 - y_pm1 * scores)
        expected = 0.5 * model.lambda_ * (model.w @ model.w) + np.mean(hinge)

        self.assertAlmostEqual(model.objective(self.X, self.y), expected)

    def test_reproducibility(self):
        first = PegasosSVM(lambda_=0.01, n_iters=100, random_state=42).fit(
            self.X, self.y
        )
        second = PegasosSVM(lambda_=0.01, n_iters=100, random_state=42).fit(
            self.X, self.y
        )

        np.testing.assert_allclose(first.w, second.w)
        self.assertEqual(first.b, second.b)
        np.testing.assert_array_equal(first.predict(self.X), second.predict(self.X))
        self.assertEqual(first.objective(self.X, self.y), second.objective(self.X, self.y))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            PegasosSVM().decision_function(self.X)
        with self.assertRaises(ValueError):
            PegasosSVM().fit(self.X, np.array([0, 1]))
        with self.assertRaises(ValueError):
            PegasosSVM().fit(self.X, np.array([0, 0, 0, 1, 1, 2]))
        with self.assertRaises(ValueError):
            PegasosSVM().fit(np.empty((0, 2)), np.array([], dtype=int))
        with self.assertRaises(ValueError):
            PegasosSVM(lambda_=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            PegasosSVM(n_iters=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            PegasosSVM(learning_rate="invalid").fit(self.X, self.y)


class TestRFFPegasosSVM(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [-2.0, -1.0],
            [-1.0, -2.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
        ])
        self.y = np.array([0, 0, 0, 1, 1, 1])

    def test_fit_predict_and_outputs(self):
        model = RFFPegasosSVM(
            gamma=0.1,
            n_components=20,
            lambda_=0.01,
            n_iters=100,
            random_state=42,
        )

        self.assertIs(model.fit(self.X, self.y), model)
        predictions = model.predict(self.X)
        scores = model.decision_function(self.X)

        self.assertEqual(predictions.shape, self.y.shape)
        self.assertTrue(np.all(np.isin(predictions, [0, 1])))
        self.assertEqual(scores.shape, (len(self.X),))
        self.assertTrue(np.isfinite(model.objective(self.X, self.y)))

    def test_reproducibility(self):
        first = RFFPegasosSVM(
            gamma=0.1,
            n_components=20,
            lambda_=0.01,
            n_iters=100,
            random_state=42,
        ).fit(self.X, self.y)
        second = RFFPegasosSVM(
            gamma=0.1,
            n_components=20,
            lambda_=0.01,
            n_iters=100,
            random_state=42,
        ).fit(self.X, self.y)

        np.testing.assert_allclose(first.W_, second.W_)
        np.testing.assert_allclose(first.c_, second.c_)
        np.testing.assert_allclose(first.linear_.w, second.linear_.w)
        self.assertEqual(first.linear_.b, second.linear_.b)
        np.testing.assert_array_equal(first.predict(self.X), second.predict(self.X))
        self.assertEqual(first.objective(self.X, self.y), second.objective(self.X, self.y))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            RFFPegasosSVM().decision_function(self.X)
        with self.assertRaises(ValueError):
            RFFPegasosSVM().objective(self.X, self.y)
        with self.assertRaises(ValueError):
            RFFPegasosSVM(gamma=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RFFPegasosSVM(n_components=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            RFFPegasosSVM().fit(np.empty((0, 2)), np.array([], dtype=int))
        with self.assertRaises(ValueError):
            RFFPegasosSVM().fit(self.X, np.array([0, 1]))


if __name__ == "__main__":
    unittest.main()
