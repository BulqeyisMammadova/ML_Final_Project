import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from src.decision_tree import DecisionTree, gini, entropy, mse


def our_structure(tree):
    result = []

    def visit(node):
        if node.is_leaf():
            result.append("leaf")
            return
        result.append((node.feature, node.threshold))
        visit(node.left)
        visit(node.right)

    visit(tree.root)
    return result


def sklearn_structure(sk_tree):
   
    t = sk_tree.tree_
    result = []

    def visit(i):
        if t.children_left[i] == -1:          
            result.append("leaf")
            return
        result.append((int(t.feature[i]), float(t.threshold[i])))
        visit(t.children_left[i])
        visit(t.children_right[i])

    visit(0)
    return result


def assert_same_structure(ours, theirs):
    assert len(ours) == len(theirs), f"Node count differs: {len(ours)} vs {len(theirs)}"
    for a, b in zip(ours, theirs):
        if a == "leaf" or b == "leaf":
            assert a == b, f"One is leaf, other is not: {a} vs {b}"
        else:
            assert a[0] == b[0], f"Different feature: {a} vs {b}"
            assert np.isclose(a[1], b[1], atol=1e-5), f"Different threshold: {a} vs {b}"




X_TINY = np.array([[1, 50], [3, 60], [1, 90], [3, 95], [1, 110], [3, 120], [2, 130], [2, 45]], dtype=float)
Y_TINY = np.array([0, 0, 0, 1, 0, 1, 1, 0])

X_TINY_REG = np.array([[1, 40], [2, 55], [3, 70], [1, 85], [2, 100], [3, 130]], dtype=float)
Y_TINY_REG = np.array([50.0, 65.0, 90.0, 110.0, 140.0, 220.0])


def test_impurity_values():
    assert np.isclose(gini([0, 0, 1, 1]), 0.5)
    assert np.isclose(gini([0, 0, 0, 1]), 0.375)
    assert np.isclose(gini([1, 1, 1, 1]), 0.0)
    assert np.isclose(entropy([0, 0, 1, 1]), 1.0)
    assert np.isclose(entropy([1, 1, 1, 1]), 0.0)
    assert np.isclose(mse([100, 200]), 2500.0)


def test_same_splits_as_sklearn_classification():
    for criterion in ["gini", "entropy"]:
        ours = DecisionTree(criterion=criterion).fit(X_TINY, Y_TINY)
        for seed in range(10):  
            theirs = DecisionTreeClassifier(criterion=criterion, random_state=seed).fit(X_TINY, Y_TINY)
            assert_same_structure(our_structure(ours), sklearn_structure(theirs))


def test_same_splits_as_sklearn_regression():
    ours = DecisionTree(task="regression", criterion="mse", max_depth=2).fit(X_TINY_REG, Y_TINY_REG)
    for seed in range(10):
        theirs = DecisionTreeRegressor(max_depth=2, random_state=seed).fit(X_TINY_REG, Y_TINY_REG)
        assert_same_structure(our_structure(ours), sklearn_structure(theirs))
        assert np.allclose(ours.predict(X_TINY_REG), theirs.predict(X_TINY_REG))


def test_predictions_match_sklearn_on_random_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    X_new = rng.normal(size=(100, 3))
    ours = DecisionTree(max_depth=3).fit(X, y)
    theirs = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    assert np.array_equal(ours.predict(X_new), theirs.predict(X_new))
    assert np.allclose(ours.predict_proba(X_new), theirs.predict_proba(X_new))
    assert np.allclose(ours.feature_importances_, theirs.feature_importances_)


def test_each_hyperparameter_changes_the_tree():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + X[:, 1] + rng.normal(scale=0.5, size=200) > 0).astype(int)
    full = DecisionTree().fit(X, y).get_n_leaves()
    for params in [{"max_depth": 3}, {"min_samples_split": 40},
                   {"min_samples_leaf": 20}, {"min_impurity_decrease": 0.01}]:
        n = DecisionTree(**params).fit(X, y).get_n_leaves()
        assert n < full, f"{params} ağacı kiçiltmədi ({n} vs {full})"


def test_predict_proba_rows_sum_to_one():
    proba = DecisionTree(max_depth=1).fit(X_TINY, Y_TINY).predict_proba(X_TINY)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_string_labels():
    y = np.where(Y_TINY == 1, "premium", "standart")
    pred = DecisionTree().fit(X_TINY, y).predict([[3, 100], [1, 70]])
    assert list(pred) == ["premium", "standart"]

def test_fast_split_same_as_simple():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4))
    X[:, 2] = rng.integers(0, 4, 300)                  
    y_cls = (X[:, 0] + X[:, 2] + rng.normal(size=300) > 1).astype(int)
    y_reg = X[:, 0] * 3 + X[:, 2] + rng.normal(size=300)

    for task, crit, y in [("classification", "gini", y_cls),
                          ("classification", "entropy", y_cls),
                          ("regression", "mse", y_reg)]:
        t = DecisionTree(task=task, criterion=crit)
        if task == "classification":
            t.classes_ = np.unique(y)
        old = t._best_split(X, y)
        new = t._best_split_fast(X, y)
        assert old[0] == new[0]                         
        assert np.isclose(old[1], new[1])               
        assert np.isclose(old[2], new[2])   


test_fast_split_same_as_simple()
print("OK: test_fast_split_same_as_simple")


def test_close_values_do_not_loop():
    # iki çox yaxın rəqəm: ortası yuvarlaqlaşıb böyüyünə bərabər olur
    a = 40.4
    b = np.nextafter(a, 100)        # a-dan sonrakı ən kiçik mümkün rəqəm
    X = np.array([[a], [b], [b]])
    y = np.array([1.0, 2.0, 3.0])
    tree = DecisionTree(task="regression", criterion="mse").fit(X, y)
    assert np.allclose(tree.predict(X), [1.0, 2.5, 2.5])

test_close_values_do_not_loop()
print("OK: test_close_values_do_not_loop")



if __name__ == "__main__":
    tests = [f for name, f in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"OK    {test.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"Error  {test.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} test passed")