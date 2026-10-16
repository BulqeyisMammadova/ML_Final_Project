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
        self.n_samples = 0        # int: number of samples at this node
        self.gain = 0.0          # float: impurity decrease from the split

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
        max_features: int | None = None,
        random_state: int = 42,
    ):
        self.task = task
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
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
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_gain = 0.0

        for j in range(n_features):                     
            values = np.unique(X[:, j])                 
            if len(values) < 2:                         
                continue
            thresholds = (values[:-1] + values[1:]) / 2
            thresholds = np.where(thresholds >= values[1:], values[:-1], thresholds) 

            for t in thresholds:                        
                left_mask = X[:, j] <= t                
                n_left = left_mask.sum()
                n_right = n_samples - n_left

                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gain = self._impurity_decrease(y, y[left_mask], y[~left_mask])

                if gain > best_gain:                    
                    best_gain = gain
                    best_feature = j
                    best_threshold = t

        if best_feature is None:
            return None
        return best_feature, best_threshold, best_gain
    #Fast split
    def _impurity_from_counts(self, counts, n):
        p = counts / n[:, None]                       
        if self.criterion == "gini":
            return 1.0 - np.sum(p ** 2, axis=1)       
        logp = np.log2(np.where(p > 0, p, 1.0))       
        return -np.sum(p * logp, axis=1)

    def _best_split_fast(self, X, y):
        n_samples, n_features = X.shape
        parent = self._impurity(y)
        best_gain, best_feature, best_threshold = 0.0, None, None

        if self.task == "classification":
            y_idx = np.searchsorted(self.classes_, y)
            Y = np.eye(len(self.classes_))[y_idx]
            total = Y.sum(axis=0)
        else:                                                        
            y_c = y.astype(float) - y.mean()                         

        n_left = np.arange(1, n_samples)
        n_right = n_samples - n_left
        if self.max_features is None:
            features = range(n_features)
        else:
            features = self._rng.permutation(n_features)
        checked = 0

        for j in features:
            if self.max_features is not None and checked >= self.max_features:
                break
            order = np.argsort(X[:, j], kind="mergesort")
            x_sorted = X[order, j]
            if x_sorted[0] == x_sorted[-1]:
                continue
            checked += 1

            
            valid = (x_sorted[:-1] != x_sorted[1:]) \
                    & (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf)
            if not valid.any():
                continue

            
            if self.task == "classification":
                left_counts = np.cumsum(Y[order], axis=0)[:-1]
                right_counts = total - left_counts
                imp_left = self._impurity_from_counts(left_counts, n_left)
                imp_right = self._impurity_from_counts(right_counts, n_right)
            else:                                                    
                ys = y_c[order]                                      
                s1 = np.cumsum(ys)[:-1]                              
                s2 = np.cumsum(ys ** 2)[:-1]                         
                t1, t2 = ys.sum(), (ys ** 2).sum()                   
                imp_left = s2 / n_left - (s1 / n_left) ** 2          
                imp_right = (t2 - s2) / n_right - ((t1 - s1) / n_right) ** 2   

            gains = parent - (n_left / n_samples) * imp_left - (n_right / n_samples) * imp_right
            gains[~valid] = -np.inf

           
            i = np.argmax(gains)
            if gains[i] > best_gain:
                best_gain = gains[i]
                best_feature = j
                best_threshold = (x_sorted[i] + x_sorted[i + 1]) / 2
                if best_threshold >= x_sorted[i + 1]:      
                    best_threshold = x_sorted[i]

        if best_feature is None:
            return None
        return best_feature, best_threshold, best_gain
    

    def _leaf_value(self, y):
        if self.task == "regression":
            return float(np.mean(y))
        counts = np.array([np.sum(y == c) for c in self.classes_])
        return counts / len(y)

    # --- grow / predict -----------------------------------------------------
    def _grow(self, X, y, depth: int) -> Node:
        
        node = Node()
        node.n_samples = len(y)
        node.value = self._leaf_value(y)     

        if self.max_depth is not None and depth >= self.max_depth:
            return node                      
        if len(y) < self.min_samples_split:
            return node                      
        if self._impurity(y) == 0:
            return node                      

        split = self._best_split_fast(X, y)
        if split is None:
            return node                      
        feature, threshold, gain = split

        
        if (len(y) / self._n_total) * gain < self.min_impurity_decrease:
            return node

       
        left_mask = X[:, feature] <= threshold
        if left_mask.all() or not left_mask.any():
            return node
        node.feature = feature
        node.threshold = threshold
        node.gain = gain
        node.left = self._grow(X[left_mask], y[left_mask], depth + 1)
        node.right = self._grow(X[~left_mask], y[~left_mask], depth + 1)
        return node
    
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if self.task == "classification":
            self.classes_ = np.unique(y)
        self._n_total = len(y)
        self.n_features_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)
        self.root = self._grow(X, y, depth=0)
        self.feature_importances_ = self._compute_importances()
        return self

    def print_tree(self, node=None, indent=""):
        if node is None:
            node = self.root
        if node.is_leaf():
            print(f"{indent}leaf: {node.value}  (n={node.n_samples})")
            return
        print(f"{indent}[column {node.feature} <= {node.threshold}]  (n={node.n_samples})")
        self.print_tree(node.left, indent + "    ")
        self.print_tree(node.right, indent + "    ")

    def get_depth(self, node=None):
        if node is None:
            node = self.root
        if node.is_leaf():
            return 0
        return 1 + max(self.get_depth(node.left), self.get_depth(node.right))

    def get_n_leaves(self, node=None):
        if node is None:
            node = self.root
        if node.is_leaf():
            return 1
        return self.get_n_leaves(node.left) + self.get_n_leaves(node.right)

    def _compute_importances(self):
        importances = np.zeros(self.n_features_)   

        def visit(node):
            if node.is_leaf():
                return                             
            importances[node.feature] += node.n_samples * node.gain
            visit(node.left)
            visit(node.right)

        visit(self.root)

        total = importances.sum()
        if total > 0:
            importances = importances / total      
        return importances

    

    

    def _predict_one(self, x, node: Node):
        while not node.is_leaf():
            if x[node.feature] <= node.threshold:
                node = node.left             
            else:
                node = node.right            
        return node.value


    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.task == "regression":
            return np.array([self._predict_one(x, self.root) for x in X])
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        if self.task != "classification":
            raise ValueError("predict_probaa only for classification")
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.root) for x in X])
