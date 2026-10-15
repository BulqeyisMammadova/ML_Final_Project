import os
import numpy as np

import matplotlib.pyplot as plt


from src.data_prep import get_splits, get_classification_splits
from src.decision_tree import DecisionTree


# --- Metrikalar -------------------------------------------------------------

def mae(y, p):
    return np.mean(np.abs(y - p))

def rmse(y, p):
    return np.sqrt(np.mean((y - p) ** 2))

def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

def precision_recall_f1(y, p):
    tp = np.sum((p == 1) & (y == 1))     
    fp = np.sum((p == 1) & (y == 0))     
    fn = np.sum((p == 0) & (y == 1))     
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def roc_auc(y, scores):
    pos = scores[y == 1]          
    neg = scores[y == 0]          

    points = 0.0
    for p in pos:                                  
        points += np.sum(neg < p)                  
        points += 0.5 * np.sum(neg == p)           

    return points / (len(pos) * len(neg))


# --- Data -------------------------------------------------------------------

def load_regression_data():
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_splits()

    names = list(X_tr.columns)

    X_tr = X_tr.to_numpy(dtype=float)
    X_val = X_val.to_numpy(dtype=float)
    X_te = X_te.to_numpy(dtype=float)

    y_tr = np.log(y_tr.to_numpy())
    y_val = np.log(y_val.to_numpy())
    y_te = np.log(y_te.to_numpy())

    return X_tr, y_tr, X_val, y_val, X_te, y_te, names

def load_classification_data():
    X_tr, y_tr, X_val, y_val, X_te, y_te = get_classification_splits()
    names = list(X_tr.columns)
    X_tr = X_tr.to_numpy(dtype=float)
    X_val = X_val.to_numpy(dtype=float)
    X_te = X_te.to_numpy(dtype=float)
    y_tr = y_tr.to_numpy()
    y_val = y_val.to_numpy()
    y_te = y_te.to_numpy()
    return X_tr, y_tr, X_val, y_val, X_te, y_te, names


# --- Prob -------------------------------------------------------------

def task_a_first_try():
    X_tr, y_tr, X_val, y_val, X_te, y_te, names = load_regression_data()

    tree = DecisionTree(task="regression", criterion="mse", max_depth=8)
    tree.fit(X_tr, y_tr)

    pred = tree.predict(X_val)

    print("Task A, max_depth=8, validation:")
    print(f"  MAE  = {mae(y_val, pred):.3f}")
    print(f"  RMSE = {rmse(y_val, pred):.3f}")
    print(f"  R²   = {r2(y_val, pred):.3f}")



def task_a_max_depth():                                         
    X_tr, y_tr, X_val, y_val, X_te, y_te, names = load_regression_data()

    depths = list(range(1, 17))          
    train_scores = []
    val_scores = []

    for d in depths:
        tree = DecisionTree(task="regression", criterion="mse", max_depth=d)
        tree.fit(X_tr, y_tr)
        train_scores.append(r2(y_tr, tree.predict(X_tr)))
        val_scores.append(r2(y_val, tree.predict(X_val)))
        print(f"max_depth={d:2d}   train R²={train_scores[-1]:.3f}   val R²={val_scores[-1]:.3f}")

    best_depth = depths[int(np.argmax(val_scores))]
    print("Ən yaxşı max_depth (validation-a görə):", best_depth)

    plt.figure(figsize=(7, 4.2))
    plt.plot(depths, train_scores, marker="o", linewidth=2, color="#2a78d6", label="train")
    plt.plot(depths, val_scores, marker="o", linewidth=2, color="#eb6834", label="validation")
    plt.axvline(best_depth, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("max_depth")
    plt.ylabel("R² (log qiymət)")
    plt.title("Task A: max_depth artdıqca train və validation R²")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)

    os.makedirs("report/figures", exist_ok=True)
    plt.savefig("report/figures/tree_max_depth_regression.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Qrafik saxlanıldı: report/figures/tree_max_depth_regression.png")


def task_a_min_samples_leaf():
    X_tr, y_tr, X_val, y_val, X_te, y_te, names = load_regression_data()

    leaf_sizes = [1, 5, 10, 20, 50, 100, 200, 500]
    train_scores = []
    val_scores = []

    for m in leaf_sizes:
        tree = DecisionTree(task="regression", criterion="mse", min_samples_leaf=m)
        tree.fit(X_tr, y_tr)
        train_scores.append(r2(y_tr, tree.predict(X_tr)))
        val_scores.append(r2(y_val, tree.predict(X_val)))
        print(f"min_samples_leaf={m:3d}   train R²={train_scores[-1]:.3f}   "
              f"val R²={val_scores[-1]:.3f}   leaf sayı={tree.get_n_leaves()}")

    best_leaf = leaf_sizes[int(np.argmax(val_scores))]
    print("Ən yaxşı min_samples_leaf (validation-a görə):", best_leaf)

    plt.figure(figsize=(7, 4.2))
    plt.plot(leaf_sizes, train_scores, marker="o", linewidth=2, color="#2a78d6", label="train")
    plt.plot(leaf_sizes, val_scores, marker="o", linewidth=2, color="#eb6834", label="validation")
    plt.axvline(best_leaf, color="gray", linestyle="--", linewidth=1)
    plt.xscale("log")
    plt.xlabel("min_samples_leaf (log şkala)")
    plt.ylabel("R² (log qiymət)")
    plt.title("Task A: min_samples_leaf artdıqca train və validation R²")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)

    os.makedirs("report/figures", exist_ok=True)
    plt.savefig("report/figures/tree_min_samples_leaf_regression.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Qrafik saxlanıldı: report/figures/tree_min_samples_leaf_regression.png")



def task_b_gini_vs_entropy():
    X_tr, y_tr, X_val, y_val, X_te, y_te, names = load_classification_data()
    print("premium payı (train):", round(y_tr.mean(), 3))

    depths = list(range(1, 17))
    f1_scores = {"gini": [], "entropy": []}

    for criterion in ["gini", "entropy"]:
        for d in depths:
            tree = DecisionTree(task="classification", criterion=criterion, max_depth=d)
            tree.fit(X_tr, y_tr)
            pred = tree.predict(X_val)
            p, r, f1 = precision_recall_f1(y_val, pred)
            auc = roc_auc(y_val, tree.predict_proba(X_val)[:, 1])
            f1_scores[criterion].append(f1)
            print(f"{criterion:7s} max_depth={d:2d}   precision={p:.3f}  recall={r:.3f}  "
                  f"F1={f1:.3f}  ROC-AUC={auc:.3f}")

    for criterion in ["gini", "entropy"]:
        best = int(np.argmax(f1_scores[criterion]))
        print(f"Ən yaxşı {criterion}: max_depth={depths[best]}, val F1={f1_scores[criterion][best]:.3f}")

    plt.figure(figsize=(7, 4.2))
    plt.plot(depths, f1_scores["gini"], marker="o", linewidth=2, color="#2a78d6", label="gini")
    plt.plot(depths, f1_scores["entropy"], marker="o", linewidth=2, color="#eb6834", label="entropy")
    plt.xlabel("max_depth")
    plt.ylabel("F1 (validation)")
    plt.title("Task B: gini və entropy, validation F1")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)

    os.makedirs("report/figures", exist_ok=True)
    plt.savefig("report/figures/tree_gini_vs_entropy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Qrafik saxlanıldı: report/figures/tree_gini_vs_entropy.png")

if __name__ == "__main__":
   # task_a_first_try()
   # task_a_max_depth() 
    #task_a_min_samples_leaf()
    task_b_gini_vs_entropy()