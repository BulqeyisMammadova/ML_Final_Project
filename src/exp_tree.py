import os
import numpy as np

import matplotlib.pyplot as plt

from src.data_prep import get_splits
from src.decision_tree import DecisionTree


# --- Metrikalar -------------------------------------------------------------

def mae(y, p):
    return np.mean(np.abs(y - p))

def rmse(y, p):
    return np.sqrt(np.mean((y - p) ** 2))

def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


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



if __name__ == "__main__":
    task_a_first_try()
    task_a_max_depth() 