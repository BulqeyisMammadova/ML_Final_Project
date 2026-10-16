import os
import numpy as np
import time
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from src.random_forest import RandomForest
from sklearn.ensemble import RandomForestRegressor

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


def choose_best_params(task, criterion, X_tr, y_tr, X_val, y_val, score_name):
    best_score = -np.inf
    best_params = None

    for d in [4, 6, 8, 10, 12, None]:
        for leaf in [1, 10, 20, 50]:
            tree = DecisionTree(task=task, criterion=criterion,
                                max_depth=d, min_samples_leaf=leaf)
            tree.fit(X_tr, y_tr)
            pred = tree.predict(X_val)

            if score_name == "R2":
                score = r2(y_val, pred)
            else:
                score = precision_recall_f1(y_val, pred)[2]

            print(f"  max_depth={str(d):4s}  min_samples_leaf={leaf:3d}   val {score_name}={score:.3f}")

            if score > best_score:
                best_score = score
                best_params = {"max_depth": d, "min_samples_leaf": leaf}

    print(f"  -> Ən yaxşı: {best_params}, val {score_name}={best_score:.3f}")
    return best_params


def compare_with_sklearn(task, params, X_tr, y_tr, X_val, y_val):
    if task == "regression":
        ours = DecisionTree(task="regression", criterion="mse", **params)
    else:
        ours = DecisionTree(task="classification", criterion="gini", **params)

    start = time.perf_counter()
    ours.fit(X_tr, y_tr)
    our_fit_time = time.perf_counter() - start

    start = time.perf_counter()
    our_pred = ours.predict(X_val)
    our_pred_time = time.perf_counter() - start


    if task == "regression":
        theirs = DecisionTreeRegressor(criterion="squared_error", random_state=0, **params)
    else:
        theirs = DecisionTreeClassifier(criterion="gini", random_state=0, **params)

    start = time.perf_counter()
    theirs.fit(X_tr, y_tr)
    sk_fit_time = time.perf_counter() - start

    start = time.perf_counter()
    sk_pred = theirs.predict(X_val)
    sk_pred_time = time.perf_counter() - start

    if task == "regression":
        name = "R2"
        our_score = r2(y_val, our_pred)
        sk_score = r2(y_val, sk_pred)
    else:
        name = "F1"
        our_score = precision_recall_f1(y_val, our_pred)[2]
        sk_score = precision_recall_f1(y_val, sk_pred)[2]

    same = np.mean(np.isclose(our_pred, sk_pred))

    print(f"  bizim:   val {name}={our_score:.3f}  fit={our_fit_time:.2f}s  "
          f"predict={our_pred_time:.3f}s  leaves={ours.get_n_leaves()}")
    print(f"  sklearn: val {name}={sk_score:.3f}  fit={sk_fit_time:.2f}s  "
          f"predict={sk_pred_time:.3f}s  leaves={theirs.get_n_leaves()}")
    print(f"  Proqnozların {same * 100:.1f}%-i eynidir. "
          f"sklearn {our_fit_time / sk_fit_time:.0f} dəfə sürətlidir.")
    return ours



def plot_feature_importance(tree, names, title, filename):
    importances = tree.feature_importances_
    order = np.argsort(importances)            

    print("  Feature importance:")
    for i in order[::-1]:                      
        print(f"    {names[i]:15s} {importances[i]:.3f}")

    plt.figure(figsize=(7, 4))
    plt.barh([names[i] for i in order], importances[order], color="#2a78d6")
    plt.xlabel("Feature importance")
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)

    os.makedirs("report/figures", exist_ok=True)
    plt.savefig("report/figures/" + filename, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Qrafik saxlanıldı: report/figures/" + filename)





def error_analysis(tree, X_val, y_val, names):
    pred = tree.predict(X_val)
    true_azn = np.exp(y_val)                      
    pred_azn = np.exp(pred)
    pct_error = np.abs(pred_azn - true_azn) / true_azn * 100

    print("  Orta faiz səhvi (median):", round(np.median(pct_error), 1), "%")

    print("  Qiymət qrupuna görə səhv:")
    edges = np.percentile(true_azn, [0, 20, 40, 60, 80, 100])
    for k in range(5):
        low, high = edges[k], edges[k + 1]
        in_group = (true_azn >= low) & (true_azn <= high)
        print(f"    {low:>10,.0f} - {high:>10,.0f} AZN:  median səhv = "
              f"{np.median(pct_error[in_group]):5.1f}%")

    print("  Ən böyük 5 səhv:")
    area = X_val[:, names.index("area_m2")]
    worst = np.argsort(pct_error)[::-1][:5]
    for i in worst:
        print(f"    sahə={area[i]:7.0f} m²   əsl={true_azn[i]:>12,.0f}   "
              f"proqnoz={pred_azn[i]:>10,.0f}   səhv={pct_error[i]:6.0f}%")


    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_val, pred, s=3, alpha=0.25, color="#2a78d6")
    lo, hi = y_val.min(), y_val.max()
    plt.plot([lo, hi], [lo, hi], color="#eb6834", linewidth=2, label="mükəmməl proqnoz")
    plt.xlabel("Əsl qiymət (log)")
    plt.ylabel("Proqnoz (log)")
    plt.title("Task A: əsl və proqnoz (validation)")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25)
    os.makedirs("report/figures", exist_ok=True)
    plt.savefig("report/figures/tree_errors_regression.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Qrafik saxlanıldı: report/figures/tree_errors_regression.png")


def error_near_threshold(tree, X_val, y_val):
    _, price_tr, _, price_val, _, _ = get_splits()   
    threshold = np.median(price_tr)                  
    price_val = price_val.to_numpy()

    wrong = tree.predict(X_val) != y_val             
    distance = np.abs(price_val - threshold) / threshold * 100
    near = distance < 10                             

    print(f"  Premium həddi: {threshold:,.0f} AZN")
    print(f"  Həddə yaxın evlər (±10%):  {near.sum():5d} ev, səhv = {wrong[near].mean() * 100:.1f}%")
    print(f"  Həddən uzaq evlər:         {(~near).sum():5d} ev, səhv = {wrong[~near].mean() * 100:.1f}%")




def forest_curve(X_tr, y_tr, X_val, y_val, single_tree_score):
    start = time.perf_counter()
    forest = RandomForest(task="regression", n_trees=50, max_features=2, min_samples_leaf=5)
    forest.fit(X_tr, y_tr)
    our_time = time.perf_counter() - start

    
    all_preds = np.array([tree.predict(X_val) for tree in forest.trees])
    n_trees = list(range(1, 51))
    scores = []
    for k in n_trees:
        pred_k = all_preds[:k].mean(axis=0)
        scores.append(r2(y_val, pred_k))

    
    start = time.perf_counter()
    sk = RandomForestRegressor(n_estimators=50, max_features=2, min_samples_leaf=5, random_state=0)
    sk.fit(X_tr, y_tr)
    sk_time = time.perf_counter() - start
    sk_score = r2(y_val, sk.predict(X_val))

    print(f"  tək ağac:       val R2={single_tree_score:.3f}")
    print(f"  bizim meşə:     val R2={scores[0]:.3f} (1 ağac) -> {scores[-1]:.3f} (50 ağac), fit={our_time:.0f}s")
    print(f"  sklearn meşəsi: val R2={sk_score:.3f}, fit={sk_time:.1f}s")

    
    plt.figure(figsize=(7, 4.2))
    plt.plot(n_trees, scores, linewidth=2, color="#2a78d6", label="bizim random forest")
    plt.axhline(single_tree_score, color="gray", linestyle="--", label="tək ağac (9.6)")
    plt.axhline(sk_score, color="#eb6834", linestyle=":", linewidth=2, label="sklearn random forest")
    plt.xlabel("Ağac sayı")
    plt.ylabel("R² (validation)")
    plt.title("Random forest: ağac sayı artdıqca validation R²")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    os.makedirs("report/figures", exist_ok=True)
    plt.savefig("report/figures/forest_n_trees.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Qrafik saxlanıldı: report/figures/forest_n_trees.png")
    return forest






def task_final(run_test=False):
    # Task A
    X_tr, y_tr, X_val, y_val, X_te, y_te, names = load_regression_data()
    print("Task A (regression):")
    params_a = choose_best_params("regression", "mse", X_tr, y_tr, X_val, y_val, "R2")
    tree_a = compare_with_sklearn("regression", params_a, X_tr, y_tr, X_val, y_val)
    plot_feature_importance(tree_a, names, "Task A: qiymət", "tree_importance_regression.png")
    error_analysis(tree_a, X_val, y_val, names)
    print("Bonus: random forest (Task A):")
    forest = forest_curve(X_tr, y_tr, X_val, y_val, r2(y_val, tree_a.predict(X_val)))

    # Task B
    Xc_tr, yc_tr, Xc_val, yc_val, Xc_te, yc_te, names = load_classification_data()
    print("Task B (classification):")
    params_b = choose_best_params("classification", "gini", Xc_tr, yc_tr, Xc_val, yc_val, "F1")
    tree_b = compare_with_sklearn("classification", params_b, Xc_tr, yc_tr, Xc_val, yc_val)
    plot_feature_importance(tree_b, names, "Task B: klassifikasiya", "tree_importance_classification.png")
    error_near_threshold(tree_b, Xc_val, yc_val)


    if not run_test:
        print("Test hələ işə salınmadı (run_test=False).")
        return

   
    tree_a = DecisionTree(task="regression", criterion="mse", **params_a).fit(X_tr, y_tr)
    p = tree_a.predict(X_te)
    print(f"TEST Task A:  MAE={mae(y_te, p):.3f}  RMSE={rmse(y_te, p):.3f}  R2={r2(y_te, p):.3f}")
    p = forest.predict(X_te)
    print(f"TEST Forest:  MAE={mae(y_te, p):.3f}  RMSE={rmse(y_te, p):.3f}  R2={r2(y_te, p):.3f}")

    tree_b = DecisionTree(task="classification", criterion="gini", **params_b).fit(Xc_tr, yc_tr)
    p = tree_b.predict(Xc_te)
    prec, rec, f1 = precision_recall_f1(yc_te, p)
    auc = roc_auc(yc_te, tree_b.predict_proba(Xc_te)[:, 1])
    print(f"TEST Task B:  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")

if __name__ == "__main__":
    # task_a_first_try()
    # task_a_max_depth() 
    #task_a_min_samples_leaf()
    #task_b_gini_vs_entropy()
    task_final()