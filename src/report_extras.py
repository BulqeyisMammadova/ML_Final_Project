"""Extra report figures and tables on the run_all split (training/validation only).

Run from the repository root:  python -m src.report_extras

Produces
  report/figures/eda_price.png          price and log-price distribution (train rows)
  report/figures/tree_overfit_depth.png train vs validation log-RMSE of our tree
  results/report_extras.json            numbers quoted in the report
The test split is never used here.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import data_prep, evaluate
from . import run_all as R
from .decision_tree import DecisionTree
from .svm import PegasosSVM


def main():
    raw = data_prep.load_raw(str(R.ROOT / data_prep.DATA_PATH))
    features, price, _ = data_prep.make_features(data_prep.clean(raw))
    ok = np.isfinite(price.to_numpy(float)) & (price.to_numpy(float) > 0)
    features, price = features.loc[ok], price.loc[ok]
    tx_raw, ty, vx_raw, vy, _, _, threshold = R._split(features, price)
    tx, vx = R._impute(tx_raw, vx_raw)
    train_price, val_price = ty.to_numpy(float), vy.to_numpy(float)
    out = {}
    R.FIGURES.mkdir(parents=True, exist_ok=True)
    R.RESULTS.mkdir(parents=True, exist_ok=True)

    # 1) EDA: price and log-price distribution on training rows only.
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.6))
    axes[0].hist(train_price / 1e3, bins=np.linspace(0, 1500, 61), color="#2a78d6")
    axes[0].set_xlabel("Price (thousand AZN, clipped at 1,500)")
    axes[0].set_ylabel("Listings")
    axes[1].hist(np.log1p(train_price), bins=60, color="#2a78d6")
    axes[1].axvline(np.log1p(threshold), color="#b4471a", linestyle="--",
                    label="Median 210k AZN")
    axes[1].set_xlabel("log(1 + price)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R.FIGURES / "eda_price.png", dpi=200)
    plt.close(fig)
    out["train_price_skew_mean_over_median"] = float(train_price.mean() / np.median(train_price))

    # 2) Tree overfitting: train vs validation log-RMSE as depth grows.
    log_tr, log_va = np.log1p(train_price), np.log1p(val_price)
    depths = list(range(2, 21, 2))
    curve = []
    for depth in depths:
        tree = DecisionTree(task="regression", criterion="mse", max_depth=depth,
                            min_samples_leaf=1, random_state=R.SEED).fit(tx, log_tr)
        curve.append({"max_depth": depth,
                      "train_log_RMSE": float(evaluate.rmse(log_tr, tree.predict(tx))),
                      "val_log_RMSE": float(evaluate.rmse(log_va, tree.predict(vx)))})
    out["tree_overfit_depth"] = curve
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.plot(depths, [c["train_log_RMSE"] for c in curve], "o-", label="Train")
    ax.plot(depths, [c["val_log_RMSE"] for c in curve], "s-", label="Validation")
    ax.set_xlabel("max_depth (min_samples_leaf = 1)")
    ax.set_ylabel("log-RMSE")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R.FIGURES / "tree_overfit_depth.png", dpi=200)
    plt.close(fig)

    # 3) SVM: learning-rate schedule and number of updates (lambda = 1e-3).
    st, sv = data_prep.standardize(tx, vx)
    yt = (train_price > threshold).astype(int)
    yv = (val_price > threshold).astype(int)
    rows = []
    for schedule in ("pegasos", "sqrt"):
        for n_iters in (2000, 5000, 20000):
            svm = PegasosSVM(lambda_=1e-3, n_iters=n_iters, random_state=R.SEED,
                             learning_rate=schedule).fit(st, yt)
            norm = float(np.linalg.norm(svm.w))
            rows.append({"schedule": schedule, "n_iters": n_iters,
                         "val_F1": float(evaluate.precision_recall_f1(yv, svm.predict(sv))[2]),
                         "weight_norm": norm, "margin_width": 2.0 / norm,
                         "objective": float(svm.objective(st, yt))})
    out["svm_schedule_updates"] = rows

    with open(R.RESULTS / "report_extras.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
