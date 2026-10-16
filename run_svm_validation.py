"""Reproduce Linear Pegasos and RFF validation metrics with current preprocessing."""

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src import data_prep
from src.svm import PegasosSVM, RFFPegasosSVM


def report_validation_metrics(name, model, X_val, y_val):
    predictions = model.predict(X_val)
    matrix = confusion_matrix(y_val, predictions, labels=[0, 1])

    print("\n{}".format(name))
    print("Accuracy:  {:.6f}".format(accuracy_score(y_val, predictions)))
    print("Precision: {:.6f}".format(precision_score(y_val, predictions, zero_division=0)))
    print("Recall:    {:.6f}".format(recall_score(y_val, predictions, zero_division=0)))
    print("F1-score:  {:.6f}".format(f1_score(y_val, predictions, zero_division=0)))
    print("Confusion matrix (rows=true [0, 1], columns=predicted [0, 1]):")
    print(matrix)


def main():
    cleaned = data_prep.clean(data_prep.load_raw())
    X, y_price, _ = data_prep.make_features(cleaned)
    (
        X_train,
        y_train_price,
        X_val,
        y_val_price,
        _,
        _,
    ) = data_prep.train_val_test_split(X, y_price)

    X_train = pd.get_dummies(X_train, columns=["property_type"], dtype=int)
    X_val = pd.get_dummies(
        X_val, columns=["property_type"], dtype=int
    ).reindex(columns=X_train.columns, fill_value=0)

    print("Preprocessing statistics:")
    print("- land_area_m2 missing values are replaced with fixed value 0.")
    X_train["land_area_m2"] = X_train["land_area_m2"].fillna(0)
    X_val["land_area_m2"] = X_val["land_area_m2"].fillna(0)

    train_medians = X_train.median()
    print("- Remaining imputation medians are computed from X_train only.")
    X_train = X_train.fillna(train_medians)
    X_val = X_val.fillna(train_medians)

    y_train, threshold = data_prep.make_tier_label(y_train_price)
    y_val, _ = data_prep.make_tier_label(y_val_price, threshold)
    print("- Premium threshold is the training-price median: {:.6f}.".format(threshold))

    X_train_scaled, X_val_scaled = data_prep.standardize(X_train, X_val)
    print("- Scaling means/stds are computed from X_train only; the same transform is applied to X_val.")

    feature_names = X_train.columns.tolist()
    print("\nTrain samples: {}".format(len(X_train_scaled)))
    print("Validation samples: {}".format(len(X_val_scaled)))
    print("Feature count: {}".format(len(feature_names)))
    print("Feature names (training column order): {}".format(feature_names))
    print("Test partition: not evaluated, inspected, or used for model selection.")

    linear_model = PegasosSVM(
        lambda_=0.01,
        n_iters=3 * len(X_train_scaled),
        random_state=42,
    ).fit(X_train_scaled, y_train)
    report_validation_metrics(
        "Linear Pegasos SVM", linear_model, X_val_scaled, y_val
    )

    rff_model = RFFPegasosSVM(
        gamma=1,
        lambda_=0.0001,
        n_components=500,
        n_iters=20000,
        random_state=42,
    ).fit(X_train_scaled, y_train)
    report_validation_metrics("RFF Pegasos SVM", rff_model, X_val_scaled, y_val)


if __name__ == "__main__":
    main()
