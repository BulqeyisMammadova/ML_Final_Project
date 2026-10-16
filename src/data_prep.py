"""
data_prep.py — loading, cleaning, splitting, and the price-tier label.

Fill in the TODOs. Keep ALL data wrangling here so the rest of the code just
calls these functions. Fix your random seed everywhere (see SEED).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Config -----------------------------------------------------------------
SEED = 42
DATA_PATH = "data/house_sale.csv"   # change if your filename differs

# Columns that LEAK the target (derived from price). Drop these before training.
# TODO: list the actual leakage columns you find (e.g. a unit/per-m2 price).
LEAKAGE_COLUMNS: list[str] = [
    "unit_price",
    "total_price",
]


def load_raw(path: str = DATA_PATH):
    """Read the raw CSV into a DataFrame (keep UTF-8). Return it unmodified."""
    df=pd.read_csv(path,encoding="utf-8")
    return df


def clean(df):
    """
    Clean the raw frame:
      - fix dtypes (prices/areas to numeric; strip currency/text),
      - handle missing values,
      - drop duplicates,
      - handle outliers (document your rule),
      - drop LEAKAGE_COLUMNS.
    Return the cleaned DataFrame.
    """
    df=df.copy()
    df["datetime_scrape_x"]=pd.to_datetime(
        df["datetime_scrape_x"],
        format="mixed",
        utc=True,
        errors="coerce"
    )
    

    df=df.sort_values(
        by="datetime_scrape_x",
        ascending=True,
        kind="mergesort",
    ).drop_duplicates(
        keep="last",
        subset="estate_rel_url",
        ignore_index=True
    )
    df["area_unit"]=df["Sahə"].str.extract(
        pat=r"(m²|sot)",
        expand=False)
    df["area_value"]=df["Sahə"].str.extract(
        pat=r"(\d+(?:\.\d+)?)",
        expand=False
    )
    df["area_value"]=pd.to_numeric(
        df["area_value"],
        errors="coerce"
    )
    df["area_m2"] = df["area_value"].copy()
    df.loc[df["area_unit"] == "sot", "area_m2"] = (
    df.loc[df["area_unit"] == "sot", "area_m2"] * 100
    )
    floors = df["Mərtəbə"].str.split("/", expand=True)
    df["floor_number"]=pd.to_numeric(
        floors[0].str.strip(),
        errors="coerce"
    )
    df["total_floors"]=pd.to_numeric(
        floors[1].str.strip(),
        errors="coerce"
    )
    df["land_area_value"]=df["Torpaq sahəsi"].str.extract(
        pat=r"(\d+(?:\.\d+)?)",
        expand=False
    )
    df["land_area_value"]=pd.to_numeric(
        df["land_area_value"],
        errors="coerce"
    )
    df["land_area_m2"]=df["land_area_value"]*100
    df=df.drop(columns=LEAKAGE_COLUMNS)
    df=df.drop(columns=["mortgage"])
    df=df.drop(columns=["repair"])
    df=df.drop(columns=["bill_of_sale"])

    df["room_count"]=df["Otaq sayı"].copy()
    df["property_type"]=df["Kateqoriya"].copy()

    df["land_area_known"]=df["land_area_m2"].notna().astype(int)

    return df 

def make_features(df):
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot / ordinal — your choice, justify it),
      - assemble X (features) and y (target `price`).
    Return (X, y, feature_names).
    """
    y=df["price"].copy()
    numeric_features=["area_m2","floor_number","total_floors","lat","lng","room_count","land_area_m2"]
    categorical_features=["property_type"]
    X=df[numeric_features+categorical_features].copy()
    feature_names=X.columns.tolist()
    return X, y, feature_names


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >  threshold), standard = 0.
    Return (y_tier, threshold).  The threshold must come from TRAIN only
    (no peeking at val/test) to avoid leakage.
    """
    if threshold is None:
        threshold=y_price.median()
    y_tier=(y_price>threshold).astype(int)
    return y_tier, threshold


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    n=len(y)
    rng=np.random.default_rng(seed)
    indexs=rng.permutation(n)
    n_test=int(n*test_size)
    n_validation=int(n*val_size)
    n_train=n-(n_test+n_validation)
    train_index=indexs[:n_train]
    validation_index=indexs[n_train:n_train+n_validation]
    test_index=indexs[n_train+n_validation:]
    X_train=X.iloc[train_index]
    y_train=y.iloc[train_index]
    X_validation=X.iloc[validation_index]
    y_validation=y.iloc[validation_index]
    X_test=X.iloc[test_index]
    y_test=y.iloc[test_index]

    return X_train, y_train, X_validation, y_validation, X_test, y_test



def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    train = np.asarray(X_tr, dtype=float)
    if train.ndim != 2 or train.shape[0] == 0 or not np.isfinite(train).all():
        raise ValueError("Training features must be a nonempty finite 2D array")
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1.0
    result = []
    for features in (X_tr, *others):
        values = np.asarray(features, dtype=float)
        if values.ndim != 2 or values.shape[1] != train.shape[1] or not np.isfinite(values).all():
            raise ValueError("Every feature set must be finite with matching columns")
        result.append((values - mean) / std)
    return tuple(result)


def get_splits():
    df=load_raw()
    df=clean(df)
    X, y, feature_names=make_features(df)
    X_train,y_train,X_validation,y_validation,X_test,y_test=train_val_test_split(X,y)

    X_train=pd.get_dummies(X_train,columns=["property_type"],dtype=int)
    X_validation=pd.get_dummies(X_validation,columns=["property_type"],dtype=int)
    X_validation=X_validation.reindex(columns=X_train.columns,fill_value=0)
    X_test=pd.get_dummies(X_test,columns=["property_type"],dtype=int)
    X_test=X_test.reindex(columns=X_train.columns,fill_value=0)
    # y_tier_train,threshold=make_tier_label(y_train)
    # y_tier_validation,validation_threshold=make_tier_label(y_validation,threshold)
    # y_tier_test,test_threshold=make_tier_label(y_test,threshold)

    X_train["land_area_m2"]=X_train["land_area_m2"].fillna(0)
    X_validation["land_area_m2"]=X_validation["land_area_m2"].fillna(0)
    X_test["land_area_m2"]=X_test["land_area_m2"].fillna(0)

    train_medians=X_train.median()
    X_train=X_train.fillna(train_medians)
    X_validation=X_validation.fillna(train_medians)
    X_test=X_test.fillna(train_medians)
    return X_train,y_train,X_validation,y_validation,X_test,y_test

    
    
def get_classification_splits():
    X_train,y_train,X_validation,y_validation,X_test,y_test=get_splits()
    y_tier_train,threshold=make_tier_label(y_train)
    y_tier_validation,validation_threshold=make_tier_label(y_validation,threshold)
    y_tier_test,test_threshold=make_tier_label(y_test,threshold)
    return X_train,y_tier_train,X_validation,y_tier_validation,X_test,y_tier_test
