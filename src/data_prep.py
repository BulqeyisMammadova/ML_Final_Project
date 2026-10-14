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
    return df 

def make_features(df):
    """
    Turn the cleaned frame into a numeric design matrix.
      - encode categoricals (one-hot / ordinal — your choice, justify it),
      - assemble X (features) and y (target `price`).
    Return (X, y, feature_names).
    """
    raise NotImplementedError("TODO: build X, y")


def make_tier_label(y_price, threshold=None):
    """
    Derived classification target: price TIER (premium vs standard).
    Use the MEDIAN of the TRAINING price as the split threshold.
      - premium = 1 (price >  threshold), standard = 0.
    Return (y_tier, threshold).  The threshold must come from TRAIN only
    (no peeking at val/test) to avoid leakage.
    """
    raise NotImplementedError("TODO: derive the price-tier label")


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED):
    """
    Deterministic split into train / validation / test.
    Return (X_tr, y_tr, X_val, y_val, X_te, y_te).
    """
    raise NotImplementedError("TODO: split the data")


def standardize(X_tr, *others):
    """
    Standardize features using TRAIN statistics only (mean/std from X_tr),
    then apply the same transform to the other splits.
    Return the scaled arrays in the same order as given.
    (Needed for the SVM; trees don't require it.)
    """
    raise NotImplementedError("TODO: standardize using train stats")
