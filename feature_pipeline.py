"""
feature_pipeline.py — Encode categoricals, scale numerics, train/test split.

Takes cleaned Titanic & Lusitania CSVs and produces ready-to-model feature
matrices for three scenarios:
  1. Titanic only       (11 features: sex, age, class, fare, sibsp, parch, embarked)
  2. Lusitania only     ( 6 features: sex, age, class, adult_minor, passenger_crew)
  3. Pooled (common)    ( 4 features: sex, age, class)

Outputs are saved to `processed/` as parquet so every modelling script starts
from the same preprocessed data.

Usage:
    python feature_pipeline.py                     # run all three scenarios
    python feature_pipeline.py --scenario titanic  # run one scenario only
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DATA_DIR = HERE
PROCESSED_DIR = HERE / "processed"
TITANIC_CLEANED = DATA_DIR / "titanic_cleaned.csv"
LUSITANIA_CLEANED = DATA_DIR / "lusitania_cleaned.csv"

# ── Settings ────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET = "survived"

# ── Feature roles per dataset ──────────────────────────────────────────────
# Columns common to both after cleaning
COMMON_NUMERIC = ["age", "time_to_sink_min"]
COMMON_CATEGORICAL = ["sex", "class"]
COMMON_CATEGORICAL_ORDINAL = ["class"]  # for ordinal encoding (First > Second > Third)

# Titanic-specific features
TITANIC_NUMERIC = ["fare", "sibsp", "parch"]
TITANIC_CATEGORICAL = ["embarked"]

# Lusitania-specific features
LUSITANIA_CATEGORICAL = ["adult_minor", "passenger_crew"]


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_cleaned_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both cleaned CSVs. Returns (titanic, lusitania)."""
    titanic = pd.read_csv(TITANIC_CLEANED)
    lusitania = pd.read_csv(LUSITANIA_CLEANED)

    # Convert bool target to int (0/1) for sklearn compatibility
    for df in (titanic, lusitania):
        df[TARGET] = df[TARGET].astype(int)

    # Fare is entirely NaN in Lusitania — drop it so we don't ship a dead column
    if "fare" in lusitania.columns and lusitania["fare"].isna().all():
        lusitania = lusitania.drop(columns=["fare"])

    return titanic, lusitania


# ══════════════════════════════════════════════════════════════════════════════
# 2.  ENCODING + SCALING — BUILD PREPROCESSORS
# ══════════════════════════════════════════════════════════════════════════════

def _ordinal_class() -> OrdinalEncoder:
    """Ordinal encoder for class: First=0, Second=1, Third=2 (or unknown)."""
    return OrdinalEncoder(
        categories=[["First", "Second", "Third"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float64,
    )


def build_combined_preprocessor(
    numeric_cols: list[str],
    onehot_cols: list[str],
    ordinal_cols: list[str] | None = None,
) -> ColumnTransformer:
    """
    Build a single ColumnTransformer that applies the right transform per column.

    - Numeric cols   → StandardScaler
    - One-hot cols   → OneHotEncoder(drop='first')  (k-1 dummies → avoids multicollinearity)
    - Ordinal cols   → OrdinalEncoder with known category order
    """
    transformers = []

    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))

    if onehot_cols:
        transformers.append(
            (
                "oh",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                onehot_cols,
            )
        )

    if ordinal_cols:
        transformers.append(("ord", _ordinal_class(), ordinal_cols))

    ct = ColumnTransformer(transformers, verbose_feature_names_out=False)
    return ct


def get_feature_names(ct: ColumnTransformer, input_df: pd.DataFrame) -> list[str]:
    """
    Extract human-readable feature names from a fitted ColumnTransformer.
    """
    names: list[str] = []
    for name, transformer, columns in ct.transformers_:
        if transformer == "drop" or transformer is None:
            continue
        if name == "num":
            names.extend(columns)
        elif name == "ord":
            names.extend(columns)
        elif name == "oh":
            # OneHotEncoder: get feature names from the encoder
            cats = transformer.categories_
            for col, cats_for_col in zip(columns, cats):
                # drop='first' → skip the first category
                for cat in cats_for_col[1:]:
                    names.append(f"{col}_{cat}")
    return names


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def stratified_split(
    X: pd.DataFrame, y: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Stratified train/test split preserving survival-class proportions."""
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FULL PIPELINE PER SCENARIO
# ══════════════════════════════════════════════════════════════════════════════

def run_scenario(
    *,
    name: str,
    df: pd.DataFrame,
    numeric_cols: list[str],
    onehot_cols: list[str],
    ordinal_cols: list[str] | None = None,
) -> dict:
    """
    Run the full feature pipeline: encode → scale → split.

    Returns a dict with X_train, X_test, y_train, y_test, feature_names, preprocessor.
    """
    print(f"\n{'-' * 60}")
    print(f"  Scenario: {name}")
    print(f"  Rows: {len(df)}")
    print(f"{'-' * 60}")

    # --- Prepare feature matrix and target ---
    y = df[TARGET].values
    all_feature_cols = numeric_cols + onehot_cols + (ordinal_cols or [])
    X_raw = df[all_feature_cols].copy()

    # Quick report
    n_survived = y.sum()
    print(f"  Target:  {n_survived}/{len(y)} survived ({y.mean() * 100:.1f}%)")
    print(f"  Raw features ({len(all_feature_cols)}): {all_feature_cols}")
    for c in numeric_cols:
        print(f"    [num]  {c:20s}  mean={X_raw[c].mean():8.2f}  std={X_raw[c].std():8.2f}")
    for c in onehot_cols:
        print(f"    [oh]   {c:20s}  categories={list(X_raw[c].unique())}")
    for c in ordinal_cols or []:
        print(f"    [ord]  {c:20s}  categories={list(X_raw[c].unique())}")

    # --- Build and fit preprocessor ---
    ct = build_combined_preprocessor(numeric_cols, onehot_cols, ordinal_cols)
    X_encoded = ct.fit_transform(X_raw)

    # --- Build nice feature names ---
    feature_names = get_feature_names(ct, X_raw)
    X = pd.DataFrame(X_encoded, columns=feature_names, index=X_raw.index)
    print(f"  Encoded features ({len(feature_names)}): {feature_names}")

    # --- Train / test split ---
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    print(f"\n  Train: {len(X_train)} rows  (survival={y_train.mean() * 100:.1f}%)")
    print(f"  Test:  {len(X_test)} rows  (survival={y_test.mean() * 100:.1f}%)")

    return {
        "name": name,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "preprocessor": ct,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SAVE PROCESSED DATA
# ══════════════════════════════════════════════════════════════════════════════

def save_scenario(result: dict) -> None:
    """Write train/test splits and a metadata file to the processed directory."""
    PROCESSED_DIR.mkdir(exist_ok=True)
    name = result["name"].lower().replace(" ", "_")

    # Parquet preserves dtypes and is fast — no CSV type-munging on reload
    result["X_train"].to_parquet(PROCESSED_DIR / f"X_train_{name}.parquet")
    result["X_test"].to_parquet(PROCESSED_DIR / f"X_test_{name}.parquet")

    np.save(PROCESSED_DIR / f"y_train_{name}.npy", result["y_train"])
    np.save(PROCESSED_DIR / f"y_test_{name}.npy", result["y_test"])

    # Metadata as a tiny JSON
    meta = {
        "scenario": result["name"],
        "train_rows": len(result["X_train"]),
        "test_rows": len(result["X_test"]),
        "features": result["feature_names"],
        "n_features": len(result["feature_names"]),
        "train_survival_rate": float(result["y_train"].mean()),
        "test_survival_rate": float(result["y_test"].mean()),
    }
    pd.Series(meta).to_json(PROCESSED_DIR / f"meta_{name}.json")

    print(f"  -> Saved to {PROCESSED_DIR / f'{{X,y}}_{{train,test}}_{name}.*'}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Feature pipeline: encode → scale → split.")
    parser.add_argument(
        "--scenario",
        choices=["titanic", "lusitania", "pooled", "all"],
        default="all",
        help="Which scenario(s) to run (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MIH Feature Pipeline")
    print("  Encode categoricals - Scale numerics - Train/test split")
    print("=" * 60)

    # ── Load ────────────────────────────────────────────────────────────────
    titanic, lusitania = load_cleaned_datasets()

    # ── Pooled (common features only) ──────────────────────────────────────
    common_features = list(
        set(COMMON_NUMERIC + COMMON_CATEGORICAL)
        & set(titanic.columns)
        & set(lusitania.columns)
    )
    common_numeric = [c for c in COMMON_NUMERIC if c in common_features]
    common_onehot = [c for c in COMMON_CATEGORICAL if c not in COMMON_CATEGORICAL_ORDINAL]
    common_ordinal = [c for c in COMMON_CATEGORICAL if c in COMMON_CATEGORICAL_ORDINAL]
    # Move class from onehot to ordinal if it's there
    common_onehot = [c for c in common_onehot if c not in (common_ordinal or [])]

    # Build pooled DataFrame — only common columns
    cols_to_concat = common_features + [TARGET]
    titanic_common = titanic[cols_to_concat].copy()
    lusitania_common = lusitania[[c for c in cols_to_concat if c in lusitania.columns]].copy()

    # Filter Lusitania to passengers only (exclude crew)
    if "passenger_crew" in lusitania.columns:
        lusitania_common = lusitania_common.loc[
            lusitania.loc[lusitania_common.index, "passenger_crew"] == "Passenger"
        ].copy()

    # Add disaster context
    titanic_common["time_to_sink_min"] = 160.0   # Titanic: ~2h40m
    lusitania_common["time_to_sink_min"] = 18.0   # Lusitania: ~18 min

    pooled = pd.concat([titanic_common, lusitania_common], ignore_index=True)

    # Interaction term: sex_female × time_to_sink_min
    # sex is still raw ('male'/'female') at this point
    pooled["sex_x_time"] = (pooled["sex"] == "female").astype(float) * pooled["time_to_sink_min"]

    scenarios = []

    if args.scenario in ("titanic", "all"):
        scenarios.append(
            ("Titanic", titanic, COMMON_NUMERIC + TITANIC_NUMERIC,
             [c for c in COMMON_CATEGORICAL + TITANIC_CATEGORICAL if c not in COMMON_CATEGORICAL_ORDINAL],
             COMMON_CATEGORICAL_ORDINAL)
        )

    if args.scenario in ("lusitania", "all"):
        scenarios.append(
            ("Lusitania", lusitania, COMMON_NUMERIC,
             [c for c in COMMON_CATEGORICAL + LUSITANIA_CATEGORICAL if c not in COMMON_CATEGORICAL_ORDINAL],
             COMMON_CATEGORICAL_ORDINAL)
        )

    if args.scenario in ("pooled", "all"):
        # time_to_sink_min is added after common_features is computed,
        # so include it explicitly in the numeric columns
        pooled_numeric = ["age", "time_to_sink_min", "sex_x_time"]
        scenarios.append(
            ("Pooled", pooled, pooled_numeric, common_onehot, common_ordinal)
        )

    # ── Run ────────────────────────────────────────────────────────────────
    results = []
    for name, df, num_cols, oh_cols, ord_cols in scenarios:
        result = run_scenario(
            name=name,
            df=df,
            numeric_cols=num_cols,
            onehot_cols=oh_cols,
            ordinal_cols=ord_cols,
        )
        save_scenario(result)
        results.append(result)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for r in results:
        print(f"  {r['name']:15s}  "
              f"Train={len(r['X_train']):>4d}  "
              f"Test={len(r['X_test']):>4d}  "
              f"Features={len(r['feature_names']):>2d}")
    print(f"\n  All outputs -> {PROCESSED_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
