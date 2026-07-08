"""
clean_lusitania_data.py

Full, reusable data-cleaning pipeline. Walks through the 5 anomaly types
in a fixed order, reports every change made, and writes a validated
cleaned CSV to disk. Never overwrites the raw input file.

Usage:
    python clean_lusitania_data.py                                   # uses defaults below
    python clean_lusitania_data.py LusitaniaManifest.csv lusitania_cleaned.csv
"""

import sys
import pandas as pd


CONFIG = {
    "drop_columns_high_missing": [
        "Position", "Status", "City", "Lifeboat", "Rescue Vessel",
    ],
    "drop_rows_missing": ["Personal name", "Citizenship"],
    "impute_group_median": {
        "Age": ["Sex"],
    },
    "categorical_columns": [
        "Fate", "Title", "Department/Class", "Passenger/Crew",
        "Citizenship", "Adult/Minor", "Sex",
    ],
    "id_column": None,
    "target_column": "Fate",
    "text_columns_to_standardize": {
        "Sex": "lower",
        "Fate": "lower",
    },
    "required_no_nulls": [
        "Fate", "Sex", "Age", "Adult/Minor",
    ],
}


# ---------------------------------------------------------------------------
# STEP 1: MISSING VALUES
# ---------------------------------------------------------------------------
def handle_missing_values(df, config):
    print("\n" + "=" * 70)
    print("1. MISSING VALUES")
    print("=" * 70)
    print("Before:")
    print(df.isnull().sum().sort_values(ascending=False))

    before_shape = df.shape

    cols_to_drop = [c for c in config["drop_columns_high_missing"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    if cols_to_drop:
        print(f"\nDropped columns (too sparse): {cols_to_drop}")

    rows_to_check = [c for c in config["drop_rows_missing"] if c in df.columns]
    if rows_to_check:
        n_before = len(df)
        df = df.dropna(subset=rows_to_check)
        print(f"Dropped {n_before - len(df)} rows missing {rows_to_check}")

    for col, group_cols in config["impute_group_median"].items():
        if col not in df.columns:
            continue
        available_group_cols = [g for g in group_cols if g in df.columns]

        print(f"\n{col} missingness rate by {available_group_cols} (checking randomness):")
        print(
            df.groupby(available_group_cols, observed=True)[col]
            .apply(lambda x: x.isnull().mean().round(2))
        )

        # Flag which rows were imputed BEFORE filling, for transparency
        df[f"{col.lower()}_was_imputed"] = df[col].isnull()

        df[col] = df.groupby(available_group_cols, observed=True)[col].transform(
            lambda x: x.fillna(x.median())
        )
        df[col] = df[col].fillna(df[col].median())  # fallback if a group was 100% missing

    print(f"\nShape change: {before_shape} -> {df.shape}")
    return df


# ---------------------------------------------------------------------------
# STEP 2: DUPLICATED RECORDS
# ---------------------------------------------------------------------------
def handle_duplicates(df, config):
    print("\n" + "=" * 70)
    print("2. DUPLICATED RECORDS")
    print("=" * 70)

    exact_dupes = df.duplicated().sum()
    print("Exact duplicate rows:", exact_dupes)

    id_col = config.get("id_column")
    if id_col and id_col in df.columns:
        id_dupes = df[id_col].duplicated().sum()
        print(f"Duplicate {id_col} values:", id_dupes)
        if id_dupes > 0:
            print("WARNING: duplicate IDs found -- investigate before dropping.")
            print(df[df[id_col].duplicated(keep=False)].sort_values(id_col))

    if exact_dupes > 0:
        df = df.drop_duplicates()
        print(f"Dropped {exact_dupes} exact duplicate rows")

    return df


# ---------------------------------------------------------------------------
# STEP 3: INCORRECT DATA TYPES
# ---------------------------------------------------------------------------
def fix_data_types(df, config):
    print("\n" + "=" * 70)
    print("3. DATA TYPES")
    print("=" * 70)
    print("Before:")
    print(df.dtypes)

    for col in config["categorical_columns"]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print("\nAfter:")
    print(df.dtypes)
    return df


# ---------------------------------------------------------------------------
# STEP 4: OUTLIERS (flag only -- never auto-remove)
# ---------------------------------------------------------------------------
def flag_outliers(df, numeric_cols=None):
    print("\n" + "=" * 70)
    print("4. OUTLIERS (flagged for review, not auto-removed)")
    print("=" * 70)

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        desc = df[col].describe()
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        skew = df[col].skew()
        print(f"\n{col}: min={desc['min']:.2f} max={desc['max']:.2f} skew={skew:.2f} "
              f"IQR bounds=({lower:.2f}, {upper:.2f}) -> {n_outliers} potential outliers")

    return df


# ---------------------------------------------------------------------------
# STEP 5: INCONSISTENT FORMATS
# ---------------------------------------------------------------------------
def standardize_formats(df, config):
    print("\n" + "=" * 70)
    print("5. INCONSISTENT FORMATS")
    print("=" * 70)

    for col, rule in config["text_columns_to_standardize"].items():
        if col not in df.columns:
            continue
        before_vals = df[col].astype(str).unique().tolist()
        stripped = df[col].astype(str).str.strip()
        df[col] = stripped.str.lower() if rule == "lower" else stripped.str.upper()
        after_vals = df[col].unique().tolist()
        print(f"{col} ({rule}): {before_vals} -> {after_vals}")
        df[col] = df[col].astype("category")

    return df


# ---------------------------------------------------------------------------
# STEP 6: VALIDATE
# ---------------------------------------------------------------------------
def validate(df, config):
    print("\n" + "=" * 70)
    print("6. VALIDATION")
    print("=" * 70)

    problems = []

    for col in config["required_no_nulls"]:
        if col in df.columns and df[col].isnull().sum() > 0:
            problems.append(f"{col} still has {df[col].isnull().sum()} nulls")

    id_col = config.get("id_column")
    if id_col and id_col in df.columns and df[id_col].duplicated().any():
        problems.append(f"{id_col} still has duplicates")

    if problems:
        raise AssertionError("Cleaning validation failed:\n" + "\n".join(problems))

    print("PASSED -- no required-column nulls, no duplicate IDs.")
    print("Final shape:", df.shape)


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------
def clean(df, config):
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = handle_missing_values(df, config)
    df = handle_duplicates(df, config)
    df = fix_data_types(df, config)
    df = flag_outliers(df)
    df = standardize_formats(df, config)
    validate(df, config)
    return df


def main(input_path="LusitaniaManifest.csv", output_path="lusitania_cleaned.csv"):
    raw = pd.read_csv(input_path)
    print(f"Loaded {input_path} -> {raw.shape}")

    cleaned = clean(raw, CONFIG)

    cleaned.to_csv(output_path, index=False)
    print(f"\nSaved -> {output_path}")
    return cleaned


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        main()
