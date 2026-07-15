"""
Baseline Model — Cross-Disaster Survival Analysis
==================================================
Logistic Regression + Random Forest with SHAP interpretability.

Scenarios:
  1. Titanic only
  2. Lusitania only
  3. Pooled (both disasters, common features)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score

import shap

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent
PROCESSED_DIR = ROOT / "processed"
RESULTS_DIR = ROOT / "results"

RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_scenario(name: str) -> dict:
    """Load train/test splits for a given scenario."""
    X_train = pd.read_parquet(PROCESSED_DIR / f"X_train_{name}.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / f"X_test_{name}.parquet")
    y_train = np.load(PROCESSED_DIR / f"y_train_{name}.npy")
    y_test = np.load(PROCESSED_DIR / f"y_test_{name}.npy")

    with open(PROCESSED_DIR / f"meta_{name}.json") as f:
        meta = json.load(f)

    print(f"\n{'='*60}")
    print(f"  Scenario: {name.title()}")
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Features: {X_train.columns.tolist()}")
    print(f"  Survival rate: train={y_train.mean():.1%} test={y_test.mean():.1%}")
    print(f"{'='*60}")

    return {
        "name": name,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "meta": meta,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODELS
# ══════════════════════════════════════════════════════════════════════════════

def fit_logistic_regression(X_train, y_train):
    """Fit logistic regression baseline."""
    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def fit_random_forest(X_train, y_train):
    """Fit random forest baseline."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 3.  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, X_test, y_test, model_name: str) -> dict:
    """Compute metrics and print classification report."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n--- {model_name} ---")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SHAP INTERPRETABILITY
# ══════════════════════════════════════════════════════════════════════════════

def shap_analysis(model, X_train, X_test, feature_names, scenario_name: str):
    """Run SHAP analysis and print top features."""
    print(f"\n--- SHAP Analysis: {scenario_name} ---")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle different SHAP output shapes
    if isinstance(shap_values, list):
        # List of 2 arrays (one per class) — take class 1 (survived)
        shap_vals = shap_values[1]
    elif shap_values.ndim == 3:
        # 3D array (samples, features, classes) — take class 1
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values

    # Mean absolute SHAP values per feature
    mean_shap = np.abs(shap_vals).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    print("\nFeature importance (mean |SHAP|):")
    print(importance_df.to_string(index=False))

    return importance_df


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results(results: list[dict]) -> None:
    """Save all results to CSV."""
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for r in results:
        for model_name, metrics in r["metrics"].items():
            row = {"scenario": r["name"], "model": model_name, **metrics}
            rows.append(row)

    df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "baseline_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")
    print(df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Cross-Disaster Survival Analysis — Baseline Models")
    print("  Logistic Regression + Random Forest + SHAP")
    print("=" * 60)

    scenarios = ["titanic", "lusitania", "pooled"]
    all_results = []

    for scenario_name in scenarios:
        data = load_scenario(scenario_name)
        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]
        feature_names = X_train.columns.tolist()

        # --- Logistic Regression ---
        lr_model = fit_logistic_regression(X_train, y_train)
        lr_metrics = evaluate(lr_model, X_test, y_test, "Logistic Regression")

        # --- Random Forest ---
        rf_model = fit_random_forest(X_train, y_train)
        rf_metrics = evaluate(rf_model, X_test, y_test, "Random Forest")

        # --- SHAP (on Random Forest) ---
        shap_importance = shap_analysis(
            rf_model, X_train, X_test, feature_names, scenario_name
        )

        all_results.append({
            "name": scenario_name,
            "metrics": {
                "logistic_regression": lr_metrics,
                "random_forest": rf_metrics,
            },
            "shap": shap_importance,
        })

    # --- Save ---
    save_results(all_results)

    print(f"\n{'='*60}")
    print("  Baseline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
