"""
Model Strategy — Two-Model Approach
====================================
1. Logistic Regression (statsmodels) — primary, interpretable, with p-values
2. Random Forest (sklearn) + SHAP — secondary validation

Focus: Does sex × time_to_sink_min interaction predict survival?
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix,
)
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
PROCESSED_DIR = HERE / "processed"
RESULTS_DIR = HERE / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_pooled_data():
    """Load pooled train/test splits."""
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train_pooled.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test_pooled.parquet")
    y_train = np.load(PROCESSED_DIR / "y_train_pooled.npy")
    y_test = np.load(PROCESSED_DIR / "y_test_pooled.npy")

    print(f"Pooled data loaded:")
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Features: {X_train.columns.tolist()}")
    print(f"  Survival rate: train={y_train.mean():.1%} test={y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# 2.  LOGISTIC REGRESSION (statsmodels — with p-values)
# ══════════════════════════════════════════════════════════════════════════════

def fit_logistic_regression_sm(X_train, y_train):
    """Fit logistic regression with statsmodels for full coefficient output."""
    X_const = sm.add_constant(X_train)
    model = sm.Logit(y_train, X_const).fit(disp=0)
    return model


def print_logistic_summary(model):
    """Print the full statsmodels summary."""
    print("\n" + "=" * 70)
    print("  LOGISTIC REGRESSION — Full Summary")
    print("=" * 70)
    print(model.summary())

    # Extract coefficient table
    params = model.params
    conf = model.conf_int()
    pvalues = model.pvalues

    coef_df = pd.DataFrame({
        "variable": params.index,
        "coef": params.values,
        "abs_coef": np.abs(params.values),
        "odds_ratio": np.exp(params.values),
        "ci_lower": conf[0].values,
        "ci_upper": conf[1].values,
        "p_value": pvalues.values,
        "significant_05": pvalues.values < 0.05,
    })

    print("\n" + "=" * 70)
    print("  COEFFICIENT TABLE (with Odds Ratios)")
    print("=" * 70)
    print(coef_df.to_string(index=False))

    return coef_df


def interpret_interaction(coef_df):
    """Interpret the interaction term coefficient."""
    print("\n" + "=" * 70)
    print("  INTERPRETATION: Sex × Time_to_Sink Interaction")
    print("=" * 70)

    interaction = coef_df[coef_df["variable"] == "sex_x_time"]
    if interaction.empty:
        print("  Interaction term not found.")
        return

    row = interaction.iloc[0]
    or_val = row["odds_ratio"]
    p_val = row["p_value"]
    sig = "YES" if row["significant_05"] else "NO"

    print(f"""
  Interaction coefficient:  {row['coef']:.4f}
  Odds ratio:               {or_val:.4f}
  95% CI:                   [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]
  p-value:                  {p_val:.4f}
  Significant at alpha=0.05: {sig}

  What this means for evacuation design:
  - OR > 1: Women's survival advantage grows with more evacuation time.
    This suggests "women and children first" is a TIME-DEPENDENT norm --
    it reasserts itself when people have time to coordinate, but erodes
    under extreme time pressure. Design implication: evacuation systems
    that buy more time (e.g., faster muster, more lifeboats) may
    indirectly protect vulnerable groups by allowing social norms to
    reassert themselves.

  - OR < 1: Women's survival advantage shrinks with more time.
    This would suggest the norm weakens when people can deliberate --
    possibly due to competing self-interest. Design implication: speed
    matters less than clear protocol.

  - OR ~ 1: Evacuation speed does NOT moderate the sex effect.
    This would suggest "women and children first" is either always
    enforced or never enforced, regardless of time pressure.
""")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  RANDOM FOREST + SHAP (validation)
# ══════════════════════════════════════════════════════════════════════════════

def fit_random_forest(X_train, y_train):
    """Fit Random Forest for validation."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_rf(model, X_test, y_test):
    """Evaluate Random Forest on test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("\n" + "=" * 70)
    print("  RANDOM FOREST — Test Set Performance")
    print("=" * 70)
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return metrics


def shap_analysis(model, X_train, X_test, feature_names):
    """Run SHAP analysis on Random Forest."""
    print("\n" + "=" * 70)
    print("  SHAP ANALYSIS — Random Forest Feature Importance")
    print("=" * 70)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle different SHAP output shapes
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif shap_values.ndim == 3:
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


def plot_shap_comparison(lr_coef_df, shap_df, save_dir):
    """Compare logistic regression coefficients with SHAP importance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Logistic Regression: coefficient magnitudes ---
    lr_plot = lr_coef_df[lr_coef_df["variable"] != "const"].copy()
    lr_plot["abs_coef"] = lr_plot["coef"].abs()
    lr_plot = lr_plot.sort_values("abs_coef", ascending=True)

    colors_lr = ["#2ecc71" if c > 0 else "#e74c3c" for c in lr_plot["coef"]]
    axes[0].barh(lr_plot["variable"], lr_plot["coef"], color=colors_lr, edgecolor="black")
    axes[0].axvline(x=0, color="black", linewidth=0.8)
    axes[0].set_xlabel("Coefficient (log-odds)")
    axes[0].set_title("Logistic Regression Coefficients")
    axes[0].grid(True, alpha=0.3)

    # Add significance markers
    for i, (_, row) in enumerate(lr_plot.iterrows()):
        sig = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*" if row["p_value"] < 0.05 else "ns"
        axes[0].text(row["coef"], i, f" {sig}", va="center", fontsize=9)

    # --- Random Forest: SHAP importance ---
    shap_sorted = shap_df.sort_values("mean_abs_shap", ascending=True)
    axes[1].barh(shap_sorted["feature"], shap_sorted["mean_abs_shap"], color="#3498db", edgecolor="black")
    axes[1].set_xlabel("Mean |SHAP| value")
    axes[1].set_title("Random Forest SHAP Importance")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Model Comparison: LR Coefficients vs RF SHAP", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / "shap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: shap_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CONCORDANCE CHECK
# ══════════════════════════════════════════════════════════════════════════════

def concordance_check(lr_coef_df, shap_df):
    """Check if LR and RF agree on which features matter most."""
    print("\n" + "=" * 70)
    print("  CONCORDANCE CHECK: Do both models agree?")
    print("=" * 70)

    # LR ranking by absolute coefficient
    lr_features = lr_coef_df[lr_coef_df["variable"] != "const"].copy()
    lr_features["lr_rank"] = lr_features["abs_coef"].rank(ascending=False)
    lr_ranking = lr_features[["variable", "lr_rank"]].sort_values("lr_rank")

    # SHAP ranking
    shap_df_copy = shap_df.copy()
    shap_df_copy["shap_rank"] = shap_df_copy["mean_abs_shap"].rank(ascending=False)
    shap_ranking = shap_df_copy[["feature", "shap_rank"]].rename(columns={"feature": "variable"})
    shap_ranking = shap_ranking.sort_values("shap_rank")

    # Merge
    comparison = pd.merge(lr_ranking, shap_ranking, on="variable", how="outer")
    comparison["avg_rank"] = comparison[["lr_rank", "shap_rank"]].mean(axis=1)
    comparison = comparison.sort_values("avg_rank")

    print("\nFeature ranking comparison:")
    print(comparison.to_string(index=False))

    # Check top-3 agreement
    lr_top3 = set(lr_ranking.head(3)["variable"])
    shap_top3 = set(shap_ranking.head(3)["variable"])
    overlap = lr_top3 & shap_top3

    print(f"\n  LR top-3:   {sorted(lr_top3)}")
    print(f"  SHAP top-3: {sorted(shap_top3)}")
    print(f"  Overlap:    {sorted(overlap)} ({len(overlap)}/3)")
    print(f"\n  Conclusion: {'Models AGREE — findings are robust.' if len(overlap) >= 2 else 'Models DISAGREE — investigate further.'}")

    return comparison


# ══════════════════════════════════════════════════════════════════════════════
# 5.  SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def save_results(coef_df, shap_df, rf_metrics, concordance_df):
    """Save all results to files."""
    RESULTS_DIR.mkdir(exist_ok=True)

    coef_df.to_csv(RESULTS_DIR / "logistic_coefficients.csv", index=False)
    shap_df.to_csv(RESULTS_DIR / "shap_importance.csv", index=False)

    # RF metrics
    rf_df = pd.DataFrame([rf_metrics])
    rf_df.to_csv(RESULTS_DIR / "rf_metrics.csv", index=False)

    concordance_df.to_csv(RESULTS_DIR / "concordance_check.csv", index=False)

    print(f"\n  Results saved to {RESULTS_DIR}/")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Model Strategy: Two-Model Approach")
    print("  1. Logistic Regression (statsmodels) — primary")
    print("  2. Random Forest + SHAP — validation")
    print("=" * 70)

    # Load data
    X_train, X_test, y_train, y_test = load_pooled_data()

    # --- Logistic Regression (primary) ---
    lr_model = fit_logistic_regression_sm(X_train, y_train)
    coef_df = print_logistic_summary(lr_model)
    interpret_interaction(coef_df)

    # --- Random Forest (validation) ---
    rf_model = fit_random_forest(X_train, y_train)
    rf_metrics = evaluate_rf(rf_model, X_test, y_test)

    # --- SHAP analysis ---
    shap_df = shap_analysis(rf_model, X_train, X_test, X_train.columns.tolist())

    # --- Comparison ---
    plot_shap_comparison(coef_df, shap_df, PLOTS_DIR)
    concordance_df = concordance_check(coef_df, shap_df)

    # --- Save ---
    save_results(coef_df, shap_df, rf_metrics, concordance_df)

    print(f"\n{'='*70}")
    print("  Model strategy complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
