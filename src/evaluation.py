"""
Evaluation Script — Cross-Disaster Survival Analysis
=====================================================
Confusion matrices, ROC curves, calibration plots, and residual analysis.
"""

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc,
    precision_recall_curve,
    brier_score_loss, log_loss,
    classification_report,
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
HERE = Path(__file__).parent
PROCESSED_DIR = ROOT / "processed"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RANDOM_STATE = 42


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING + MODEL FITTING
# ══════════════════════════════════════════════════════════════════════════════

def load_scenario(name: str) -> dict:
    X_train = pd.read_parquet(PROCESSED_DIR / f"X_train_{name}.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / f"X_test_{name}.parquet")
    y_train = np.load(PROCESSED_DIR / f"y_train_{name}.npy")
    y_test = np.load(PROCESSED_DIR / f"y_test_{name}.npy")
    return X_train, X_test, y_train, y_test


def fit_models(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    return {"Logistic Regression": lr, "Random Forest": rf}


# ══════════════════════════════════════════════════════════════════════════════
# 2.  CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrices(models, X_test, y_test, scenario, save_dir):
    fig, axes = plt.subplots(1, len(models), figsize=(12, 5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        disp = ConfusionMatrixDisplay(cm, display_labels=["Died", "Survived"])
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}\nAccuracy={tp+tn}/{tp+tn+fp+fn} ({(tp+tn)/(tp+tn+fp+fn):.1%})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle(f"Confusion Matrix — {scenario.title()}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / f"confusion_matrix_{scenario}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: confusion_matrix_{scenario}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ROC CURVE
# ══════════════════════════════════════════════════════════════════════════════

def plot_roc_curves(models, X_test, y_test, scenario, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {scenario.title()}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f"roc_curve_{scenario}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: roc_curve_{scenario}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PRECISION-RECALL CURVE
# ══════════════════════════════════════════════════════════════════════════════

def plot_precision_recall(models, X_test, y_test, scenario, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, label=f"{name} (AP = {pr_auc:.3f})", linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {scenario.title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f"precision_recall_{scenario}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: precision_recall_{scenario}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  RESIDUAL / ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def plot_residual_analysis(models, X_test, y_test, scenario, save_dir):
    fig, axes = plt.subplots(1, len(models), figsize=(14, 5))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        residuals = y_test - y_proba

        ax.scatter(y_proba, residuals, alpha=0.5, edgecolors="k", linewidth=0.5, s=40)
        ax.axhline(y=0, color="red", linestyle="--", linewidth=1.5, label="Zero residual")
        ax.set_xlabel("Predicted Probability")
        ax.set_ylabel("Residual (Actual - Predicted)")
        ax.set_title(f"{name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add color bands
        ax.axhspan(-0.1, 0.1, alpha=0.05, color="green")

    plt.suptitle(f"Residual Analysis — {scenario.title()}", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / f"residuals_{scenario}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: residuals_{scenario}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CALIBRATION PLOT
# ══════════════════════════════════════════════════════════════════════════════

def plot_calibration(models, X_test, y_test, scenario, save_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fraction_of_positives, mean_predicted = calibration_curve(y_test, y_proba, n_bins=10)
        brier = brier_score_loss(y_test, y_proba)
        ax.plot(mean_predicted, fraction_of_positives, "s-", label=f"{name} (Brier={brier:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Plot — {scenario.title()}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / f"calibration_{scenario}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: calibration_{scenario}.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  METRIC REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_metric_report(models, X_test, y_test, scenario) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        rows.append({
            "scenario": scenario,
            "model": name,
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            "roc_auc": auc(*roc_curve(y_test, y_proba)[:2]),
            "brier_score": brier_score_loss(y_test, y_proba),
            "log_loss": log_loss(y_test, y_proba),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Evaluation: Confusion Matrix + Residuals + Metrics")
    print("=" * 60)

    all_metrics = []

    for scenario in ["titanic", "lusitania", "pooled"]:
        print(f"\n{'='*60}")
        print(f"  Scenario: {scenario.title()}")
        print(f"{'='*60}")

        X_train, X_test, y_train, y_test = load_scenario(scenario)
        models = fit_models(X_train, y_train)

        # Plots
        plot_confusion_matrices(models, X_test, y_test, scenario, PLOTS_DIR)
        plot_roc_curves(models, X_test, y_test, scenario, PLOTS_DIR)
        plot_precision_recall(models, X_test, y_test, scenario, PLOTS_DIR)
        plot_residual_analysis(models, X_test, y_test, scenario, PLOTS_DIR)
        plot_calibration(models, X_test, y_test, scenario, PLOTS_DIR)

        # Metric table
        metrics_df = generate_metric_report(models, X_test, y_test, scenario)
        all_metrics.append(metrics_df)
        print(f"\n{metrics_df.to_string(index=False)}")

    # Save combined metrics
    combined = pd.concat(all_metrics, ignore_index=True)
    combined.to_csv(RESULTS_DIR / "evaluation_metrics.csv", index=False)
    print(f"\n{'='*60}")
    print(f"  All plots saved to: {PLOTS_DIR}/")
    print(f"  Metrics saved to:  {RESULTS_DIR}/evaluation_metrics.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
