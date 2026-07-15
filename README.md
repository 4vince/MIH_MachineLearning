# Cross-Disaster Survival Analysis: What Actually Predicts Who Lives?

## Overview

This project starts with the classic Titanic survival-prediction problem, but
reframes the goal. Instead of asking *"can we predict whether a given
passenger survived?"*, it asks a more useful question:

> **Which passenger and situational characteristics consistently and
> meaningfully affect survival during a maritime evacuation — and does
> their importance hold across different disasters, or is it dependent
> on the disaster's own characteristics?**

A model trained only on the Titanic can't say whether "women and children
first" is a general rule or a one-off historical artifact. To find out, this
project pools passenger-level data from **two** maritime disasters — the
RMS Titanic (1912) and the RMS Lusitania (1915) — and tests whether
variables like `sex` and `pclass` behave the same way in both.

## Why two disasters, and why these two

The Titanic and Lusitania are an unusually clean natural experiment:
similar ship size, similar overall death rate (~69% vs ~67%), only three
years apart — but with one major difference: **the Titanic took ~2h40m to
sink, while the Lusitania sank in ~18 minutes.**

That time difference is the variable this project tests as a *moderator*:
does a slower disaster give social norms (like prioritizing women and
children) more time to reassert themselves, while a fast one favors
"selfish rationality"? Prior published research (Frey, Savage & Torgler,
PNAS) found evidence for exactly this pattern — this project re-derives
and tests that finding from the raw passenger-level data.

## Research approach

1. **Clean and standardize** both datasets independently (handle missing
   values, fix inconsistent categories, recode disaster-specific fields
   into a shared schema).
2. **Pool** the two cleaned datasets into one table, tagging each row with
   its `disaster` and `time_to_sink_min`.
3. **Model** survival with an emphasis on *interpretability over raw
   accuracy* — primarily logistic regression with an explicit interaction
   term (`sex_female × time_to_sink_min`), cross-checked against a Random
   Forest + SHAP analysis to catch anything nonlinear.
4. **Interpret**, not just report accuracy — the goal is to identify which
   patterns are structural (likely to generalize to future evacuation
   design) versus context-bound (artifacts of 1912–1915 social norms or
   ship layout).

## Why these two models?

**Logistic Regression (primary):**
- Coefficients are directly interpretable: "being female multiplies
  survival odds by X, holding class and age constant"
- Explicit interaction terms (`sex_female × time_to_sink_min`) let us
  directly test the moderation hypothesis from the Frey/Savage/Torgler
  research
- Provides p-values and confidence intervals — useful for saying a variable
  *matters*, not just that it correlates
- Standard tool in the academic literature on this topic (the PNAS paper
  used regression-based approaches, not black-box ML)

**Random Forest + SHAP (secondary/validation):**
- Feature importance rankings act as a sanity check on what logistic
  regression found
- SHAP values explain individual predictions ("this passenger's low
  survival probability was driven mostly by Pclass=3, then Age")
- Catches nonlinear effects or unexpected interactions that logistic
  regression might miss
- If both models agree on which variables matter most, that's strong
  evidence the finding is real, not a modeling artifact

## Repository structure

```
.
├── data/
│   ├── raw/
│   │   ├── Titanic-Dataset.csv            # Raw Titanic dataset
│   │   └── LusitaniaManifest.csv          # Raw Lusitania manifest
│   └── cleaned/
│       ├── titanic_cleaned.csv            # Cleaned Titanic output
│       └── lusitania_cleaned.csv          # Cleaned Lusitania output
│
├── src/                                   # Source code
│   ├── clean_titanic_data.py              # Clean and standardize Titanic data
│   ├── clean_lusitania_data.py            # Clean and standardize Lusitania data
│   ├── feature_pipeline.py                # Encode, scale, split → processed/
│   ├── model_strategy.py                  # Logistic Regression + Random Forest + SHAP
│   ├── evaluation.py                      # Confusion matrices, ROC, residuals, calibration
│   ├── baseline_model.py                  # Initial baseline (before interaction term)
│   └── prediction.py                      # Reusable prediction interface
│
├── app/
│   ├── app.py                             # Streamlit app for exploration
│   └── requirements.txt                   # Dependencies
│
├── processed/                             # Train/test splits (parquet + npy)
│
├── results/                               # Model outputs
│   ├── final_model.joblib                 # Trained Random Forest
│   ├── logistic_coefficients.csv          # LR coefficients with p-values
│   ├── shap_importance.csv                # RF feature importance
│   ├── evaluation_metrics.csv             # Full metric report
│   └── plots/                             # Confusion matrices, ROC, residuals, calibration
│
├── docs/
│   ├── BASELINE.md                        # Baseline model documentation
│   └── MODEL_COMPARISON.md                # Baseline vs final model comparison
│
├── MIH_ML.ipynb                           # Interactive notebook for exploration
└── README.md
```

## Running the app

```bash
python -m streamlit run app/app.py
```

## Running the scripts

All scripts should be run from the **project root** directory:

```bash
# Step 1: Clean raw data
python src/clean_titanic_data.py
python src/clean_lusitania_data.py

# Step 2: Run feature pipeline (produces processed/)
python src/feature_pipeline.py

# Step 3: Train models
python src/model_strategy.py

# Step 4: Generate evaluation plots
python src/evaluation.py
```

## Important caveat

This project explicitly does **not** aim to predict any specific future
individual's survival odds. The goal is to identify general, transferable
evacuation-relevant factors — useful for thinking about safety design and
protocol, not for forecasting outcomes for a real person in a future
disaster. Pooling also assumes the two ships are comparable except for the
variables modeled here; other differences (wartime vs. peacetime context,
nationality mix, era-specific ticketing) aren't captured and are noted as
a limitation rather than resolved.

## Data sources

- Titanic: https://www.kaggle.com/datasets/yasserh/titanic-dataset
- Lusitania: https://www.kaggle.com/datasets/rkkaggle2/rms-lusitania-complete-passenger-manifest
