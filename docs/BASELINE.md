# Baseline Model Documentation

## What is the baseline?

The baseline establishes a performance floor using only the three features
available in both datasets without any disaster context: `age`, `sex_male`,
and `class`. This is the simplest reasonable model — no interaction terms,
no time-to-sink, no disaster identifier.

The baseline answers: **"How well can we predict survival with just passenger
characteristics, ignoring the disaster itself?"**

If adding disaster context (`time_to_sink_min`, `sex_x_time`) doesn't
improve the model, the interaction hypothesis is unsupported.

## Features

| Feature | Description |
|---------|-------------|
| `age` | Passenger age (standardized) |
| `sex_male` | 1 if male, 0 if female (one-hot encoded) |
| `class` | Passenger class (ordinal: 0=First, 1=Second, 2=Third) |

## Data

- **Train**: 1,721 rows (37.9% survived)
- **Test**: 431 rows (37.8% survived)
- Pooled from Titanic (889) + Lusitania passengers only (1,263)

## Baseline Results

### Logistic Regression

| Metric | Value |
|--------|-------|
| Accuracy | 0.6334 |
| Precision | 0.5238 |
| Recall | 0.3374 |
| F1 | 0.4104 |
| **AUC** | **0.6525** |

### Random Forest

| Metric | Value |
|--------|-------|
| Accuracy | 0.6798 |
| Precision | 0.6316 |
| Recall | 0.3681 |
| F1 | 0.4651 |
| **AUC** | **0.6971** |

## Why AUC as the primary metric?

**AUC (Area Under the ROC Curve)** is the primary metric for this project
because:

1. **Handles class imbalance**: Survival rate is ~38%, so accuracy alone is
   misleading (predicting "died" for everyone gives 62% accuracy). AUC
   measures discrimination ability across all thresholds.

2. **Threshold-independent**: We don't need to pick a cutoff. AUC tells us
   how well the model ranks survivors vs non-survivors overall.

3. **Standard for survival analysis**: The academic literature (including the
   Frey/Savage/Torgler PNAS paper) uses AUC as the primary comparison metric.

4. **Comparability**: AUC allows direct comparison between logistic regression
   and random forest despite their different output scales.

**Why not accuracy?** A model that always predicts "died" would get 62%
accuracy. Our baseline LR gets 63% — barely better than always predicting
the majority class. AUC reveals the model has some discriminative ability
(0.65) even when accuracy looks poor.

**Why not just F1?** F1 depends on a classification threshold, which is
arbitrary for this research. We care about *ranking* passengers by survival
likelihood, not about a specific cutoff.

## What the baseline tells us

The baseline AUC of 0.65-0.70 means passenger characteristics alone (age,
sex, class) have **moderate** predictive power, but leave substantial variance
unexplained. This is expected — survival also depends on location on the
ship, lifeboat access, and other factors not in the data.

The key question is whether adding disaster context improves this baseline,
and by how much.
