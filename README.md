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
variables like `sex` and `passenger class` behave the same way in both.

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

## Repository structure

```
.
├── titanic.csv                  # Raw Titanic dataset (passengers only)
├── LusitaniaManifest.csv        # Raw Lusitania manifest (passengers + crew)
└── README.md
```

## Key variables

| Column | Meaning | Hypothesis it tests |
|---|---|---|
| `survived` | 1 = survived, 0 = did not | Target variable |
| `sex` | Passenger sex | "Women and children first" norm |
| `pclass` | Ticket class (1/2/3, Lusitania's Saloon/Second/Third recoded to match) | Socio-economic access to lifeboats |
| `age` | Passenger age | Age-based prioritization |
| `disaster` | Titanic or Lusitania | Which disaster the row belongs to |
| `time_to_sink_min` | Minutes from incident to sinking (160 vs. 18) | Moderator: does evacuation speed change which norms hold? |


## Status

- [x] Load, clean, and filter Titanic dataset
- [ ] Load, clean, and filter Lusitania dataset (passengers isolated from crew)
- [ ] Pool both datasets into a shared schema
- [ ] Impute missing age within the pooled dataset
- [ ] Fit logistic regression with `sex_female × time_to_sink_min` interaction term
- [ ] Cross-check with Random Forest + SHAP
- [ ] Write up findings: which patterns are structural vs. context-bound

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
