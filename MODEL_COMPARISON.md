# Model Comparison: Baseline vs Final

## Summary

| Metric | Baseline LR (3 feats) | Baseline RF (3 feats) | LR + Disaster Context (5 feats) | Final Tuned RF |
|--------|----------------------|----------------------|--------------------------------|----------------|
| **Accuracy** | 0.6334 | 0.6798 | 0.6868 | **0.7262** |
| **Precision** | 0.5238 | 0.6316 | 0.6556 | **0.7368** |
| **Recall** | 0.3374 | 0.3681 | 0.3620 | **0.4294** |
| **F1** | 0.4104 | 0.4651 | 0.4664 | **0.5426** |
| **AUC** | 0.6525 | 0.6971 | 0.7115 | **0.7377** |

## What Changed

### Baseline (3 features)
- `age`, `sex_male`, `class`
- No disaster context — Titanic and Lusitania pooled without distinguishing which ship

### Final Model (5 features)
- `age`, `sex_male`, `class`, `time_to_sink_min`, `sex_x_time`
- Added `time_to_sink_min` (160 min Titanic, 18 min Lusitania)
- Added `sex_x_time` interaction term (sex_female x time_to_sink_min)
- Filtered Lusitania to passengers only (excluded 690 crew)
- Tuned RF: n_estimators=300, max_depth=None, min_samples_leaf=10

### Hyperparameter Tuning
| Parameter | Baseline RF | Final RF |
|-----------|------------|----------|
| n_estimators | 200 | 300 |
| max_depth | 6 | None |
| min_samples_leaf | 5 | 10 |

## Key Finding

The `sex_x_time` interaction term is the most important feature in both models:

- **Logistic Regression**: coef = 0.941, OR = 2.56, p < 0.001
- **Random Forest SHAP**: rank #1 by mean |SHAP|

**General principle or artifact?**

The finding (OR = 2.56) means the survival advantage of being female **increases** with slower sinking. This is consistent with the hypothesis that "women and children first" is a **time-dependent social norm** -- it reasserts itself when people have time to coordinate, but erodes under extreme time pressure.

**What this means for evacuation design:**
- Evacuation systems that buy more time (faster muster, more lifeboats) may indirectly protect vulnerable groups by allowing social norms to reassert themselves
- The norm is not unconditional -- it breaks down under extreme time pressure (18 min vs 160 min)
- This is a structural pattern, not just a Titanic artifact, because it holds across two different disasters with different contexts (peacetime vs wartime, different nationalities)

## Improvement Over Baseline

| Model | AUC Gain | Accuracy Gain |
|-------|----------|---------------|
| LR: baseline -> +disaster context | +0.059 | +0.053 |
| RF: baseline -> +disaster context | +0.029 | +0.006 |
| RF: +disaster context -> tuned | +0.011 | +0.039 |
| **Total: baseline RF -> final RF** | **+0.041** | **+0.046** |

## Files

- `results/final_model.joblib` — trained Random Forest (final)
- `results/logistic_coefficients.csv` — LR coefficient table with p-values
- `results/shap_importance.csv` — RF feature importance
- `results/shap_comparison.png` — LR vs RF feature comparison plot

## Running

```bash
# Feature engineering
python feature_pipeline.py --scenario pooled

# Model training + evaluation
python model_strategy.py

# Full evaluation plots
python evaluation.py

# Streamlit app
python -m streamlit run app.py
```
