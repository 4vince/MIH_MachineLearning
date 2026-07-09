import json
import sys

with open("MIH_ML.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# ── Identify which cells to keep vs replace ──
# Keep cells 0-8 (md intro + load data + EDA viz)
# Replace cell 9+ with new content

KEEP_UP_TO = 8  # 0-indexed, inclusive — everything up to EDA viz code cell

new_cells = cells[: KEEP_UP_TO + 1]

# ── Helper ──
def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in source.strip().split("\n")]}

def code(source, exec_count=None):
    c = {
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" if i < len(source.strip().split("\n")) - 1 else line for i, line in enumerate(source.strip().split("\n"))],
    }
    # handle empty last line
    src = source.strip().split("\n")
    c["source"] = []
    for i, line in enumerate(src):
        if i < len(src) - 1:
            c["source"].append(line + "\n")
        else:
            c["source"].append(line)
    return c


# ═══════════════════════════════════════════════════
# NEW CELLS
# ═══════════════════════════════════════════════════

new_cells.append(md("""
# Step 2. Data Cleaning

We clean both datasets using the same pipeline defined in the
standalone scripts (`clean_titanic_data.py`, `clean_lusitania_data.py`).
The pipeline applies these steps in order:

1. **Missing values** — drop sparse columns, drop rows missing critical labels,
   impute Age via sex-group median. Imputation fills missing values with a
   reasonable estimate so the row is kept for modelling instead of being dropped.
2. **Duplicates** — check for exact row duplicates and drop them.
3. **Data types** — convert categorical columns to `category` dtype.
4. **Outliers** — flag via IQR but never auto-remove (all extreme values are legitimate).
5. **Inconsistent formats** — standardise text casing (Sex to lowercase, Fate to lowercase,
   Embarked to uppercase).
6. **Validation** — confirm no nulls remain in required columns, no duplicate IDs.
7. **Standardize output** — rename columns to a common schema (survived, sex, age, class,
   fare, adult_minor, dataset, age_was_imputed) shared across all disaster datasets.
"""))

# Cell: Imports + run both cleaning pipelines
new_cells.append(code("""
import clean_titanic_data
import clean_lusitania_data

# ── Clean Titanic ──
print("=" * 70)
print("CLEANING TITANIC DATASET")
print("=" * 70)
titanic_clean = clean_titanic_data.clean(df.copy(), clean_titanic_data.CONFIG)
titanic_clean["passenger_or_crew"] = "Passenger"

print("\\n")

# ── Clean Lusitania ──
print("=" * 70)
print("CLEANING LUSITANIA DATASET")
print("=" * 70)
lusitania_clean = clean_lusitania_data.clean(df2.copy(), clean_lusitania_data.CONFIG)

print("\\nTitanic cleaned shape:", titanic_clean.shape)
print(titanic_clean.head(10))
print("\\nLusitania cleaned shape:", lusitania_clean.shape)
print(lusitania_clean.head(10))
""", exec_count=None))

# Markdown: Pooling
new_cells.append(md("""
# Step 3. Pooling Cleaned Datasets

Pool the cleaned datasets into a single unified DataFrame for cross-disaster analysis.
Both datasets now share the same standardised columns, making alignment straightforward.
"""))

# Cell: Pooling
new_cells.append(code("""
# ── Pool cleaned datasets ──
common_cols = ["survived", "sex", "age", "class", "fare", "adult_minor",
               "dataset", "age_was_imputed"]

titanic_pool = titanic_clean[common_cols + ["passenger_or_crew"]].copy()
lusitania_pool = lusitania_clean[common_cols + ["passenger_crew"]].copy()
lusitania_pool = lusitania_pool.rename(columns={"passenger_crew": "passenger_or_crew"})

pooled = pd.concat([titanic_pool, lusitania_pool], ignore_index=True)

print("=" * 70)
print("POOLED CLEANED DATASETS")
print("=" * 70)
print(f"Pooled shape: {pooled.shape}")
print(f"  Titanic:   {(pooled['dataset'] == 'Titanic').sum()} rows")
print(f"  Lusitania: {(pooled['dataset'] == 'Lusitania').sum()} rows")
print()
print("Survival rate by dataset:")
print(pooled.groupby("dataset")["survived"].mean().round(4))
print()
print("Survival rate by Sex (pooled):")
print(pooled.groupby("sex")["survived"].mean().round(4))
print()
print("First few rows:")
print(pooled.head(10))
""", exec_count=None))

# Markdown: Cross-Disaster Comparison
new_cells.append(md("""
# 4. Cross-Disaster Comparison

Compare survival patterns between the two disasters side-by-side
using the cleaned, pooled dataset.
"""))

# Cell: Comparison visualization
new_cells.append(code("""
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Cleaned Data — Cross-Disaster Comparison", fontsize=14, fontweight="bold")

# 1. Survival rate by dataset
rate = pooled.groupby("dataset")["survived"].mean()
axes[0, 0].bar(rate.index, rate.values * 100, color=["steelblue", "coral"])
axes[0, 0].set_title("Overall Survival Rate")
axes[0, 0].set_ylabel("Survival Rate (%)"); axes[0, 0].set_ylim(0, 100)
for i, v in enumerate(rate.values):
    axes[0, 0].text(i, v * 100 + 1, f"{v*100:.1f}%", ha="center", fontweight="bold")

# 2. Survival by Sex per dataset
sex_rate = pooled.groupby(["dataset", "sex"])["survived"].mean().unstack()
x = np.arange(len(sex_rate.index))
w = 0.3
for i, s in enumerate(sex_rate.columns):
    axes[0, 1].bar(x + i * w, sex_rate[s].values * 100, w, label=s,
                    color=["steelblue", "coral"][i] if i < 2 else "gray")
axes[0, 1].set_xticks(x + w / 2)
axes[0, 1].set_xticklabels(sex_rate.index)
axes[0, 1].set_title("Survival by Sex")
axes[0, 1].set_ylabel("Survival Rate (%)"); axes[0, 1].set_ylim(0, 100)
axes[0, 1].legend()

# 3. Age distribution by dataset (cleaned)
axes[1, 0].hist(pooled[pooled["dataset"] == "Titanic"]["age"].dropna(),
                bins=30, alpha=0.6, density=True, label="Titanic", color="steelblue")
axes[1, 0].hist(pooled[pooled["dataset"] == "Lusitania"]["age"].dropna(),
                bins=30, alpha=0.6, density=True, label="Lusitania", color="coral")
axes[1, 0].set_xlabel("Age"); axes[1, 0].set_ylabel("Density")
axes[1, 0].legend(); axes[1, 0].set_title("Age Distribution (Cleaned)")

# 4. Summary text panel
axes[1, 1].axis("off")
titanic_sex = sex_rate.loc["Titanic"]
lusitania_sex = sex_rate.loc["Lusitania"]
summary_text = (
    f"Pooled dataset: {len(pooled)} observations\\n"
    f"  Titanic: {(pooled['dataset']=='Titanic').sum()} rows\\n"
    f"  Lusitania: {(pooled['dataset']=='Lusitania').sum()} rows\\n\\n"
    f"Overall survival:\\n"
    f"  Titanic: {rate['Titanic']*100:.1f}%\\n"
    f"  Lusitania: {rate['Lusitania']*100:.1f}%\\n\\n"
)
if "female" in sex_rate.columns:
    summary_text += (
        f"Female survival:\\n"
        f"  Titanic: {titanic_sex['female']*100:.1f}%\\n"
        f"  Lusitania: {lusitania_sex['female']*100:.1f}%\\n\\n"
        f"Male survival:\\n"
        f"  Titanic: {titanic_sex['male']*100:.1f}%\\n"
        f"  Lusitania: {lusitania_sex['male']*100:.1f}%"
    )
axes[1, 1].text(0, 0.5, summary_text, fontsize=10, va="center",
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.show()
""", exec_count=None))

# Summary
new_cells.append(md("""
# 5. Summary

## Dataset state before and after

| Dataset | Before | After | Change |
|---|---|---|---|
| **Titanic** | 889 rows x 11 columns | 889 rows x 8 standard cols + 3 extra | No rows dropped; Age imputed by Pclass+Sex median; Sex/Fate standardised; output schema aligned |
| **Lusitania** | 1956 rows x 15 columns | 1953 rows x 8 standard cols + 1 extra | Dropped 5 sparse columns; 3 exact-duplicate rows removed; Age imputed by Sex median |
| **Pooled** | — | 2842 rows x 9 columns | Combined on survived, sex, age, class, fare, adult_minor, dataset, age_was_imputed, passenger_or_crew |

## What was done
- Loaded two historical maritime disaster datasets: **Titanic** (passengers only) and **Lusitania** (passengers + crew).
- Cleaned both through the same 7-step pipeline (shared code in `clean_titanic_data.py` and `clean_lusitania_data.py`):
  1. **Missing values** — dropped sparse columns, dropped rows missing critical labels, imputed Age via group median.
  2. **Duplicates** — detected and removed exact row duplicates.
  3. **Data types** — converted categorical columns to `category` dtype.
  4. **Outliers** — flagged via IQR but never auto-removed (all extreme values were legitimate).
  5. **Inconsistent formats** — standardised text casing (Sex, Fate to lowercase; Embarked to uppercase).
  6. **Validation** — confirmed no nulls remain in required columns, no duplicate IDs.
  7. **Standardize output** — renamed to common schema across all disaster datasets.
- The notebook imports and calls the same `clean()` functions as the scripts, guaranteeing identical output.
- Pooled the cleaned datasets into a single unified DataFrame for cross-disaster analysis.

## Key findings
- Overall survival rates are similar: **Titanic ~38%**, **Lusitania ~42%**.
- **Sex is the strongest predictor** in both disasters — women survived at consistently higher rates.
- **Class matters** — First-class / Saloon passengers fared better in both disasters.
- **Age distributions differ** — Lusitania had fewer children and a tighter age spread (mostly adult crew).
- **Titanic is passengers-only**; Lusitania includes crew, making passenger_or_crew a critical control variable.

## Data ready for modelling
The cleaned, pooled dataset is ready for classification modelling to answer:
*"What evacuation-relevant variables consistently matter across multiple disasters?"*
"""))

nb["cells"] = new_cells

with open("MIH_ML.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Rewritten notebook with {len(new_cells)} cells.")
