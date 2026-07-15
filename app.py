"""
Streamlit App — Cross-Disaster Survival Exploration
====================================================
Explore how passenger characteristics and evacuation speed affect survival
outcomes across the Titanic and Lusitania disasters.

This is a research tool for understanding structural evacuation patterns,
NOT a predictor for real individuals in future disasters.
"""

import streamlit as st
from prediction import predict_survival

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maritime Survival Explorer",
    page_icon="🚢",
    layout="centered",
)

st.title("Maritime Survival Explorer")
st.caption("Cross-Disaster Survival Analysis — Titanic vs Lusitania")

st.info(
    "This tool explores how passenger characteristics and evacuation speed "
    "affected survival in historical maritime disasters. It is **not** intended "
    "to predict outcomes for real individuals in future events."
)

# ── Sidebar inputs ─────────────────────────────────────────────────────────
st.sidebar.header("Explore a Passenger Profile")

disaster_label = st.sidebar.selectbox(
    "Disaster",
    ["Titanic (1912)", "Lusitania (1915)"],
    help="Titanic sank in ~160 min, Lusitania in ~18 min",
)
disaster = "Titanic" if "Titanic" in disaster_label else "Lusitania"

sex_label = st.sidebar.radio("Sex", ["Female", "Male"])
sex = "female" if sex_label == "Female" else "male"

pclass = st.sidebar.selectbox(
    "Passenger Class",
    ["First", "Second", "Third"],
)

age = st.sidebar.slider(
    "Age",
    min_value=0,
    max_value=80,
    value=30,
    help="Passenger age in years",
)

# ── Predict ────────────────────────────────────────────────────────────────
result = predict_survival(age=age, sex=sex, pclass=pclass, disaster=disaster)
survived_prob = result["survived_prob"]
died_prob = result["died_prob"]

# ── Display results ────────────────────────────────────────────────────────
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Survival Probability", value=f"{survived_prob:.1%}")

with col2:
    st.metric(label="Did Not Survive", value=f"{died_prob:.1%}")

st.progress(survived_prob, text=f"Survival likelihood: {survived_prob:.1%}")

# Interpretation
st.divider()
st.subheader("What this tells us")

if survived_prob >= 0.7:
    st.success(f"**This profile has high estimated survival** ({survived_prob:.1%})")
elif survived_prob >= 0.4:
    st.warning(f"**This profile has moderate estimated survival** ({survived_prob:.1%})")
else:
    st.error(f"**This profile has low estimated survival** ({survived_prob:.1%})")

# Key factors
f = result["features"]
st.markdown(f"""
**Key factors for this passenger:**
- **Disaster**: {disaster} (sank in {f['time_to_sink_min']:.0f} min)
- **Sex**: {sex_label} (sex_male = {f['sex_male']:.0f})
- **Class**: {pclass} (encoded as {f['class']:.0f})
- **Age**: {age} years
- **Interaction (female x time)**: {f['sex_x_time']:.0f}
""")

# ── Feature importance note ────────────────────────────────────────────────
st.divider()
with st.expander("About the model"):
    st.markdown("""
    **Model**: Random Forest (300 trees, unlimited depth, min leaf=10)

    **Purpose**: Identify which passenger characteristics and evacuation
    conditions consistently affect survival — useful for thinking about
    safety design and protocol, not for forecasting individual outcomes.

    **Features used**:
    - `age` — Passenger age
    - `time_to_sink_min` — Minutes from incident to sinking (160 vs 18)
    - `sex_male` — 1 if male, 0 if female
    - `class` — Passenger class (0=First, 1=Second, 2=Third)
    - `sex_x_time` — Interaction: being female x time available for evacuation

    **Key finding**: The survival advantage of being female increases with
    slower sinking (OR = 2.56, p < 0.001). This suggests "women and children
    first" is a time-dependent norm -- it reasserts itself when people have
    time to coordinate, but erodes under extreme time pressure. Design
    implication: evacuation systems that buy more time may indirectly protect
    vulnerable groups.

    **General principle or artifact?**: This pattern holds across two different
    disasters (peacetime vs wartime, different nationalities), suggesting it is
    structural rather than a Titanic-specific artifact. However, both ships are
    from the same era (1912-1915) with similar social norms, so the finding may
    not generalize to modern or non-Western contexts.

    **Limitations**: This model is trained on 1912/1915 data. It captures
    era-specific social norms and ship layouts. It is not generalizable
    to modern disasters or different cultural contexts.

    **Data**: 889 Titanic passengers + 1,263 Lusitania passengers (excluding crew)
    """)
