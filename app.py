"""
Maritime Survival Explorer
==========================
Explore how passenger characteristics and evacuation speed affect survival
in historical maritime disasters.

This is a research tool for understanding structural evacuation patterns,
NOT a predictor for real individuals in future disasters.
"""

import streamlit as st
from prediction import predict_survival

st.set_page_config(page_title="Maritime Survival Explorer", layout="centered")

st.title("Maritime Survival Explorer")
st.caption("Which evacuation patterns are general principles vs historical artifacts?")

# ── Inputs ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Passenger Profile")
    disaster = st.selectbox("Disaster", ["Titanic (160 min)", "Lusitania (18 min)"])
    disaster_key = "Titanic" if "Titanic" in disaster else "Lusitania"
    sex = st.radio("Sex", ["Female", "Male"])
    pclass = st.selectbox("Class", ["First", "Second", "Third"])
    age = st.slider("Age", 0, 80, 30)

# ── Run model ──────────────────────────────────────────────────────────────
result = predict_survival(
    age=age,
    sex=sex.lower(),
    pclass=pclass,
    disaster=disaster_key,
)

# ── Output ─────────────────────────────────────────────────────────────────
st.divider()

col1, col2 = st.columns(2)
col1.metric("Survived", f"{result['survived_prob']:.1%}")
col2.metric("Did not survive", f"{result['died_prob']:.1%}")

# ── Research interpretation ────────────────────────────────────────────────
st.divider()
st.subheader("What this tells us")

if result["survived_prob"] >= 0.7:
    st.success(
        f"**High estimated survival** ({result['survived_prob']:.1%}). "
        f"Being {sex.lower()} on the {disaster_key} strongly predicts survival."
    )
elif result["survived_prob"] >= 0.4:
    st.warning(
        f"**Moderate estimated survival** ({result['survived_prob']:.1%}). "
        f"This profile sits in the ambiguous range."
    )
else:
    st.error(
        f"**Low estimated survival** ({result['survived_prob']:.1%}). "
        f"Being {sex.lower()} on the {disaster_key} does not protect this profile."
    )

# ── General principle or artifact? ─────────────────────────────────────────
st.divider()
with st.expander("Is this a general principle or a historical artifact?"):
    st.markdown("""
    **The key finding**: Women's survival advantage depends on evacuation
    speed (OR = 2.56, p < 0.001). More time = stronger "women and children
    first" norm.

    **Structural?** Yes -- the pattern holds across two different disasters
    (peacetime vs wartime, different nationalities). This suggests it is not
    a Titanic-specific artifact.

    **Limitation?** Both ships are from 1912-1915 with similar social norms.
    The finding may not generalize to modern or non-Western contexts.

    **Design implication**: Evacuation systems that buy more time may
    indirectly protect vulnerable groups by allowing social norms to
    reassert themselves.
    """)
