"""
prediction.py — Reusable prediction interface for the survival model.

NOTE: This model is for research and exploration purposes only.
It identifies which passenger characteristics and situational factors
consistently affect survival in maritime evacuations. It is NOT intended
to predict whether any specific future individual would survive.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "results" / "final_model.joblib"
_model = None


def _load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_survival(
    age: float,
    sex: str,
    pclass: str,
    disaster: str,
) -> dict:
    """
    Compute survival probability for a passenger profile.

    Used for exploring how passenger characteristics (sex, class, age)
    and evacuation speed interact to affect survival outcomes. The output
    illustrates structural patterns in the data, not predictions for
    real individuals in future disasters.

    Args:
        age:        Passenger age in years (0-100)
        sex:        "male" or "female"
        pclass:     "First", "Second", or "Third"
        disaster:   "Titanic" or "Lusitania"

    Returns:
        dict with keys:
            survived_prob  — float, estimated P(survived) for this profile
            died_prob      — float, estimated P(did not survive)
            prediction     — int, 1 or 0 based on majority probability
            features       — dict of input features used by the model
    """
    time_to_sink = 160.0 if disaster == "Titanic" else 18.0
    sex_male = 1 if sex == "male" else 0
    sex_female = 1 - sex_male
    class_map = {"First": 0, "Second": 1, "Third": 2}
    class_val = class_map[pclass]

    features = pd.DataFrame([{
        "age": age,
        "time_to_sink_min": time_to_sink,
        "sex_x_time": sex_female * time_to_sink,
        "sex_male": sex_male,
        "class": class_val,
    }])

    model = _load_model()
    proba = model.predict_proba(features)[0]

    return {
        "survived_prob": float(proba[1]),
        "died_prob": float(proba[0]),
        "prediction": int(proba[1] >= 0.5),
        "features": features.iloc[0].to_dict(),
    }


if __name__ == "__main__":
    # Quick demo
    result = predict_survival(age=25, sex="female", pclass="First", disaster="Titanic")
    print(f"Survival prob: {result['survived_prob']:.1%}")
    print(f"Prediction:    {'Survived' if result['prediction'] else 'Did not survive'}")
