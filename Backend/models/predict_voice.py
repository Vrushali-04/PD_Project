import pickle
import numpy as np
import pandas as pd
import os

# ==============================
# LOAD MODEL & SCALER
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "svm_voice_model.pkl")
scaler_path = os.path.join(BASE_DIR, "voice_scaler.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# ==============================
# FEATURE NAMES (MUST MATCH TRAINING)
# ==============================

feature_names = [
    "MDVP:Fo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Shimmer",
    "HNR",
    "RPDE",
    "DFA",
    "spread1",
    "spread2",
    "PPE"
]

# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_voice(input_data):
    try:
        # Validate feature length
        if len(input_data) != 9:
            return {"error": "Expected exactly 9 input features"}

        # Convert to DataFrame (fixes sklearn feature-name warning)
        features_df = pd.DataFrame(
            [input_data],
            columns=feature_names
        )

        # Scale features
        features_scaled = scaler.transform(features_df)

        # Predict class
        prediction = int(model.predict(features_scaled)[0])

        # Predict probabilities
        probabilities = model.predict_proba(features_scaled)[0]

        # Confidence = highest probability
        confidence = round(float(np.max(probabilities)) * 100, 2)

        # ✅ Correct mapping (Original UCI dataset)
        # 0 = Healthy
        # 1 = Parkinson
        result = "detected" if prediction == 1 else "healthy"

        return {
            "prediction": result,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}