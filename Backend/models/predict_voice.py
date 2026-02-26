import pickle
import numpy as np
import os

# Get current directory path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct model paths
model_path = os.path.join(BASE_DIR, "svm_voice_model.pkl")
scaler_path = os.path.join(BASE_DIR, "voice_scaler.pkl")

# Load model and scaler
model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

def predict_voice(input_data):
    try:
        features = np.array([input_data])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]

        confidence = round(max(probability) * 100, 2)

        result = "Parkinson's Detected" if prediction == 1 else "Healthy"

        return {
            "prediction": result,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}