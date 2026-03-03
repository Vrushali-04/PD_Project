import pickle
import numpy as np
import os

# ==============================
# LOAD MODEL & SCALER
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "svm_voice_model.pkl")
scaler_path = os.path.join(BASE_DIR, "voice_scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

print("✅ Voice model loaded successfully")


# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_voice(input_data):
    try:
        # Validate feature length
        if len(input_data) != 9:
            return {"error": "Expected exactly 9 input features"}

        # Convert to numpy array
        features = np.array(input_data, dtype=float).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict class
        prediction = model.predict(features_scaled)[0]

        # Predict probabilities
        probabilities = model.predict_proba(features_scaled)[0]

        # Confidence = highest probability
        confidence = round(float(np.max(probabilities)) * 100, 2)

        # 🔎 Debug print (you can remove later)
       
        #print("Probabilities:", probabilities)

        # ✅ IMPORTANT:
        # If your training labels were:
        # 0 = Healthy
        # 1 = Parkinson
        result = "healthy" if prediction == 1 else "detected"

        # If output looks reversed after testing,
        # swap the above line with:
        # result = "healthy" if prediction == 1 else "detected"

        #print("Final Result:", result)
        #print("Confidence:", confidence)

        return {
            "prediction": result,
            "confidence": confidence
        }

    except Exception as e:
        print("❌ Voice Model Error:", e)
        return {"error": str(e)}