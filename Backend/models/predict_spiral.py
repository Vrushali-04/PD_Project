# =====================================================
# SIMPLE SPIRAL PREDICTION PIPELINE
# =====================================================

import os
import numpy as np
import cv2
import tensorflow as tf

# =====================================================
# PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPIRAL_MODEL_PATH = os.path.join(BASE_DIR, "best_spiral_model.keras")

# =====================================================
# LOAD MODEL
# =====================================================

if not os.path.exists(SPIRAL_MODEL_PATH):
    raise FileNotFoundError("Spiral model not found")

spiral_model = tf.keras.models.load_model(SPIRAL_MODEL_PATH)
print("✅ Spiral Model Loaded Successfully")

IMG_SIZE = 224
PARKINSON_THRESHOLD = 0.45  # threshold for parkinson prediction

# =====================================================
# PREPROCESS IMAGE
# =====================================================

def preprocess(image_path: str) -> np.ndarray:
    """Read and preprocess image for model prediction."""
    if not os.path.exists(image_path):
        raise ValueError("Image path does not exist")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image file")

    # Convert grayscale → RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize and normalize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img

# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_spiral(image_path: str) -> dict:
    """
    Predict whether a spiral drawing indicates Parkinson's or healthy.
    Returns a dictionary with status, prediction, and confidence.
    """
    try:
        img = preprocess(image_path)
        prob = spiral_model.predict(img, verbose=0)[0][0]
        print(f"🧠 Parkinson Probability: {prob:.4f}")

        if prob >= PARKINSON_THRESHOLD:
            result = "parkinson"
            confidence = prob
        else:
            result = "healthy"
            confidence = 1 - prob

        # Optional: mark uncertain predictions
        if 0.4 < prob < 0.6:
            status = "uncertain"
        else:
            status = "confident"

        return {
            "status": "success",
            "prediction": result,
            "confidence": round(float(confidence * 100), 2),
            "raw_probability": round(float(prob), 4),
            "model_status": status
        }

    except Exception as e:
        print("❌ Prediction Error:", str(e))
        return {
            "status": "error",
            "message": str(e)
        }

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":
    test_image = "test_image.png"
    result = predict_spiral(test_image)
    print("\n🎯 FINAL RESULT:")
    print(result)