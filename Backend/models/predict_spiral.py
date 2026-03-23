# =====================================================
# IMPROVED SPIRAL PREDICTION SCRIPT
# =====================================================

import os
import numpy as np
import cv2
import tensorflow as tf

# =====================================================
# LOAD MODEL
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_spiral_model.keras")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Spiral Model Loaded Successfully")

IMG_SIZE = 224

# 🔥 Adjustable threshold (IMPORTANT)
THRESHOLD = 0.45   # tuned (better than 0.5 in many cases)


# =====================================================
# PREPROCESS IMAGE
# =====================================================

def preprocess(image_path):

    if not os.path.exists(image_path):
        raise ValueError("Image path does not exist")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image file")

    # Convert grayscale → RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize (same as training)
    img = img.astype("float32") / 255.0

    # Expand dims
    img = np.expand_dims(img, axis=0)

    return img


# =====================================================
# PREDICTION FUNCTION
# =====================================================

def predict_spiral(image_path):

    try:
        img = preprocess(image_path)

        prob = model.predict(img, verbose=0)[0][0]

        print(f"🧠 Raw Output (Probability of Parkinson): {prob:.4f}")

        # 🔥 Decision logic
        if prob >= THRESHOLD:
            result = "parkinson"
            confidence = prob
        else:
            result = "healthy"
            confidence = 1 - prob

        # 🔥 Handle uncertain predictions
        if 0.4 < prob < 0.6:
            status = "Uncertain"
        else:
            status = "Confident"

        response = {
            "prediction": result,
            "confidence": round(float(confidence * 100), 2),
            "raw_probability": round(float(prob), 4),
            "status": status
        }

        print("✅ Final Prediction:", response)

        return response

    except Exception as e:

        print("❌ Prediction Error:", str(e))

        return {
            "error": str(e)
        }


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    test_image = "test_image.png"

    result = predict_spiral(test_image)

    print(result)