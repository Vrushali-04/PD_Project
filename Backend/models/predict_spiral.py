# =====================================================
# SPIRAL PREDICTION SCRIPT
# =====================================================

import os
import numpy as np
import cv2
import tensorflow as tf

# =====================================================
# LOAD MODEL
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model is inside the same models folder
MODEL_PATH = os.path.join(BASE_DIR, "spiral_parkinson_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Spiral Model Loaded Successfully")

IMG_SIZE = 224
THRESHOLD = 0.5


# =====================================================
# PREPROCESS IMAGE
# =====================================================

def preprocess(image_path):

    if not os.path.exists(image_path):
        raise ValueError("❌ Image path does not exist")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("❌ Invalid image file")

    print("📷 Original Image Shape:", img.shape)

    # Resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# =====================================================
# PREDICT FUNCTION
# =====================================================

def predict_spiral(image_path):

    try:

        img = preprocess(image_path)

        prediction = model.predict(img, verbose=0)[0][0]

        print("🧠 Raw Model Output:", prediction)

        # Correct classification logic
        if prediction >= THRESHOLD:
            result = "parkinson"
            confidence = prediction * 100
        else:
            result = "healthy"
            confidence = (1 - prediction) * 100

        response = {
            "prediction": result,
            "confidence": round(float(confidence), 2)
        }

        print("✅ Prediction:", response)

        return response

    except Exception as e:

        print("❌ Prediction Error:", str(e))

        return {
            "error": "Prediction failed",
            "details": str(e)
        }


# =====================================================
# TEST SCRIPT
# =====================================================

if __name__ == "__main__":

    # Change to any spiral image for testing
    test_image = "test_image.png"

    result = predict_spiral(test_image)

    print(result)