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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "saved_models",
    "best_spiral_model.keras"
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found!")

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model Loaded Successfully")

IMG_SIZE = 224
THRESHOLD = 0.5  # You can adjust later

# =====================================================
# PREPROCESS
# =====================================================

def preprocess(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# =====================================================
# PREDICTION
# =====================================================

def predict_spiral(image_path):

    if not os.path.exists(image_path):
        return {"error": "Image not found"}

    img = preprocess(image_path)

    prediction = model.predict(img)[0][0]

    print("Raw Model Output:", prediction)

    if prediction >= THRESHOLD:
        result = "healthy"
        confidence = prediction * 100
    else:
        result = "parkinson"
        confidence = (1 - prediction) * 100

    return {
        "prediction": result,
        "confidence": round(float(confidence), 2)
    }


# =====================================================
# CLI TEST
# =====================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict_spiral.py image.jpg")
        exit()

    output = predict_spiral(sys.argv[1])

    print("\nPrediction Result:")
    print(output)