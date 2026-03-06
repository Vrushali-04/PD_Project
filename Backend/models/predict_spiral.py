# ============================================
# SPIRAL HANDWRITING PREDICTION SCRIPT
# Parkinson's Disease Detection
# ============================================

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model


# ============================================
# LOAD TRAINED MODEL
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "best_spiral_model.h5")

model = load_model(MODEL_PATH)

print("✅ Spiral Model Loaded Successfully")


# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image file. Cannot read image.")

    # Resize to model input size
    img = cv2.resize(img, (224, 224))

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    img = img / 255.0

    # Reshape for CNN
    img = np.expand_dims(img, axis=0)

    return img


# ============================================
# SPIRAL VALIDATION
# ============================================

def validate_spiral_image(image_path):

    img = cv2.imread(image_path, 0)

    if img is None:
        return False

    edges = cv2.Canny(img, 50, 150)

    edge_pixels = np.sum(edges > 0)

    # If almost no edges → not drawing
    if edge_pixels < 300:
        return False

    return True


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_spiral(image_path):

    if not os.path.exists(image_path):
        return {"error": "Image file not found."}

    # Validate drawing
    if not validate_spiral_image(image_path):
        return {
            "error": "Uploaded image does not appear to be a valid spiral drawing."
        }

    try:

        img = preprocess_image(image_path)

        prediction = model.predict(img)[0][0]

        if prediction > 0.5:
            result = "parkinson"
            confidence = float(prediction * 100)
        else:
            result = "healthy"
            confidence = float((1 - prediction) * 100)
        
        return {
            "prediction": result,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {"error": str(e)}


# ============================================
# COMMAND LINE TEST
# ============================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict_spiral.py <image_path>")
        exit()

    image_path = sys.argv[1]

    result = predict_spiral(image_path)

    print("\nPrediction Result:")
    print(result)