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

# Check model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

# Load CNN model
model = load_model(MODEL_PATH)

print("✅ Spiral Model Loaded Successfully")


# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_image(image_path):
    """
    Load image and convert it to model input format
    """

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image file. Cannot read image.")

    # Resize image to model input size
    img = cv2.resize(img, (224, 224))

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize pixel values
    img = img.astype("float32") / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# ============================================
# SPIRAL IMAGE VALIDATION
# ============================================

def validate_spiral_image(image_path):
    """
    Basic check if uploaded image contains drawing lines
    """

    img = cv2.imread(image_path, 0)

    if img is None:
        return False

    edges = cv2.Canny(img, 50, 150)

    edge_pixels = np.sum(edges > 0)

    # Very low edges → probably not a drawing
    if edge_pixels < 500:
        return False

    return True


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_spiral(image_path):
    """
    Predict Parkinson's Disease using spiral drawing
    """

    if not os.path.exists(image_path):
        return {"error": "Image file not found."}

    try:

        # Validate drawing
        if not validate_spiral_image(image_path):
            return {
                "error": "Uploaded image does not appear to be a valid spiral drawing."
            }

        # Preprocess image
        img = preprocess_image(image_path)

        # Run prediction
        prediction = model.predict(img)[0][0]

        print("Raw prediction value:", prediction)

        # Convert prediction to label
        if prediction >= 0.5:
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
# COMMAND LINE TESTING
# ============================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("python models/predict_spiral.py <image_path>")
        exit()

    image_path = sys.argv[1]

    result = predict_spiral(image_path)

    print("\nPrediction Result:")
    print(result)
    print(model.summary())