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

# Get current file directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path of trained model
MODEL_PATH = os.path.join(BASE_DIR, "../saved_models/best_spiral_model.h5")

# Load CNN model
model = load_model(MODEL_PATH)


# ============================================
# IMAGE PREPROCESSING FUNCTION
# ============================================

def preprocess_image(image_path):
    """
    This function loads the image and prepares it
    for the CNN model.
    """

    # Read image
    img = cv2.imread(image_path)

    # Check if image exists
    if img is None:
        raise ValueError("Invalid image file. Cannot read image.")

    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image (same size used during training)
    img = cv2.resize(img, (128, 128))

    # Normalize pixel values
    img = img / 255.0

    # Reshape for CNN input
    img = img.reshape(1, 128, 128, 1)

    return img


# ============================================
# CHECK IF IMAGE IS SPIRAL-LIKE
# ============================================

def validate_spiral_image(image_path):
    """
    This checks if the uploaded image is likely
    to contain a spiral drawing.
    """

    img = cv2.imread(image_path, 0)

    if img is None:
        return False

    # Detect edges
    edges = cv2.Canny(img, 50, 150)

    # Count edge pixels
    edge_pixels = np.sum(edges > 0)

    # If edges are too low → probably not a drawing
    if edge_pixels < 500:
        return False

    return True


# ============================================
# PREDICTION FUNCTION
# ============================================

def predict_spiral(image_path):
    """
    Predict Parkinson's disease from spiral image
    """

    # Check if image exists
    if not os.path.exists(image_path):
        return {
            "error": "Image file not found."
        }

    # Validate spiral image
    if not validate_spiral_image(image_path):
        return {
            "error": "Uploaded image does not appear to be a valid spiral drawing."
        }

    try:

        # Preprocess image
        img = preprocess_image(image_path)

        # Predict using CNN model
        prediction = model.predict(img)[0][0]

        # Convert prediction to label
        if prediction > 0.5:
            result = "parkinson"
            confidence = float(prediction * 100)
        else:
            result = "healthy"
            confidence = float((1 - prediction) * 100)

        return {
            "result": result,
            "confidence": round(confidence, 2)
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# ============================================
# COMMAND LINE TESTING
# ============================================

if __name__ == "__main__":

    import sys

    # Check if image path is provided
    if len(sys.argv) < 2:
        print("Usage: python predict_spiral.py <image_path>")
        sys.exit()

    image_path = sys.argv[1]

    output = predict_spiral(image_path)

    # Print result
    print("\nPrediction Result:")
    print(output)