import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# =============================
# LOAD MODEL
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "saved_models", "spiral_parkinson_model.h5")

model = load_model(MODEL_PATH)

# =============================
# PREPROCESS IMAGE
# =============================
def preprocess_image(image_path):

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image file")

    # Convert BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to model input size
    img = cv2.resize(img, (224, 224))

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img


# =============================
# PREDICT FUNCTION
# =============================
def predict_spiral(image_path):

    try:

        img = preprocess_image(image_path)

        prediction = model.predict(img)

        probability = float(prediction[0][0])

        if probability > 0.5:
            label = "parkinson"
        else:
            label = "healthy"

        confidence = round(probability * 100, 2)

        return {
            "prediction": label,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}