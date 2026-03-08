import os
import numpy as np
import cv2
import tensorflow as tf

# ==============================
# LOAD MODEL
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "spiral_parkinson_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

print("Spiral Model Loaded Successfully")

IMG_SIZE = 224


# ==============================
# PREPROCESS IMAGE
# ==============================

def preprocess(image_path):

    if not os.path.exists(image_path):
        raise ValueError("Image path does not exist")

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Invalid image file")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# ==============================
# PREDICTION FUNCTION
# ==============================

def predict_spiral(image_path):

    try:

        img = preprocess(image_path)

        prediction = model.predict(img, verbose=0)[0][0]

        print("Raw Model Output:", prediction)

        # uncertainty check
        if 0.4 < prediction < 0.6:
            return {
                "error": "Invalid or unrelated image. Please upload a spiral drawing."
            }

        if prediction >= 0.5:
            result = "parkinson"
            confidence = prediction * 100
        else:
            result = "healthy"
            confidence = (1 - prediction) * 100

        return {
            "prediction": result,
            "confidence": round(float(confidence), 2)
        }

    except Exception as e:

        return {
            "error": str(e)
        }


# ==============================
# TEST
# ==============================

if __name__ == "__main__":

    test_image = "test_image.png"

    print(predict_spiral(test_image))