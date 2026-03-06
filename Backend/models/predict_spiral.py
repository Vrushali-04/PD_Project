# ======================================
# LOAD MODEL AND PREDICT SPIRAL IMAGE
# ======================================

import tensorflow as tf
import numpy as np
import cv2

# load trained model
model = tf.keras.models.load_model("saved_models/best_spiral_model.h5")

IMG_SIZE = 224


def predict_spiral(image_path):

    # read image
    img = cv2.imread(image_path)

    if img is None:
        print("Invalid image")
        return

    # resize image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # normalize
    img = img / 255.0

    # reshape
    img = np.expand_dims(img, axis=0)

    # prediction
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        print("Parkinson Detected")
    else:
        print("Healthy")


# test image
predict_spiral("datasets/handwriting/spiral/testing/healthy/V01HE01.png")