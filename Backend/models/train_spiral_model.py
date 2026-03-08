import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ======================
# PATHS
# ======================

TRAIN_DIR = "datasets/handwriting/spiral/training"
TEST_DIR = "datasets/handwriting/spiral/testing"

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20

# ======================
# DATA GENERATOR
# ======================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# ======================
# LOAD PRETRAINED MODEL
# ======================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# ======================
# CUSTOM CLASSIFIER
# ======================

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================
# TRAIN MODEL
# ======================

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS
)

# ======================
# SAVE MODEL
# ======================


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "spiral_parkinson_model.h5")

model.save(MODEL_PATH)

print("✅ Model saved at:", MODEL_PATH)