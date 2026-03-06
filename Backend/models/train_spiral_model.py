# ===============================================
# SPIRAL PARKINSON DETECTION TRAINING
# Using Transfer Learning (MobileNetV2)
# ===============================================

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ===============================================
# PATHS
# ===============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "..",
    "datasets",
    "handwriting",
    "spiral",
    "training"
)

MODEL_SAVE_PATH = os.path.join(
    BASE_DIR,
    "..",
    "saved_models",
    "best_spiral_model.h5"
)

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# ===============================================
# PARAMETERS
# ===============================================

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 25

# ===============================================
# DATA GENERATOR
# ===============================================

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=15,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

print("Class labels:", train_generator.class_indices)

# ===============================================
# LOAD PRETRAINED MODEL
# ===============================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

# ===============================================
# ADD CLASSIFICATION LAYER
# ===============================================

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)

predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ===============================================
# COMPILE
# ===============================================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================================
# CALLBACKS
# ===============================================

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# ===============================================
# TRAIN MODEL
# ===============================================

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print("\nTraining Finished")
print("Model saved at:", MODEL_SAVE_PATH)

# ===============================================
# FINAL EVALUATION
# ===============================================

loss, acc = model.evaluate(val_generator)

print("\nValidation Accuracy:", round(acc*100,2),"%")