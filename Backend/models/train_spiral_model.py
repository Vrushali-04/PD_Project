# =========================================================
# PARKINSON SPIRAL DETECTION - INDUSTRY LEVEL TRAINING
# =========================================================

import tensorflow as tf
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


# =========================================================
# DATASET PATHS
# =========================================================

train_dir = "datasets/handwriting/spiral/training"
test_dir = "datasets/handwriting/spiral/testing"


# =========================================================
# IMAGE SETTINGS
# =========================================================

IMG_SIZE = (224,224)
BATCH_SIZE = 16


# =========================================================
# DATA AUGMENTATION (VERY IMPORTANT FOR SMALL DATASET)
# =========================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,

    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,

    shear_range=0.15,
    zoom_range=0.25,

    brightness_range=[0.7,1.3],

    horizontal_flip=True,
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1./255)


# =========================================================
# LOAD DATA
# =========================================================

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)


# =========================================================
# LOAD PRETRAINED MODEL
# =========================================================

# EfficientNetV2B0 trained on ImageNet dataset

base_model = EfficientNetV2B0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# freeze initial layers
for layer in base_model.layers:
    layer.trainable = False


# =========================================================
# CUSTOM CLASSIFICATION HEAD
# =========================================================

x = base_model.output

x = layers.GlobalAveragePooling2D()(x)

x = layers.BatchNormalization()(x)

x = layers.Dense(256, activation="relu")(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(64, activation="relu")(x)

output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=base_model.input, outputs=output)


# =========================================================
# COMPILE MODEL
# =========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


# =========================================================
# CALLBACKS (IMPORTANT FOR INDUSTRY MODELS)
# =========================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3
)

checkpoint = ModelCheckpoint(
    "saved_models/best_spiral_model.h5",
    monitor="val_accuracy",
    save_best_only=True
)


# =========================================================
# TRAIN MODEL
# =========================================================

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=30,
    callbacks=[early_stop, reduce_lr, checkpoint]
)


# =========================================================
# FINE TUNING (UNFREEZE LAST LAYERS)
# =========================================================

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Starting Fine Tuning...")

model.fit(
    train_data,
    validation_data=test_data,
    epochs=10
)


# =========================================================
# SAVE FINAL MODEL
# =========================================================

os.makedirs("saved_models", exist_ok=True)

model.save("saved_models/spiral_parkinson_industry_model.h5")

print("Model training completed successfully.")