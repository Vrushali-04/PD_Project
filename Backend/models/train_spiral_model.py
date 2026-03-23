# =====================================================
# IMPROVED SPIRAL + WAVE MODEL TRAINING SCRIPT
# =====================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils import class_weight

# =====================================================
# PATH SETTINGS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(BASE_DIR)

TRAIN_DIR = os.path.join(BACKEND_DIR, "datasets", "handwriting", "training")
TEST_DIR = os.path.join(BACKEND_DIR, "datasets", "handwriting", "testing")

print("Training path:", TRAIN_DIR)
print("Testing path:", TEST_DIR)

# =====================================================
# IMAGE SETTINGS
# =====================================================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30

# =====================================================
# DATA GENERATOR
# =====================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("Class mapping:", train_generator.class_indices)

# =====================================================
# HANDLE CLASS IMBALANCE
# =====================================================

labels = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# =====================================================
# LOAD PRETRAINED MODEL
# =====================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all layers initially
for layer in base_model.layers:
    layer.trainable = False

# =====================================================
# CUSTOM CLASSIFICATION LAYERS
# =====================================================

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# =====================================================
# CALLBACKS
# =====================================================

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        min_lr=1e-6
    ),

    ModelCheckpoint(
        os.path.join(BASE_DIR, "best_spiral_model.keras"),
        monitor="val_loss",
        save_best_only=True
    )
]

# =====================================================
# PHASE 1 TRAINING (Feature Extraction)
# =====================================================

print("\n🚀 Phase 1: Training top layers...\n")

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=15,
    callbacks=callbacks,
    class_weight=class_weights
)

# =====================================================
# PHASE 2 TRAINING (Fine-tuning)
# =====================================================

print("\n🚀 Phase 2: Fine-tuning model...\n")

# Unfreeze last 30 layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# =====================================================
# SAVE MODEL
# =====================================================

MODEL_PATH = os.path.join(BASE_DIR, "spiral_parkinson_model.keras")
model.save(MODEL_PATH)

print("✅ Model saved at:", MODEL_PATH)