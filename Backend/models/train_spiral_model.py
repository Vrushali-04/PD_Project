# ===============================================
# SPIRAL HANDWRITING TRAINING SCRIPT
# Parkinson's Disease Detection
# ===============================================

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
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

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 20

# ===============================================
# DATA GENERATORS
# ===============================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.05,
    height_shift_range=0.05,
    horizontal_flip=False
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

print("\nClass Labels:", train_generator.class_indices)

# ===============================================
# CNN MODEL
# ===============================================

model = Sequential([

    Conv2D(32,(3,3),activation='relu',input_shape=(IMG_SIZE,IMG_SIZE,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128,activation='relu'),
    Dropout(0.5),

    Dense(1,activation='sigmoid')
])

# ===============================================
# COMPILE MODEL
# ===============================================

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================================
# CALLBACKS
# ===============================================

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ===============================================
# TRAIN MODEL
# ===============================================

history = model.fit(

    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]

)

print("\n✅ Training Completed")
print("Best model saved at:", MODEL_SAVE_PATH)

# ===============================================
# FINAL ACCURACY
# ===============================================

loss, acc = model.evaluate(validation_generator)

print("\nValidation Accuracy:", round(acc*100,2),"%")