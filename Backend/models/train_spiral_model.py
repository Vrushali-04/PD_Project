import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# ==========================================
# 1. DIRECTORY CONFIGURATION
# ==========================================
TRAIN_DIR = 'datasets/handwriting/combined/training'
TEST_DIR = 'datasets/handwriting/combined/testing'

IMG_SIZE = (128, 128) 
BATCH_SIZE = 32       
EPOCHS = 30           

# ==========================================
# 2. DATA AUGMENTATION
# ==========================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.05,     
    height_shift_range=0.05,    
    zoom_range=0.1,             
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

print("[INFO] Loading datasets...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode='grayscale', class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE, color_mode='grayscale', class_mode='binary', shuffle=False)

# ==========================================
# 3. CNN ARCHITECTURE
# ==========================================
model = Sequential([
    # Input Layer (Fixes the UserWarning)
    Input(shape=(128, 128, 1)),
    
    # BLOCK 1
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # BLOCK 2
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # BLOCK 3 (FIXED THE TYPO HERE: Conv2D instead of Conv2d)
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    # CLASSIFICATION HEAD
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# 4. TRAINING CALLBACKS
# ==========================================
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
checkpoint = ModelCheckpoint('models/best_spiral_cnn.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

# ==========================================
# 5. EXECUTION
# ==========================================
print("[INFO] Commencing Spiral/Wave CNN Training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=test_generator,
    callbacks=[early_stop, checkpoint, reduce_lr],
    verbose=1
)

print("[SUCCESS] Spiral model training complete.")