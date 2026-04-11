import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ==========================================
# 1. DIRECTORY CONFIGURATION & HYPERPARAMETERS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "datasets", "mri_slices")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_mri_cnn.h5")

# Standardized parameters for medical image processing
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 30 

# ==========================================
# 2. DATA PIPELINE: Training & Validation Splitting
# ==========================================
print("[INFO] Initializing MRI Dataset Pipeline...")

# Training Subset: Primary data used for weight optimization (80%)
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale" # MRI data is inherently single-channel
)

# Validation Subset: Unseen data used to monitor generalization (20%)
val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

# FEATURE SCALING: Normalizing pixel intensities to [0, 1] range for stable convergence
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))

# ==========================================
# 3. CNN ARCHITECTURE: Feature Extraction & Classification
# ==========================================
print("[INFO] Constructing Convolutional Neural Network...")

model = Sequential([
    # LAYER 1: Extracts low-level spatial features (edges and contours)
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D(2, 2), # Downsampling to reduce computational complexity
    
    # LAYER 2: Identifies complex textures and tissue densities
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # LAYER 3: Analyzes high-level structural patterns in brain slices
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    # CLASSIFICATION HEAD: Converts 2D feature maps into a 1D classification vector
    Flatten(),
    
    # DENSE LAYER: Deep reasoning and pattern correlation
    Dense(128, activation='relu'),
    
    # REGULARIZATION: Dropout prevents overfitting by deactivating 50% of neurons during training
    Dropout(0.5), 
    
    # OUTPUT: Binary classification (0: Healthy, 1: Parkinson's Disease)
    Dense(1, activation='sigmoid')
])

# Compilation using Binary Crossentropy loss for two-class optimization
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# 4. TRAINING CALLBACKS & EXECUTION
# ==========================================
# AUTO-SAVE: Persists only the highest-accuracy version of the weights
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# EARLY STOPPING: Terminates training if validation loss plateaus to preserve generalization
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("[INFO] Commencing training session...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print(f"[SUCCESS] High-accuracy model exported to: {MODEL_SAVE_PATH}")