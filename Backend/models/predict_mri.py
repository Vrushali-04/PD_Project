import os
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. THE AI BLUEPRINT (PyTorch)
# ==========================================
# WHY THIS CODE IS HERE: TensorFlow is easy—it saves both the "Brain" and 
# its "Memories" into one single file. PyTorch is different; it ONLY saves 
# the "Memories" (the math numbers). 
# Because of this, we have to build an empty "Skeleton" here in the code.
# Later, we will pour the saved memories into this skeleton so it can work!
class GatekeeperCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature Extractor: 3 Convolutional blocks to extract spatial hierarchies
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Classifier: Flattens the extracted features into a binary probability (Brain vs. Non-Brain)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid() # Outputs a strict 0.0 to 1.0 probability
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ==========================================
# 2. LOAD HYBRID MODELS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# A. Load Primary Diagnostic Model (TensorFlow / Keras)
MODEL_PATH = os.path.join(BASE_DIR, "best_mri_cnn.h5")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"[ERROR] TensorFlow model not found at {MODEL_PATH}")

mri_model = tf.keras.models.load_model(MODEL_PATH)
print("[INFO] Primary MRI CNN (TensorFlow) loaded successfully.")

# B. Load OOD Gatekeeper Model (PyTorch)
GATEKEEPER_PATH = os.path.join(BASE_DIR, "brain_classifier.pth")
device = torch.device('cpu') # Enforce CPU inference for web-server compatibility

try:
    gatekeeper = GatekeeperCNN()
    # map_location=device ensures it loads on CPU even if trained on a GPU (CUDA)
    gatekeeper.load_state_dict(torch.load(GATEKEEPER_PATH, map_location=device))
    gatekeeper.eval() # Set to evaluation mode (disables dropout/batchnorm for stable inference)
    print("[INFO] Brain Gatekeeper CNN (PyTorch) loaded successfully.")
except Exception as e:
    print(f"[ERROR] Gatekeeper failed to load: {e}")
    gatekeeper = None

IMG_SIZE = 128

# ==========================================
# 3. OOD (OUT-OF-DISTRIBUTION) GATEKEEPER
# ==========================================
def is_valid_brain(image_path):
    """
    Prevents the primary diagnostic model from making forced guesses on non-MRI data.
    """
    if gatekeeper is None:
        return True # Fail-open: If gatekeeper fails to load, allow the system to run

    try:
        # Transforms MUST perfectly match the pipeline used during train_brain_classifier.py
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(), # Converts to PyTorch Tensor and scales to [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Shifts to [-1, 1] range
        ])

        # We read as RGB because colorful non-medical photos are easier to reject in 3 channels
        img = Image.open(image_path).convert('RGB')
        
        # Add batch dimension: [Channels, Height, Width] -> [Batch, Channels, Height, Width]
        img_tensor = transform(img).unsqueeze(0).to(device)

        # torch.no_grad() disables gradient tracking, saving memory and speeding up inference
        with torch.no_grad():
            output = gatekeeper(img_tensor)
            prediction = output.item() 
            
            print(f"[METRICS] Gatekeeper Brain Probability: {prediction:.4f}")

            # Strict Thresholding: 1 = Brain, 0 = Non-Brain
            if prediction < 0.5: 
                print("[REJECTED] Image flagged as Non-Brain by Gatekeeper.")
                return False 
            
            print("[ACCEPTED] Image verified as an Axial Brain MRI.")
            return True 

    except Exception as e:
        print(f"[ERROR] Gatekeeper inference error: {e}")
        return True

# ==========================================
# 4. PREPROCESSING PIPELINE (TENSORFLOW)
# ==========================================
def preprocess_mri(image_path):
    """
    Prepares the verified MRI scan for the primary TensorFlow diagnostic model.
    """
    if not os.path.exists(image_path):
        raise ValueError(f"Image path does not exist: {image_path}")

    # Read as Grayscale because true MRI structural data is single-channel
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image file format.")

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0 # Normalize pixel intensities to [0, 1]
    
    # Keras expects shape [Batch, Height, Width, Channels]
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img

# ==========================================
# 5. PREDICTION EXECUTOR
# ==========================================
def predict_mri_scan(image_path):
    """
    Orchestrates the Dual-Framework inference pipeline.
    """
    try:
        # Phase 1: Security & Validation (PyTorch)
        if not is_valid_brain(image_path):
            return {"error": "Invalid Image: Please upload a valid Axial Brain MRI scan."}

        # Phase 2: Diagnostic Inference (TensorFlow)
        img = preprocess_mri(image_path)
        prediction = mri_model.predict(img, verbose=0)[0][0]
        
        print(f"[PREDICTION] Raw TensorFlow PD Score: {prediction:.4f}")

        # Phase 3: Response Formatting
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
        print(f"[ERROR] MRI Pipeline failed: {e}")
        return {"error": str(e)}