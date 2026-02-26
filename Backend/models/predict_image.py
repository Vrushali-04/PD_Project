import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle

# ================================
# DEVICE
# ================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# PATHS
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "cnn_image_model.pth")       # Parkinson model
LABELS_PATH = os.path.join(BASE_DIR, "image_labels.pkl")         # Parkinson labels
BRAIN_MODEL_PATH = os.path.join(BASE_DIR, "brain_classifier.pth") # Brain vs non-brain model

# ================================
# IMAGE TRANSFORMS
# ================================
IMG_SIZE = 128
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

brain_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ================================
# PARKINSON CNN MODEL
# ================================
class ParkinsonCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ================================
# BRAIN VS NON-BRAIN CNN MODEL
# ================================
class BrainCNN(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ================================
# LOAD MODELS
# ================================
# Parkinson model
parkinson_model = ParkinsonCNN().to(DEVICE)
parkinson_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
parkinson_model.eval()

# Labels
with open(LABELS_PATH, "rb") as f:
    CLASS_NAMES = pickle.load(f)

# Brain model
brain_model = BrainCNN().to(DEVICE)
brain_model.load_state_dict(torch.load(BRAIN_MODEL_PATH, map_location=DEVICE))
brain_model.eval()

# ================================
# FUNCTIONS
# ================================
def predict_brain_image(image_path: str) -> str:
    """Returns 'brain' or 'not-brain'"""
    image = Image.open(image_path).convert("RGB")
    image = brain_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = brain_model(image)
        pred = output.item()  # sigmoid output
    return "brain" if pred >= 0.5 else "not-brain"

def predict_image(image_path: str):
    """Main prediction function"""
    if not os.path.exists(image_path):
        return {"error": "Image file not found"}

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return {"error": "Invalid image file"}

    # Step 1: Brain check
    brain_result = predict_brain_image(image_path)
    if brain_result == "not-brain":
        return {"error": "This does not appear to be a valid brain image"}

    # Step 2: Parkinson prediction
    image = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = parkinson_model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    label = CLASS_NAMES[predicted.item()]
    return {"prediction": label, "confidence": round(confidence.item(), 4)}

# ================================
# RUN DIRECTLY
# ================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict_image(image_path)

    if "error" in result:
        print("Error:", result["error"])
    else:
        print("Prediction:", result["prediction"])
        print("Confidence:", result["confidence"])
