import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. SETTINGS & HARDWARE
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10

# Fixed path: If we run from Backend, we save into models/ folder
MODEL_SAVE_PATH = "models/brain_classifier.pth"

# FIXED PATHS: Removed "../" so it looks inside the current Backend directory
BRAIN_TRAIN_DIRS = [
    "datasets/mri_slices/healthy",
    "datasets/mri_slices/parkinson"
]

NON_BRAIN_TRAIN_DIR = "datasets/scan_type/train"

# Standardizing image format for the CNN
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ==========================================
# 2. DATASET LOADER (The Organizer)
# ==========================================
class BrainVsNonBrainDataset(Dataset):
    def __init__(self, brain_dirs, non_brain_dir, transform=None):
        self.samples = []
        self.transform = transform

        # Label 1: Real MRI Brain Slices
        for folder in brain_dirs:
            if os.path.exists(folder):
                for img in os.listdir(folder):
                    if img.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(folder, img), 1))
            else:
                print(f"[WARNING] Brain folder not found: {folder}")

        # Label 0: Miscellaneous non-medical images
        if os.path.exists(non_brain_dir):
            for root, _, files in os.walk(non_brain_dir):
                for img in files:
                    if img.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append((os.path.join(root, img), 0))
        else:
            print(f"[WARNING] Non-brain folder not found: {non_brain_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# ==========================================
# 3. CNN ARCHITECTURE (The Gatekeeper)
# ==========================================
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# ==========================================
# 4. TRAINING EXECUTION
# ==========================================
# This will now correctly find the images when run from the Backend folder
dataset = BrainVsNonBrainDataset(BRAIN_TRAIN_DIRS, NON_BRAIN_TRAIN_DIR, transform)

if len(dataset) == 0:
    print("[ERROR] No images found! Check your folder paths in the script.")
else:
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"[INFO] Initializing training with {len(dataset)} images.")

    model = CNN().to(DEVICE)
    criterion = nn.BCELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("[INFO] Commencing Gatekeeper weight optimization...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  -> Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(loader):.4f}")

    # Ensure the models directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[SUCCESS] Brain classifier weights exported to {MODEL_SAVE_PATH}")