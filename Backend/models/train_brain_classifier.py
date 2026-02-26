import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# =====================
# CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10
MODEL_SAVE_PATH = "models/brain_classifier.pth"

# Brain image folders
BRAIN_TRAIN_DIRS = [
    "datasets/images/train/healthy",
    "datasets/images/train/parkinson"
]

# Non-brain folders
NON_BRAIN_TRAIN_DIR = "datasets/scan_type/train"

# =====================
# TRANSFORMS
# =====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# =====================
# CUSTOM DATASET
# =====================
class BrainVsNonBrainDataset(Dataset):
    def __init__(self, brain_dirs, non_brain_dir, transform=None):
        self.samples = []
        self.transform = transform

        # Brain images → label 1
        for folder in brain_dirs:
            if not os.path.exists(folder):
                raise FileNotFoundError(f"Folder not found: {folder}")
            for img in os.listdir(folder):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder, img), 1))

        # Non-brain images → label 0
        if not os.path.exists(non_brain_dir):
            raise FileNotFoundError(f"Folder not found: {non_brain_dir}")
        for root, _, files in os.walk(non_brain_dir):
            for img in files:
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(root, img), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# =====================
# LOAD DATA
# =====================
dataset = BrainVsNonBrainDataset(BRAIN_TRAIN_DIRS, NON_BRAIN_TRAIN_DIR, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"✅ Total training images: {len(dataset)}")

# =====================
# CNN MODEL
# =====================
class CNN(nn.Module):
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

model = CNN().to(DEVICE)

# =====================
# LOSS & OPTIMIZER
# =====================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =====================
# TRAINING LOOP
# =====================
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

    avg_loss = total_loss / len(loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# =====================
# SAVE MODEL
# =====================
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Brain classifier saved at {MODEL_SAVE_PATH}")
