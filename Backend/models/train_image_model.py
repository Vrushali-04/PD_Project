import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pickle

# =====================
# CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 10

# Brain image folders
BRAIN_TRAIN_DIRS = [
    "datasets/images/train/healthy",
    "datasets/images/train/parkinson"
]

# Non-brain folders (Caltech-101 etc.)
NON_BRAIN_TRAIN_DIR = "datasets/scan_type/train"

# =====================
# TRANSFORMS
# =====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
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
            for img in os.listdir(folder):
                if img.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(folder, img), 1))

        # Non-brain images (ALL subfolders) → label 0
        for root, _, files in os.walk(non_brain_dir):
            for img in files:
                if img.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(root, img), 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# =====================
# LOAD DATA
# =====================
dataset = BrainVsNonBrainDataset(
    BRAIN_TRAIN_DIRS,
    NON_BRAIN_TRAIN_DIR,
    transform
)

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
            nn.MaxPool2d(2),
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
# TRAINING
# =====================
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(loader):.4f}")

# =====================
# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/brain_classifier.pth")
print("✅ Brain classifier saved to models/brain_classifier.pth")
