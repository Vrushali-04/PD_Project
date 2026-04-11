import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. SETTINGS & PATHS
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "models/drawing_gatekeeper.pth"

# Positive: Your real handwritten spirals
DRAWING_DIR = "datasets/handwriting/combined/training"

# Negative: The folders from your screenshot (will search all subfolders)
NON_DRAWING_DIR = "datasets/scan_type/train/non_brain"

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==========================================
# 2. DATASET LOADER
# ==========================================
class DrawingGatekeeperDataset(Dataset):
    def __init__(self, draw_dir, non_draw_dir, transform=None):
        self.samples = []
        self.transform = transform

        # Label 1: Drawings
        for root, _, files in os.walk(draw_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(root, f), 1))

        # Label 0: Non-Drawings (Airplanes, Butterflies, etc.)
        for root, _, files in os.walk(non_draw_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(root, f), 0))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# ==========================================
# 3. THE BOUNCER MODEL
# ==========================================
class DrawingBouncer(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.main(x)

# ==========================================
# 4. TRAINING
# ==========================================
dataset = DrawingGatekeeperDataset(DRAWING_DIR, NON_DRAWING_DIR, transform)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = DrawingBouncer().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

print(f"[INFO] Training Gatekeeper on {len(dataset)} images...")

for epoch in range(5): # 5 epochs is enough for this simple task
    model.train()
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} Complete.")

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"[SUCCESS] Drawing Gatekeeper saved to {MODEL_SAVE_PATH}")