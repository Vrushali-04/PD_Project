import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from torchvision import transforms
from PIL import Image

# ==========================================
# 1. CONFIGURATION & MODELS
# ==========================================
DEVICE = "cpu" 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

bouncer = DrawingBouncer().to(DEVICE)
bouncer_path = os.path.join(BASE_DIR, "drawing_gatekeeper.pth")
if os.path.exists(bouncer_path):
    bouncer.load_state_dict(torch.load(bouncer_path, map_location=DEVICE))
bouncer.eval()

model_pd = tf.keras.models.load_model(os.path.join(BASE_DIR, "best_spiral_cnn.h5"))

# ==========================================
# 2. PREDICTION EXECUTOR
# ==========================================
def predict_spiral(image_path):
    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return {"error": "File Error", "message": "Could not read image."}
            
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # --- PHASE 1: HYBRID SECURITY CHECK ---
        
        # A. AI TEXTURE CHECK
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        gate_transform = transforms.Compose([
            transforms.Resize((128, 128)), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])
        img_gate = gate_transform(img_pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            drawing_probability = bouncer(img_gate).item()

        # B. GEOMETRIC CHECK (Improved for shadows)
        # Apply blur to remove camera noise/shadow grain
        blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        is_digital_document = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > (img_gray.size * 0.10): # 10% of image
                # Check Solidity: (Area / Convex Hull Area)
                # Digital logos/text blocks are very solid (>0.5)
                # Hand-drawn spirals are "airy" and hollow (<0.3)
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area)/hull_area if hull_area > 0 else 0
                
                if solidity > 0.45: # High solidity = Digital Graphic/Text Block
                    is_digital_document = True
                    break

        print(f"[SECURITY] AI Prob: {drawing_probability:.4f}, Solid Block: {is_digital_document}")

        # GATEKEEPER LOGIC
        # Priority 1: High confidence AI drawing detection bypasses block check
        if drawing_probability > 0.85:
            pass 
        # Priority 2: Rejection
        elif drawing_probability < 0.40 or is_digital_document:
            return {
                "error": "Invalid Image",
                "message": "Security Reject: This doesn't look like a clear hand-drawn spiral. Please use plain white paper and clear lighting.",
                "confidence": 0,
                "prediction": "N/A"
            }

        # --- PHASE 2: MEDICAL ANALYSIS ---
        img_cv = cv2.resize(img_gray, (128, 128))
        img_pd = img_cv.astype("float32") / 255.0
        img_pd = np.expand_dims(np.expand_dims(img_pd, axis=-1), axis=0)

        prediction = model_pd.predict(img_pd, verbose=0)[0][0]
        
        result = "parkinson" if prediction >= 0.5 else "healthy"
        confidence = prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100

        return {
            "prediction": result,
            "confidence": round(float(confidence), 2),
            "message": "Analysis Successful: Drawing verified."
        }

    except Exception as e:
        print(f"[ERROR] Engine Failure: {e}")
        return {"error": "System Error", "message": str(e)}