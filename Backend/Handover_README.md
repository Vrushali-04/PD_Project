# 🛡️ Project Handover: The "Dual-Sentinel" Security System

This document explains the advanced security layers added to the backend to prevent "AI Hallucinations" and ensure the system only diagnoses real medical data.

---

### 🧠 1. The Four-Model Architecture
The /models folder now contains four distinct AI "Brains." We use a Guard & Doctor strategy for both MRI and Spiral analysis to ensure clinical integrity.

| Component | File Name | Framework | Role |
| :--- | :--- | :--- | :--- |
| 🔍 MRI Guard | brain_classifier.pth | PyTorch | Sentinel: Rejects any image that isn't a Brain MRI. |
| 🩺 MRI Doctor | best_mri_cnn.h5 | TensorFlow | Diagnostic: Predicts Parkinson's from verified scans. |
| 🔍 Drawing Guard | drawing_gatekeeper.pth | PyTorch | Sentinel: Rejects digital text, logos, or screenshots. |
| 🩺 Drawing Doctor | best_spiral_cnn.h5 | TensorFlow | Diagnostic: Predicts Parkinson's from verified drawings. |

---

### 🛠️ 2. The Hybrid Drawing Validation (predict_spiral.py)
To catch "sneaky" digital documents (like certificates with curved logos), we implemented a Hybrid Validation Pipeline:

* 🎨 Layer A: Neural Texture Analysis (AI)
    * Uses a PyTorch CNN trained on 12,505 images.
    * It recognizes the "texture" of pen-on-paper vs. computer-generated pixels.
* 📐 Layer B: Geometric Blob Filtering (Math)
    * Uses cv2.adaptiveThreshold to handle shadows and uneven lighting from phone cameras.
    * The Logic: Hand-drawn spirals are thin, hollow lines. Digital logos and heavy text (like "CERTIFICATE") appear as Solid Blobs.
    * The Rule: If any solid object takes up more than 15% of the image area, the system flags it as a "Digital Document" and rejects it.

---

### 💻 3. Frontend Integration Requirements
The Backend now sends a standardized JSON object for security rejections. The React frontend must be configured to "listen" for these specific keys:

* error: If this exists (e.g., "Invalid Image"), the Sentinel has blocked the upload.
* message: This contains the specific explanation for the user (e.g., "Security Reject: Digital document detected").

How to handle it in UI:
1. PredictionForm.tsx: Must capture the message from the backend and pass it to the explanation prop.
2. PredictionResult.tsx: Must accept result="error" to trigger the Yellow Warning Shield instead of a Red/Green result.

---

### 🎤 4. Presentation Pro-Tips (For the Demo)
When demonstrating to the judges, use these "Defense Points":

* "Input Validation Layer": We don't just trust the user; we verify the data is clinically relevant.
* "Adversarial Robustness": Our system distinguishes between a robot-drawn logo and a human-drawn spiral, even with poor lighting or shadows.
* "Zero-Hallucination Policy": By refusing to diagnose "garbage data," we ensure our Parkinson's models are only used for their intended purpose.

---

### ✅ Developer Checklist
* [ ] Define DEVICE = "cpu" at the top level of predict_spiral.py.
* [ ] Ensure PredictionResult.tsx interface includes result: "healthy" | "detected" | "error".
* [ ] Restart the Engine: Run python app.py to ensure the new Hybrid logic is active in memory.