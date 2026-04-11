# 🧠 Parkinson’s Disease Prediction System (Sentinel Edition)

An AI-powered multi-modal diagnostic platform designed to assist in the early detection of Parkinson’s Disease using Brain MRI images and hand-drawn spiral patterns. This version features a **Dual-Sentinel Security Framework** to ensure data integrity and prevent non-medical or fraudulent uploads.

---

📌 ## Overview

Parkinson’s Disease is a progressive neurological disorder affecting movement and coordination. Early diagnosis is critical but often challenging. 

This system improves detection accuracy by using a "Guard & Doctor" architecture:
* **🧠 Brain MRI Analysis:** Validates brain scans before neurological pattern detection.
* **✍️ Spiral Drawing Analysis:** Uses a Hybrid Gatekeeper to distinguish between hand-drawn sketches and digital documents/logos.

---

⚙️ ## Core Architecture: The Dual-Sentinel System

To ensure clinical reliability, each diagnostic module is split into two layers:

### 1. 🧠 Brain MRI Pipeline
* **Sentinel Guard (PyTorch):** A texture-analysis model that validates if the upload is a genuine Brain MRI. It rejects X-rays, standard photos, or noise.
* **Medical Doctor (TensorFlow):** A CNN classifier that detects neurological indicators of Parkinson's in confirmed MRI scans.

### 2. ✍️ Spiral Pattern Pipeline
* **Hybrid Gatekeeper (OpenCV + PyTorch):** * **Neural Layer:** Detects "Paper-and-Ink" textures vs. digital pixels.
    * **Geometric Layer:** Uses blob-filtering to reject solid digital objects (like curved logos or certificates).
* **Medical Doctor (TensorFlow):** A CNN-LSTM model that analyzes tremor patterns in valid hand-drawn spirals.

---

🌐 ## Web Application

* **🎨 Frontend:** Built using **React.js** with **TypeScript**.
    * Features: MRI upload • Spiral upload • Hybrid security feedback • Responsive UI.
* **⚙️ Backend:** Built using **Flask (Python)**.
    * Responsibilities: Global `DEVICE` management (CPU-optimized), real-time security filtering, and ML prediction.

---

🧠 ## AI Models & Technology Stack

| Prediction Input Type | Model Architecture | Security Layer |
| :--- | :--- | :--- |
| **Brain MRI Images** | CNN (MobileNetV2) | Sentinel Texture Guard |
| **Spiral Drawings** | CNN–LSTM Hybrid | Hybrid Geometric Gatekeeper |

### 🛠️ Tech Stack:
* **AI/ML:** TensorFlow, PyTorch, Keras, OpenCV.
* **Data:** NumPy, Pandas, Scikit-learn.
* **Frontend:** Vite, Tailwind CSS, Lucide React.

---

⚙️ ## Setup Instructions

### 1️⃣ Database Setup (MySQL)
1.  Install MySQL and create a database named `parkinson_db`.
2.  Update credentials in `Backend/db.py`.

### 2️⃣ Backend Setup
1.  Navigate to the `Backend` folder.
2.  Create a virtual environment: `python -m venv venv`.
3.  Activate and install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the server: `python app.py`.

### 3️⃣ Frontend Setup
1.  Navigate to the `PD_Detector` folder.
2.  Install packages: `npm install`.
3.  Start the app: `npm run dev`.

---

📂 ## Project Structure

* `Backend/`: Flask API, MRI/Spiral model weights, and security logic.
* `PD_Detector/`: React frontend source code and UI components.
* `models/`: Pre-trained weights for both "Guard" and "Doctor" models.

---

🎯 ## Our Mission
To leverage Artificial Intelligence to provide accessible, reliable, and secure early-detection screening for Parkinson's Disease.