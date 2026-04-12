## 🧠 Parkinson’s Disease Prediction System

This platform is designed as a secure, AI-driven diagnostic aid for early detection of Parkinson's Disease.

Unlike traditional systems, it combines:

🧠 Structural neurological data (MRI)

✍️ Motor-control patterns (spiral drawings)

👉 Along with a custom-built “Sentinel Security Layer” to ensure only valid medical inputs are analyzed.

--------------------------------------------------------------------------------------------------------------------------------
## System Architecture: 

Dual-Sentinel Framework : To eliminate unreliable predictions and AI hallucinations, the system follows a:

### Layer 1: Sentinel Guard (Security Layer) : 

Built using PyTorch-based CNN models

Validates authenticity before diagnosis

Key Functions:

✅ MRI Sentinel → Confirms presence of real brain scan

✅ Spiral Sentinel → Detects real pen-on-paper texture vs digital images

🚫 Rejects invalid / fraudulent inputs

### Layer 2: Medical Doctor (Diagnosis Layer) : 

Activated only after validation

Uses TensorFlow deep learning models

Capabilities:

🧠 Parkinson’s classification

📊 High-precision prediction

🔍 Dataset-trained diagnostic models


--------------------------------------------------------------------------------------------------------------------------------

## 🌐 Web Application

🎨 Frontend: Built using React.js

Features: MRI image upload • Spiral drawing upload • Instant prediction results • Responsive UI

⚙️ Backend: Built using Flask (Python)

Responsibilities: API handling • Data preprocessing • ML model loading • Prediction generation

--------------------------------------------------------------------------------------------------------------------------------
## 🧠 Deep Learning Model Architecture

| Modality      | Phase     | Architecture         | Framework     | Role       |
| ------------- | --------- | -------------------- | ------------- | ---------- |
| Brain MRI     | Security  | Custom CNN (Texture) | PyTorch       | Validator  |
| Brain MRI     | Diagnosis | MobileNetV2          | TensorFlow    | Classifier |
| Spiral Sketch | Security  | CNN + OpenCV Hybrid  | PyTorch / CV2 | Anti-Fraud |
| Spiral Sketch | Diagnosis | CNN–LSTM Hybrid      | TensorFlow    | Classifier |


--------------------------------------------------------------------------------------------------------------------------------

## 🧪 Model Training & Evaluation

Libraries: TensorFlow • Keras • Scikit-learn • OpenCV • NumPy • Pandas

Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, Cross-Validation.

--------------------------------------------------------------------------------------------------------------------------------

## 🛠️ Technology Stack

Frontend : ⚛️ React.js 🟦 TypeScript 🎨 CSS ⚡ Vite 🌬️ Tailwind CSS

Backend : 🐍 Python 🌐 Flask 🔗 REST APIs

Machine Learning / AI : 🔥 TensorFlow 🧠 Keras 📊 Scikit-learn 📷 OpenCV 🔢 NumPy 🐼 Pandas

Deep Learning Models : 📡 MobileNetV2 (Transfer Learning CNN) 🔗 CNN-LSTM Hybrid Model

Database : 🐬 MySQL

--------------------------------------------------------------------------------------------------------------------------------
## ⚙️ Setup Instructions :

1️⃣ Database Setup (MySQL)

1. Install MySQL and MySQL Workbench.
2. Create a database named **parkinson_db**.
3. Update database credentials in `backend/db.py`.
--------------------------------------------------------------------------------------------------------------------------------
2️⃣ Backend Setup (Flask)

<img width="963" height="353" alt="Screenshot 2026-03-10 182300" src="https://github.com/user-attachments/assets/f5154068-b64d-49c8-942e-f72c7a2662c5" />

--------------------------------------------------------------------------------------------------------------------------------
3️⃣ Frontend Setup (React)

<img width="958" height="215" alt="Screenshot 2026-03-10 180741" src="https://github.com/user-attachments/assets/e5211443-ce35-455a-9630-09249aa79f73" />

--------------------------------------------------------------------------------------------------------------------------------
📂 Project Structure

Parkinson-Disease-Prediction/
│
├── PD_Detector/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   │   ├── ui/
│   │   │   ├── AboutSection.tsx
│   │   │   ├── ContactSection.tsx
│   │   │   ├── HeroSection.tsx
│   │   │   ├── ImageUpload.tsx
│   │   │   ├── Navbar.tsx
│   │   │   ├── PredictionForm.tsx
│   │   │   ├── PredictionResult.tsx
│   │   │   ├── SpiralUpload.tsx
│   │   │   └── TeamSection.tsx
│   │   ├── hooks/
│   │   ├── lib/
│   │   ├── pages/
│   │   ├── App.css
│   │   ├── App.tsx
│   │   ├── index.css
│   │   ├── main.tsx
│   │   │   └── vite-env.d.ts
│   ├── index.html
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── vite.config.ts
│   └── tsconfig.json
│
├── Backend/
│   ├── datasets/
│   │   ├── handwriting/
│   │   │   └── combined/
│   │   │       ├── testing/
│   │   │       └── training/
│   │   │           ├── healthy/
│   │   │           └── parkinson/
│   │   ├── mri_slices/
│   │   └── scan_type/
│   ├── models/
│   │   ├── best_mri_cnn.h5
│   │   ├── best_spiral_cnn.h5
│   │   ├── brain_classifier.pth
│   │   ├── drawing_gatekeeper.pth
│   │   ├── extract_mri.py
│   │   ├── predict_mri.py
│   │   ├── predict_spiral.py
│   │   ├── train_brain_classifier.py
│   │   ├── train_drawing_gatekeeper.py
│   │   ├── train_mri_model.py
│   │   └── train_spiral_model.py
│   ├── uploads/
│   ├── app.py
│   ├── create_non_brain.py
│   └── db.py
│
└── README.md


<img width="585" height="659" alt="Screenshot 2026-03-08 175850" src="https://github.com/user-attachments/assets/3893a193-170d-4561-a51b-cb9efe8f714d" />

--------------------------------------------------------------------------------------------------------------------------------

📊 Results & Screenshots: 


🔐 User Authentication Interface : Users can sign up or log in to securely access the Parkinson’s disease prediction system.

<img width="1911" height="913" alt="Screenshot 2026-03-08 181544" src="https://github.com/user-attachments/assets/7b37c052-16b2-492d-811a-20a421bbb7f9" />



--------------------------------------------------------------------------------------------------------------------------------
🏠 Landing Page : The landing page introduces the system and allows users to start the AI-based prediction process.

<img width="1899" height="909" alt="Screenshot 2026-03-08 181940" src="https://github.com/user-attachments/assets/97cd548a-6fe6-4899-880a-9b0d9f7c4978" />



--------------------------------------------------------------------------------------------------------------------------------
🎤 Voice-Based Parkinson’s Prediction : Users enter voice measurement parameters to analyze vocal patterns and generate an AI prediction result.

<img width="1896" height="857" alt="Screenshot 2026-03-08 182309" src="https://github.com/user-attachments/assets/f792368f-088f-49c3-9d84-805b06443f11" />



--------------------------------------------------------------------------------------------------------------------------------
🧠 Brain MRI Image Prediction: Users upload brain MRI images, which are analyzed by the AI model to detect possible Parkinson’s indicators.

<img width="1810" height="825" alt="Screenshot 2026-03-08 182450" src="https://github.com/user-attachments/assets/a662f8d6-cab5-438f-b400-e4001cd11e3d" />



--------------------------------------------------------------------------------------------------------------------------------
✍️ Spiral Drawing Pattern Prediction: Users upload spiral drawings to analyze motor control patterns related to Parkinson’s disease. 

<img width="1894" height="904" alt="Screenshot 2026-03-08 171413" src="https://github.com/user-attachments/assets/90130a1f-5f8a-4506-89ff-26e9f616fb64" />



--------------------------------------------------------------------------------------------------------------------------------
🎯 About Our Mission: To use Artificial Intelligence for early detection of Parkinson’s disease and support accessible, reliable screening.

<img width="1858" height="742" alt="Screenshot 2026-03-08 185108" src="https://github.com/user-attachments/assets/e8510dbe-d27e-4d5a-833a-fdc3fa06b0da" />

