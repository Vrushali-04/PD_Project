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


<img width="1354" height="728" alt="Screenshot 2026-04-12 131919" src="https://github.com/user-attachments/assets/44e327bf-0b00-4667-947b-faaeaf912318" />
<img width="1392" height="697" alt="Screenshot 2026-04-12 132047" src="https://github.com/user-attachments/assets/54e737e9-a953-4872-b76a-9ac4917444a1" />

--------------------------------------------------------------------------------------------------------------------------------

📊 Results & Screenshots: 


🔐 User Authentication Interface : Users can sign up or log in to securely access the Parkinson’s disease prediction system.

<img width="1911" height="913" alt="Screenshot 2026-03-08 181544" src="https://github.com/user-attachments/assets/7b37c052-16b2-492d-811a-20a421bbb7f9" />



--------------------------------------------------------------------------------------------------------------------------------
🏠 Landing Page : The landing page introduces the system and allows users to start the AI-based prediction process.

<img width="1899" height="909" alt="Screenshot 2026-03-08 181940" src="https://github.com/user-attachments/assets/97cd548a-6fe6-4899-880a-9b0d9f7c4978" />

--------------------------------------------------------------------------------------------------------------------------------
🧠 Brain MRI Image Prediction: Users upload brain MRI images, which are analyzed by the AI model to detect possible Parkinson’s indicators.


<img width="1574" height="857" alt="Screenshot 2026-04-12 120530" src="https://github.com/user-attachments/assets/882dd1d1-81bd-4495-ad53-a84705237589" />


--------------------------------------------------------------------------------------------------------------------------------
✍️ Spiral Drawing Pattern Prediction: Users upload spiral drawings to analyze motor control patterns related to Parkinson’s disease. 

<img width="1461" height="850" alt="Screenshot 2026-04-12 120618" src="https://github.com/user-attachments/assets/fe105e49-9481-4246-b3df-28146ead699a" />



--------------------------------------------------------------------------------------------------------------------------------
🎯 About Our Mission: To use Artificial Intelligence for early detection of Parkinson’s disease and support accessible, reliable screening.

<img width="1858" height="742" alt="Screenshot 2026-03-08 185108" src="https://github.com/user-attachments/assets/e8510dbe-d27e-4d5a-833a-fdc3fa06b0da" />

