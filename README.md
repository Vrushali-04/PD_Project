🧠 Parkinson’s Disease Prediction System

An AI-powered multi-modal diagnostic platform designed to assist in the early detection of Parkinson’s Disease using biomedical data such as voice signals, brain MRI images, and hand-drawn spiral patterns.

--------------------------------------------------------------------------------------------------------------------------------

📌 Overview

Parkinson’s Disease is a progressive neurological disorder that affects movement, speech, and coordination. Early diagnosis is challenging because symptoms appear gradually.

This system improves detection using multi-modal AI analysis of:

🎤 Voice Signals

🧠 Brain MRI Images

✍️ Spiral Drawing Patterns

By combining multiple data sources, the model generates more reliable predictions than single-input systems.

⚙️ Core Architecture

🎤 Voice Analysis

MFCC feature extraction

Jitter & shimmer analysis

Signal preprocessing

SVM classification

🧠 Brain MRI Analysis

Image preprocessing (OpenCV)

Feature extraction using CNN

Neurological pattern detection

✍️ Spiral Pattern Analysis

Drawing preprocessing

Image feature extraction

CNN–LSTM tremor pattern detection

--------------------------------------------------------------------------------------------------------------------------------

🌐 Web Application

🎨 Frontend: Built using React.js

Features: Voice upload • MRI image upload • Spiral drawing upload • Instant prediction results • Responsive UI

⚙️ Backend: Built using Flask (Python)

Responsibilities: API handling • Data preprocessing • ML model loading • Prediction generation

--------------------------------------------------------------------------------------------------------------------------------

🧠 AI Models

| Prediction Input Type | Machine Learning / Deep Learning Model  |
|-----------------------|-----------------------------------------|
| Voice Data            | Support Vector Machine (SVM)            |
| Brain MRI Images      | Convolutional Neural Network (CNN)      |
| Spiral Drawings       | CNN–LSTM Hybrid Model                   |

--------------------------------------------------------------------------------------------------------------------------------

🧪 Model Training & Evaluation

Libraries: TensorFlow • Keras • Scikit-learn • OpenCV • NumPy • Pandas

Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, Cross-Validation.

--------------------------------------------------------------------------------------------------------------------------------

🛠️ Technology Stack

Frontend : ⚛️ React.js 🟦 TypeScript 🎨 CSS ⚡ Vite 🌬️ Tailwind CSS

Backend : 🐍 Python 🌐 Flask 🔗 REST APIs

Machine Learning / AI : 🔥 TensorFlow 🧠 Keras 📊 Scikit-learn 📷 OpenCV 🔢 NumPy 🐼 Pandas

Deep Learning Models : 📡 MobileNetV2 (Transfer Learning CNN) 📈 Support Vector Machine (SVM) 🔗 CNN-LSTM Hybrid Model

--------------------------------------------------------------------------------------------------------------------------------
⚙️ Setup Instructions :

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

