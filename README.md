🧠 Parkinson’s Disease Prediction System

An AI-powered multi-modal diagnostic platform designed to assist in the early detection of Parkinson’s Disease using biomedical data such as voice signals, brain MRI images, and hand-drawn spiral patterns.

📌 Overview

Parkinson’s Disease is a progressive neurological disorder that affects movement, speech, and coordination. Since symptoms develop gradually, early diagnosis can be difficult.

This system improves early detection by combining multiple analysis techniques:

🎤 Voice Signal Analysis

🧠 Brain MRI Image Analysis

✍️ Spiral Drawing Pattern Analysis

Using multi-modal AI models, the system produces more reliable predictions than single-data approaches.

⚙️ Core Architecture

🎤 Voice Analysis:

MFCC feature extraction

Jitter & shimmer analysis

Signal preprocessing

SVM-based classification

🧠 Brain Image Analysis:

Image preprocessing using OpenCV

Feature extraction with CNN

Detection of neurological patterns

✍️ Spiral Pattern Analysis:

Drawing preprocessing

Feature extraction from images

CNN–LSTM model for tremor pattern detection

🌐 Web Application

🎨 Frontend: Built using React.js

Features: Voice upload • MRI image upload • Spiral drawing upload • Instant prediction results • Responsive UI

⚙️ Backend: Built using Flask (Python)

Responsibilities: API handling • Data preprocessing • ML model loading • Prediction generation

🧠 AI Models

| Prediction Input Type | Machine Learning / Deep Learning Model  |
|-----------------------|-----------------------------------------|
| Voice Data            | Support Vector Machine (SVM)            |
| Brain MRI Images      | Convolutional Neural Network (CNN)      |
| Spiral Drawings       | CNN–LSTM Hybrid Model                   |


🧪 Model Training & Evaluation

Libraries: TensorFlow • Keras • Scikit-learn • OpenCV • NumPy • Pandas

Evaluation Metrics: Accuracy, Precision, Recall, F1 Score, Cross-Validation.

🛠️ Technology Stack

Frontend : ⚛️ React.js 🟦 TypeScript 🎨 CSS ⚡ Vite 🌬️ Tailwind CSS

Backend : 🐍 Python 🌐 Flask 🔗 REST APIs

Machine Learning / AI : 🔥 TensorFlow 🧠 Keras 📊 Scikit-learn 📷 OpenCV 🔢 NumPy 🐼 Pandas

Deep Learning Models : 📡 MobileNetV2 (Transfer Learning CNN) 📈 Support Vector Machine (SVM) 🔗 CNN-LSTM Hybrid Model

📂 Project Structure

<img width="585" height="659" alt="Screenshot 2026-03-08 175850" src="https://github.com/user-attachments/assets/3893a193-170d-4561-a51b-cb9efe8f714d" />

📊 Results & Screenshots: 


🔐 User Authentication Interface : This interface allows users to create an account or sign in to securely access the Parkinson’s disease prediction system.

<img width="1911" height="913" alt="Screenshot 2026-03-08 181544" src="https://github.com/user-attachments/assets/7b37c052-16b2-492d-811a-20a421bbb7f9" />




🏠 Landing Page : 
The landing page introduces the Parkinson’s Disease Prediction System, highlighting the role of AI and biomedical analysis in early disease detection. Users can start the prediction process by clicking the “Get Started / Try Prediction” button.

<img width="1899" height="909" alt="Screenshot 2026-03-08 181940" src="https://github.com/user-attachments/assets/97cd548a-6fe6-4899-880a-9b0d9f7c4978" />




🎤 Voice-Based Parkinson’s Prediction :
This interface allows users to enter biomedical voice measurement parameters to analyze vocal patterns associated with Parkinson’s disease. The system processes the input features and generates an AI-based prediction result with confidence level and explanation summary.

<img width="1896" height="857" alt="Screenshot 2026-03-08 182309" src="https://github.com/user-attachments/assets/f792368f-088f-49c3-9d84-805b06443f11" />




🧠 Brain MRI Image Prediction:
This module allows users to upload brain MRI images for AI-based analysis using a deep learning model. The system processes the image and provides a prediction result indicating the probability of Parkinson’s disease along with confidence and an AI explanation summary.

<img width="1810" height="825" alt="Screenshot 2026-03-08 182450" src="https://github.com/user-attachments/assets/a662f8d6-cab5-438f-b400-e4001cd11e3d" />




✍️ Spiral Drawing Pattern Prediction:
This module allows users to upload spiral drawing patterns for AI-based analysis of motor control irregularities associated with Parkinson’s disease. The system evaluates the drawing and provides a prediction result with confidence score and AI explanation summary.

<img width="1894" height="904" alt="Screenshot 2026-03-08 171413" src="https://github.com/user-attachments/assets/90130a1f-5f8a-4506-89ff-26e9f616fb64" />




🎯 About Our Mission: 
This section highlights the mission of the Parkinson’s Disease Prediction System, which is to use Artificial Intelligence to support early detection and improve patient outcomes. The platform aims to provide an AI-driven, medically inspired, and human-centered approach for accessible and reliable disease screening.

<img width="1858" height="742" alt="Screenshot 2026-03-08 185108" src="https://github.com/user-attachments/assets/e8510dbe-d27e-4d5a-833a-fdc3fa06b0da" />

