🧠 Parkinson’s Disease Prediction System

A multi-modal AI-based medical diagnostic platform designed to improve early detection of Parkinson’s Disease by analyzing multiple biological signals including voice recordings, brain MRI images, and hand-drawn spiral patterns.

The system combines machine learning and deep learning models to extract features from different data sources and produce accurate diagnostic predictions through an intuitive web application interface.

📌 Project Overview

Parkinson’s Disease is a progressive neurological disorder that affects movement, speech, and coordination. Early diagnosis is challenging because symptoms develop gradually.

This project proposes a multi-modal prediction system that integrates:

🎤 Voice Signal Analysis

🧠 Brain Image Analysis

✍️ Hand-Drawn Pattern Analysis

Using deep learning based feature fusion, the system combines these heterogeneous data sources to generate more reliable predictions compared to single-modality systems.

⚙️ Core Technical Architecture

🔹 Multi-Modal Data Processing

🎤 1. Voice Data Analysis

Acoustic feature extraction using MFCC

Extraction of jitter and shimmer features

Signal preprocessing and normalization

Classification using Support Vector Machine (SVM)

🧠 2. Brain Image Analysis

Image preprocessing using OpenCV

Feature extraction with Convolutional Neural Networks (CNN)

Spatial pattern recognition for detecting neurological anomalies

✍️ 3. Hand-Drawn Pattern Analysis

Spiral and handwriting pattern enhancement

Feature extraction from drawing images

Sequential modeling using CNN-LSTM architecture

Detection of tremor-based patterns in drawings


🌐 Web Application Architecture

🔹 Frontend (Client Side): Built using React.js with modern UI tools.

Features : 

Upload voice recordings

Upload brain MRI images

Upload spiral drawings

Display prediction results instantly

Responsive and interactive UI

🔹 Backend (Server Side): Developed using Flask (Python).

Backend Responsibilities : 

Handling API requests

Data preprocessing

Loading trained ML models

Performing model inference

Returning prediction results

🧠 Machine Learning & Deep Learning Models

The system integrates multiple models specialized for different types of data.

Models Used

1. Support Vector Machine (SVM): Used for structured voice feature classification

2. Convolutional Neural Network (CNN): Used for spatial feature extraction from brain MRI images

3. CNN-LSTM Hybrid Model: Used for analyzing sequential drawing patterns

🧪 Model Training & Evaluation

The models are implemented using Python-based AI frameworks.

Libraries Used: TensorFlow, Keras, Scikit-learn, OpenCV, NumPy, Pandas

Evaluation Metrics

The models are evaluated using the following metrics:

1. Accuracy
  
2. Precision

3. Recall

4. F1-Score

5. Cross-Validation

These metrics ensure that the models are robust, reliable, and generalize well to unseen data.

🛠️ Technology Stack

Frontend : ⚛️ React.js 🟦 TypeScript 🎨 CSS ⚡ Vite 🌬️ Tailwind CSS

Backend : 🐍 Python 🌐 Flask 🔗 REST APIs

Machine Learning / AI : 🔥 TensorFlow 🧠 Keras 📊 Scikit-learn 📷 OpenCV 🔢 NumPy 🐼 Pandas

Deep Learning Models : 📡 MobileNetV2 (Transfer Learning CNN) 📈 Support Vector Machine (SVM) 🔗 CNN-LSTM Hybrid Model

📂 Project Structure

<img width="585" height="659" alt="Screenshot 2026-03-08 175850" src="https://github.com/user-attachments/assets/3893a193-170d-4561-a51b-cb9efe8f714d" />

Results And Scrinshots 

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

<img width="1810" height="825" alt="Screenshot 2026-03-08 182450" src="https://github.com/user-attachments/assets/a308f540-9241-46bc-8618-fef63a75ff28" />

