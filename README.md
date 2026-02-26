🧠 Parkinson’s Disease Prediction System
📌 Project Overview

The Parkinson’s Disease Prediction System is a multi-modal medical diagnostic platform designed to improve early detection accuracy by analyzing voice signals, brain imaging data, and hand-drawn patterns. The system uses deep learning–based feature fusion to combine heterogeneous data sources and generate robust predictions.

⚙️ Core Technical Architecture
🔹 Multi-Modal Data Processing

1.Voice Data Analysis
Acoustic feature extraction (MFCC, jitter, shimmer)
Signal preprocessing and normalization
Classification using Support Vector Machines (SVM)

2.Brain Image Analysis
Image preprocessing using OpenCV
Feature learning via Convolutional Neural Networks (CNN)
Spatial pattern recognition for neurological anomalies

3.Hand-Drawn Pattern Analysis
Spiral and drawing image enhancement
Temporal feature extraction
Sequential modeling using CNN-LSTM architecture

🧠 Machine Learning & Deep Learning Models

SVM for structured voice feature classification
CNN for spatial feature extraction from brain images
CNN-LSTM for combined spatial-temporal analysis of drawing inputs
Model fusion to aggregate predictions from all modalities

Training, validation, and testing using optimized hyperparameters

🧪 Model Training & Evaluation

Implemented in Python
Frameworks: TensorFlow / Keras

Evaluation metrics:
Accuracy
Precision
Recall
F1-Score
Cross-validation for improved generalization

🌐 Web Application Layer
🔹 Frontend (Client Side)
Built using React.js
Real-time data visualization and prediction results
Responsive UI with component-based architecture
User input handling for audio, image, and drawing uploads

🔹 Backend (Server Side)
Developed using Flask
RESTful APIs for:
Data preprocessing
Model inference
Result delivery
Efficient request handling and model integration

🛠️ Technology Stack
🔹 Frontend
React.js
🔹 Backend
Flask (Python)
REST APIs
🔹 Machine Learning / AI
Python
TensorFlow
Keras
Scikit-learn
OpenCV
