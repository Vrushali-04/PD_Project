from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pickle

app = Flask(__name__)
CORS(app)

# ---------------------- VOICE MODEL SETUP ----------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models")

# Load voice model & scaler
voice_model = joblib.load(os.path.join(MODEL_PATH, "svm_voice_model.pkl"))
voice_scaler = joblib.load(os.path.join(MODEL_PATH, "voice_scaler.pkl"))

# ---------------------- IMAGE MODEL SETUP ----------------------
# CNN Model Definition (must match training script)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 30 * 30, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Load model and label classes
image_model_path = os.path.join(MODEL_PATH, "cnn_image_model.pth")
label_path = os.path.join(MODEL_PATH, "image_labels.pkl")

image_model = CNNModel()
state_dict = torch.load(image_model_path, map_location=torch.device('cpu'))
image_model.load_state_dict(state_dict)
image_model.eval()

with open(label_path, "rb") as f:
    image_classes = pickle.load(f)

# Define preprocessing (must match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ---------------------- ROUTES ----------------------
@app.route('/')
def home():
    return "✅ Parkinson's Detection API (Voice + Image) is running!"

# ---------- VOICE PREDICTION ----------
@app.route('/predict_voice', methods=['POST'])
def predict_voice():
    try:
        data = request.get_json()

        # Extract features
        features = [
            data["MDVP_Fo_Hz"],
            data["MDVP_Jitter_percent"],
            data["MDVP_Shimmer"],
            data["HNR"],
            data["RPDE"],
            data["DFA"],
            data["Spread1"],
            data["Spread2"],
            data["PPE"]
        ]

        input_data = np.array(features).reshape(1, -1)
        scaled_data = voice_scaler.transform(input_data)
        prediction = voice_model.predict(scaled_data)[0]
        result = "Parkinson's Detected" if prediction == 1 else "Healthy"

        return jsonify({"type": "voice", "prediction": int(prediction), "result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

# ---------- IMAGE PREDICTION ----------
@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        temp_path = os.path.join("temp_" + file.filename)
        file.save(temp_path)

        image = Image.open(temp_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = image_model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            prediction = image_classes[predicted.item()]

        os.remove(temp_path)

        return jsonify({"type": "image", "prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------------- MAIN ENTRY ----------------------
if __name__ == '__main__':
    app.run(debug=True)
