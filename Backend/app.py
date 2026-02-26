from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import bcrypt
from db import db
from models.predict_image import predict_image
from models.predict_voice import predict_voice

# ==============================
# CONFIG
# ==============================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ==============================
# ROUTES
# ==============================

# Home / Health check
@app.route("/")
def home():
    return jsonify({"status": "PD Backend Running ✅"})


# Serve favicon to avoid 404
@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon"
    )

# ==============================
# 🔐 AUTH ROUTES
# ==============================

@app.route("/signup", methods=["POST"])
def signup():
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"message": "No JSON data received"}), 400

        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if not name or not email or not password:
            return jsonify({"message": "All fields are required"}), 400

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

        cursor = db.cursor()
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, hashed_pw.decode())
        )
        db.commit()

        return jsonify({"message": "Signup successful"}), 201

    except Exception as e:
        print("Signup Error:", e)
        return jsonify({"message": "Server error during signup"}), 500


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.json
        email = data["email"]
        password = data["password"]

        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if user and bcrypt.checkpw(password.encode(), user["password"].encode()):
            return jsonify({"message": "Login successful", "name": user["name"]})
        else:
            return jsonify({"message": "Invalid email or password"}), 401

    except Exception as e:
        print("Login Error:", e)
        return jsonify({"message": "Server error during login"}), 500


# ==============================
# 🖼️ IMAGE PREDICTION
# ==============================
@app.route("/predict_image", methods=["POST"])
def predict_image_api():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        result = predict_image(file_path)

        os.remove(file_path)

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        print("Image Prediction Error:", e)
        return jsonify({"error": "Failed to process image"}), 500


# ==============================
# 🎤 VOICE PREDICTION
# ==============================

@app.route("/predict_voice", methods=["POST"])
def predict_voice_route():
    try:
        print("🔥 Voice API Called 🔥")

        data = request.get_json()
        print("Received Data:", data)

        features = [
            float(data["MDVP_Fo_Hz"]),
            float(data["MDVP_Jitter_percent"]),
            float(data["MDVP_Shimmer"]),
            float(data["HNR"]),
            float(data["RPDE"]),
            float(data["DFA"]),
            float(data["Spread1"]),
            float(data["Spread2"]),
            float(data["PPE"])
        ]

        print("Processed Features:", features)

        result = predict_voice(features)
        print("Prediction Result:", result)

        return jsonify(result)

    except Exception as e:
        print("Voice Prediction Error:", e)
        return jsonify({"error": str(e)}), 400

# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)