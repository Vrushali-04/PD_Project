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
# HOME ROUTE
# ==============================
@app.route("/")
def home():
    return jsonify({"status": "PD Backend Running ✅"})


# Avoid favicon 404 error
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
        data = request.get_json()

        if not data:
            return jsonify({"message": "No JSON data received"}), 400

        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if not name or not email or not password:
            return jsonify({"message": "All fields are required"}), 400

        cursor = db.cursor(dictionary=True)

        # Check if email already exists
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        existing_user = cursor.fetchone()
        if existing_user:
            return jsonify({"message": "Email already registered"}), 400

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, hashed_pw)
        )
        db.commit()

        print("✅ New user registered:", email)

        return jsonify({"message": "Signup successful"}), 201

    except Exception as e:
        print("❌ Signup Error:", e)
        return jsonify({"message": "Server error during signup"}), 500


@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"message": "No JSON data received"}), 400

        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"message": "Email and password required"}), 400

        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        if user and bcrypt.checkpw(password.encode(), user["password"].encode()):
            print("✅ Login successful:", email)
            return jsonify({
                "message": "Login successful",
                "name": user["name"]
            })
        else:
            return jsonify({"message": "Invalid email or password"}), 401

    except Exception as e:
        print("❌ Login Error:", e)
        return jsonify({"message": "Server error during login"}), 500

@app.route("/users", methods=["GET"])
def get_users():
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id, name, email FROM users")
        users = cursor.fetchall()

        return jsonify(users)

    except Exception as e:
        print("Error:", e)
        return jsonify({"message": "Error fetching users"}), 500
    
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

        print("📷 Image received:", file.filename)

        result = predict_image(file_path)

        os.remove(file_path)

        if not isinstance(result, dict):
            return jsonify({"error": "Invalid prediction result"}), 500

        print("✅ Image Prediction:", result)

        return jsonify(result)

    except Exception as e:
        print("❌ Image Prediction Error:", e)
        return jsonify({"error": "Failed to process image"}), 500


# ==============================
# 🎤 VOICE PREDICTION
# ==============================

@app.route("/predict_voice", methods=["POST"])
def predict_voice_route():
    try:
        print("\n🔥 Voice API Called 🔥")

        data = request.get_json()

        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # ✅ Feature order MUST match training order exactly
        feature_order = [
            "mdvpFo",        # MDVP:Fo(Hz)
            "mdvpJitter",    # MDVP:Jitter(%)
            "mdvpShimmer",   # MDVP:Shimmer
            "hnr",           # HNR
            "rpde",          # RPDE
            "dfa",           # DFA
            "spread1",       # spread1
            "spread2",       # spread2
            "ppe"            # PPE
        ]

        features = []

        for field in feature_order:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

            try:
                value = float(data[field])
                features.append(value)
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid numeric value for {field}"}), 400

        #print("📊 Final Ordered Features:", features)

        # Call model prediction
        result = predict_voice(features)

        if "error" in result:
            return jsonify(result), 400

        print("✅ Prediction Result:", result)

        return jsonify(result), 200

    except Exception as e:
        print("❌ Voice Prediction Error:", e)
        return jsonify({"error": "Voice prediction failed"}), 500
# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    print("🚀 Starting PD Backend Server...")
    app.run(host="0.0.0.0", port=5000, debug=False)