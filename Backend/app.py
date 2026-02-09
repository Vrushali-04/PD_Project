from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from models.predict_image import predict_image  # your updated predict_image.py

# ==============================
# CONFIG
# ==============================
app = Flask(__name__)
CORS(app)  # allow cross-origin requests

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


# Image prediction route
@app.route("/predict_image", methods=["POST"])
def predict_image_api():
    # ⚡ The key must match frontend FormData key: "image"
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        # Save uploaded file temporarily
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)

        # Predict
        result = predict_image(file_path)

        # Remove temporary file
        os.remove(file_path)

        # Handle prediction errors (brain check etc.)
        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        # Catch unexpected errors
        print("Error:", e)
        return jsonify({"error": "Failed to process image"}), 500


# ==============================
# RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)
