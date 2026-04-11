from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import bcrypt
import uuid

from db import db

# Import our AI "Doctors" so we can ask them questions later
from models.predict_mri import predict_mri_scan
from models.predict_spiral import predict_spiral

# ==========================================
# CONFIGURATION (SETTING UP THE SERVER)
# ==========================================
app = Flask(__name__)

# WHY WE NEED CORS: React usually runs on one port (like localhost:3000 or 8081) 
# and Flask runs on another (localhost:5000). Browsers naturally block them from 
# talking to each other for security. CORS is the "VIP Pass" that lets React talk to Flask.
CORS(app)

# We need a temporary "waiting room" to hold images before the AI looks at them.
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True) # Creates the folder if it doesn't exist yet
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================================
# HOME & SYSTEM ROUTES
# ==========================================
# A simple heartbeat check to make sure the server didn't crash
@app.route("/")
def home():
    return jsonify({"status": "PD Backend Running"})

# Browsers always automatically ask for a tiny icon. This stops the terminal from printing an error.
@app.route("/favicon.ico")
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, "static"),
        "favicon.ico",
        mimetype="image/vnd.microsoft.icon"
    )


# ==========================================
# AUTHENTICATION ROUTES (LOGIN/SIGNUP)
# ==========================================

@app.route("/signup", methods=["POST"])
def signup():
    try:
        # 1. Get the data from the React frontend
        data = request.get_json()
        if not data:
            return jsonify({"message": "No JSON data received"}), 400

        name = data.get("name")
        email = data.get("email")
        password = data.get("password")

        if not name or not email or not password:
            return jsonify({"message": "All fields are required"}), 400

        cursor = db.cursor(dictionary=True)

        # 2. Check if this email is already in our database
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        existing_user = cursor.fetchone()

        if existing_user:
            return jsonify({"message": "Email already registered"}), 400

        # 3. THE PAPER SHREDDER (bcrypt)
        # We NEVER save the actual password (like "password123") in the database. 
        # We "shred" it into a random hash. If a hacker steals the database, they can't read the passwords.
        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        # 4. Save the new user
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (name, email, hashed_pw)
        )
        db.commit()

        print(f"[SUCCESS] New user registered: {email}")
        return jsonify({"message": "Signup successful"}), 201

    except Exception as e:
        print(f"[ERROR] Signup Error: {e}")
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

        # 1. Find the user in the database
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()

        # 2. Compare the password typed in React to the "shredded" password in the database
        if user and bcrypt.checkpw(password.encode(), user["password"].encode()):
            print(f"[SUCCESS] Login successful: {email}")
            return jsonify({
                "message": "Login successful",
                "name": user["name"]
            })

        return jsonify({"message": "Invalid email or password"}), 401

    except Exception as e:
        print(f"[ERROR] Login Error: {e}")
        return jsonify({"message": "Server error during login"}), 500


@app.route("/users", methods=["GET"])
def get_users():
    try:
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT id, name, email FROM users")
        users = cursor.fetchall()
        return jsonify(users)
    except Exception as e:
        print(f"[ERROR] Fetch Users Error: {e}")
        return jsonify({"message": "Error fetching users"}), 500


# ==========================================
# MRI PREDICTION ROUTE
# ==========================================
@app.route("/predict_image", methods=["POST"])
def predict_image_api():
    try:
        # 1. Check if React actually sent a file
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # 2. THE RENAME TRICK (UUID)
        # If two users upload "brain.jpg" at the exact same time, they will overwrite each other.
        # UUID generates a random gibberish name (like "8f7d9a...jpg") to prevent collisions.
        ext = file.filename.rsplit(".", 1)[-1].lower()
        filename = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        print(f"[INFO] MRI Image Received: {filename}")

        # 3. Hand the saved image to the MRI AI Pipeline
        result = predict_mri_scan(file_path)

        # 4. THE JANITOR
        # We don't want our server hard drive to fill up with thousands of user images. 
        # Now that the AI is done looking at it, delete it immediately.
        if os.path.exists(file_path):
            os.remove(file_path)

        print(f"[SUCCESS] MRI Prediction: {result}")
        return jsonify(result) # Send the answer back to React

    except Exception as e:
        print(f"[ERROR] MRI Prediction Error: {e}")
        return jsonify({"error": "Failed to process image"}), 500

# ==========================================
# SPIRAL HANDWRITING PREDICTION ROUTE
# ==========================================
@app.route("/predict_spiral", methods=["POST"])
def predict_spiral_route():
    try:
        # 1. Grab the image
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # 2. Security Check: Make sure they didn't upload a PDF or an .exe virus
        allowed_extensions = {"png", "jpg", "jpeg"}
        ext = file.filename.rsplit(".", 1)[-1].lower()

        if ext not in allowed_extensions:
            return jsonify({"error": "Invalid file type"}), 400

        # 3. Rename and save the image temporarily
        filename = f"{uuid.uuid4()}.{ext}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        print(f"[INFO] Spiral Image Received: {filename}")

        # 4. Hand it to the Spiral AI Pipeline (which runs the OpenCV Bouncer first)
        result = predict_spiral(file_path)

        print(f"[SUCCESS] Spiral Prediction: {result}")

        # 5. Delete the image to save server space
        if os.path.exists(file_path):
            os.remove(file_path)

        return jsonify(result)

    except Exception as e:
        print(f"[ERROR] Spiral Prediction Error: {e}")
        return jsonify({"error": "Failed to process spiral image"}), 500


# ==========================================
# RUN THE SERVER
# ==========================================
if __name__ == "__main__":
    print("[INFO] Starting PD Backend Server...")
    # host="0.0.0.0" means the server will accept connections from anywhere (like your React frontend)
    app.run(host="0.0.0.0", port=5000, debug=True)