import os
import base64
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
import mediapipe as mp

app = Flask(__name__)

os.makedirs("dataset/users", exist_ok=True)

# Face detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)


# ---------------- LOGIN ----------------
@app.route("/")
def login():
    return render_template("login.html")


# ---------------- VERIFY PAGE ----------------
@app.route("/verify/<regno>")
def verify(regno):
    return render_template("verify.html", regno=regno)


# ---------------- REGISTER PAGE ----------------
@app.route("/register")
def register():
    return render_template("register.html")


# ---------------- REGISTER USER ----------------
@app.route("/register_user", methods=["POST"])
def register_user():

    data = request.json
    regno = data["regno"]
    name = data["name"]
    images = data["images"]

    user_folder = f"dataset/users/{regno}"
    os.makedirs(user_folder, exist_ok=True)

    for i, img in enumerate(images):
        img = img.split(",")[1]
        img_bytes = base64.b64decode(img)

        with open(f"{user_folder}/face_{i}.jpg", "wb") as f:
            f.write(img_bytes)

    with open(f"{user_folder}/info.txt", "w") as f:
        f.write(name)

    return jsonify({"status": "success"})


# ---------------- VERIFY FACE ----------------
@app.route("/verify_face/<regno>")
def verify_face(regno):

    user_folder = f"dataset/users/{regno}"

    if not os.path.exists(user_folder):
        return jsonify({"status": "not_registered"})

    camera = cv2.VideoCapture(0)

    ret, frame = camera.read()
    camera.release()

    if not ret:
        return jsonify({"status": "failed"})

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_detection.process(rgb)

    if not faces.detections:
        return jsonify({"status": "no_face_detected"})

    h, w, _ = frame.shape
    bbox = faces.detections[0].location_data.relative_bounding_box

    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    w_box = int(bbox.width * w)
    h_box = int(bbox.height * h)

    face = frame[y:y+h_box, x:x+w_box]

    if face.size == 0:
        return jsonify({"status": "failed"})

    face = cv2.resize(face, (200, 200))

    scores = []

    # Compare with all 3 images
    for i in range(3):

        stored_path = f"{user_folder}/face_{i}.jpg"

        if not os.path.exists(stored_path):
            continue

        stored = cv2.imread(stored_path)
        stored = cv2.resize(stored, (200, 200))

        diff = cv2.absdiff(face, stored)
        score = diff.mean()

        print(f"Image {i} score:", score)

        scores.append(score)

    if len(scores) == 0:
        return jsonify({"status": "failed"})

    avg_score = sum(scores) / len(scores)

    print("Average score:", avg_score)

    # STRICT threshold
    if avg_score < 50:
        return jsonify({"status": "success"})

    return jsonify({"status": "failed"})


# ---------------- DASHBOARD ----------------
@app.route("/dashboard/<regno>")
def dashboard(regno):

    name = "Unknown"

    info_file = f"dataset/users/{regno}/info.txt"

    if os.path.exists(info_file):
        with open(info_file) as f:
            name = f.read()

    return render_template("dashboard.html", regno=regno, name=name)


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)