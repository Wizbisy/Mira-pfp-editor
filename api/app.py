import os
import uuid
import time
import shutil
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Folder paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
UPLOAD_FOLDER = os.path.join("/tmp", "uploads")
OUTPUT_FOLDER = os.path.join("/tmp", "static", "output")
STATIC_FOLDER = os.path.join(ROOT_DIR, "static")
INDEX_HTML = os.path.join(ROOT_DIR, "index.html")
BACKGROUND_PATH = os.path.join(STATIC_FOLDER, "custom_background.jpg")
CHARACTER_PATH = os.path.join(STATIC_FOLDER, "character.png")

# Create temp folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Helpers
def cleanup_old_files(folder, max_age_seconds=3600):
    now = time.time()
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and os.path.getmtime(path) < now - max_age_seconds:
            try:
                os.remove(path)
            except Exception as e:
                print(f"❌ Failed to delete {path}: {e}")

def remove_background(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to read image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    result = cv2.bitwise_and(image, image, mask=mask)

    if not os.path.exists(BACKGROUND_PATH):
        raise FileNotFoundError("Background not found")
    background = cv2.imread(BACKGROUND_PATH)
    background = cv2.resize(background, (image.shape[1], image.shape[0]))
    background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))
    final = cv2.add(result, background)

    if os.path.exists(CHARACTER_PATH):
        character = cv2.imread(CHARACTER_PATH, cv2.IMREAD_UNCHANGED)
        if character is not None and character.shape[2] == 4:
            char_scale = 0.3
            char_width = int(image.shape[1] * char_scale)
            char_height = int(character.shape[0] * (char_width / character.shape[1]))
            character = cv2.resize(character, (char_width, char_height), interpolation=cv2.INTER_AREA)

            x, y = image.shape[1] - char_width - 20, image.shape[0] - char_height - 20
            alpha = character[:, :, 3] / 255.0
            for c in range(3):
                final[y:y+char_height, x:x+char_width, c] = (
                    (1. - alpha) * final[y:y+char_height, x:x+char_width, c] +
                    alpha * character[:, :, c]
                )

    return final

# === Routes ===

@app.route("/")
def serve_frontend():
    return send_file(INDEX_HTML)

@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_FOLDER, path)

@app.route("/static/output/<filename>")
def serve_output(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype="image/jpeg")

@app.route("/api/process", methods=["POST"])
def process_image():
    try:
        cleanup_old_files(UPLOAD_FOLDER)
        cleanup_old_files(OUTPUT_FOLDER)

        file = request.files.get("file")
        if not file or file.filename == "":
            return jsonify({"error": "No file uploaded"}), 400

        file.seek(0, os.SEEK_END)
        if file.tell() > 2 * 1024 * 1024:
            return jsonify({"error": "Image must be under 2MB"}), 400
        file.seek(0)

        tmp = NamedTemporaryFile(delete=False, dir=UPLOAD_FOLDER)
        file.save(tmp.name)

        result = remove_background(tmp.name)
        os.remove(tmp.name)

        filename = f"{uuid.uuid4().hex}.jpg"
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(out_path, result)

        return jsonify({"image_url": f"/static/output/{filename}"})

    except Exception as e:
        print("🔥 ERROR:", e)
        return jsonify({"error": str(e)}), 500

# Optional: for health check
@app.route("/api/hello")
def hello():
    return jsonify({"message": "Mira Background Changer is live ✅"}), 200

# Local debug
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
