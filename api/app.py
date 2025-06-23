import os
import uuid
import time
import shutil
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
UPLOAD_FOLDER = os.path.join("/tmp", "uploads")
OUTPUT_FOLDER = os.path.join("/tmp", "static", "output")
STATIC_FOLDER = os.path.join(ROOT_DIR, "static")
INDEX_HTML_PATH = os.path.join(ROOT_DIR, "index.html")
BACKGROUND_PATH = os.path.join(STATIC_FOLDER, "custom_background.jpg")
CHARACTER_PATH = os.path.join(STATIC_FOLDER, "character.png")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def cleanup_old_files(folder, max_age_seconds=3600):
    now = time.time()
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path) and os.path.getmtime(path) < now - max_age_seconds:
            os.remove(path)

def remove_background(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")

    height, width = image.shape[:2]

    # Step 1: GrabCut mask
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 1, width - 2, height - 2)

    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Step 2: Smooth alpha mask
    soft_mask = cv2.GaussianBlur(mask2 * 255, (7, 7), 0)
    alpha = soft_mask.astype(float) / 255.0
    alpha_3c = cv2.merge([alpha, alpha, alpha])  # <-- Fix: match 3 channels

    # Step 3: Prepare foreground and background
    foreground = image.astype(float)
    foreground = cv2.multiply(alpha_3c, foreground)

    if not os.path.exists(BACKGROUND_PATH):
        raise FileNotFoundError("custom_background.jpg not found")

    background = cv2.imread(BACKGROUND_PATH)
    background = cv2.resize(background, (width, height)).astype(float)
    background = cv2.multiply(1.0 - alpha_3c, background)

    blended = cv2.add(foreground, background).astype(np.uint8)

    # Step 4: Add character overlay
    if os.path.exists(CHARACTER_PATH):
        character = cv2.imread(CHARACTER_PATH, cv2.IMREAD_UNCHANGED)
        if character is not None and character.shape[2] == 4:
            scale = 0.3
            new_width = int(width * scale)
            new_height = int(character.shape[0] * (new_width / character.shape[1]))
            character = cv2.resize(character, (new_width, new_height))

            x = max(0, width - new_width - 20)
            y = max(0, height - new_height - 20)
            alpha_c = character[:, :, 3] / 255.0
            alpha_c_3c = cv2.merge([alpha_c, alpha_c, alpha_c])

            for c in range(3):
                blended[y:y+new_height, x:x+new_width, c] = (
                    (1 - alpha_c_3c[:, :, c]) * blended[y:y+new_height, x:x+new_width, c] +
                    alpha_c_3c[:, :, c] * character[:, :, c]
                )

    return blended

@app.route("/")
def serve_index():
    return send_file(INDEX_HTML_PATH)

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

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        file.seek(0, os.SEEK_END)
        if file.tell() > 2 * 1024 * 1024:
            return jsonify({"error": "Image must be under 2MB"}), 400
        file.seek(0)

        temp_file = NamedTemporaryFile(delete=False, dir=UPLOAD_FOLDER)
        file.save(temp_file.name)

        final_image = remove_background(temp_file.name)
        os.remove(temp_file.name)

        filename = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        cv2.imwrite(output_path, final_image)

        return jsonify({"image_url": f"/static/output/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/hello")
def hello():
    return jsonify({"message": "Mira API is live ✅"}), 200

if __name__ == "__main__":
    app.run(debug=True)
