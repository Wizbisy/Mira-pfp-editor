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

    mask = np.zeros(image.shape[:2], np.uint8)

    # Create models for GrabCut (required, but unused explicitly)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Define initial rectangle for subject (covering whole image)
    height, width = image.shape[:2]
    rect = (1, 1, width-2, height-2)

    # Apply GrabCut
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Create soft alpha mask
    blurred_mask = cv2.GaussianBlur(mask2 * 255, (7, 7), 0)
    alpha = blurred_mask.astype(float) / 255.0

    # Extract foreground
    fg = image.astype(float)
    fg = cv2.multiply(alpha[:, :, np.newaxis], fg)

    # Load and prepare background
    background = cv2.imread(BACKGROUND_PATH)
    background = cv2.resize(background, (width, height))
    bg = cv2.multiply(1.0 - alpha[:, :, np.newaxis], background.astype(float))

    # Blend foreground and background
    final = cv2.add(fg, bg).astype(np.uint8)

    # Optional: Add character overlay
    if os.path.exists(CHARACTER_PATH):
        character = cv2.imread(CHARACTER_PATH, cv2.IMREAD_UNCHANGED)
        if character is not None and character.shape[2] == 4:
            scale = 0.3
            new_width = int(width * scale)
            new_height = int(character.shape[0] * (new_width / character.shape[1]))
            character = cv2.resize(character, (new_width, new_height))

            x = width - new_width - 20
            y = height - new_height - 20
            alpha_c = character[:, :, 3] / 255.0

            for c in range(3):
                final[y:y+new_height, x:x+new_width, c] = (
                    (1 - alpha_c) * final[y:y+new_height, x:x+new_width, c] +
                    alpha_c * character[:, :, c]
                )

    return final

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
