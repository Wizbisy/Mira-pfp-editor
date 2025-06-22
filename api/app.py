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
    copy_static_files()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create binary mask using OTSU thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert mask to get foreground
    binary = cv2.bitwise_not(binary)

    # Improve mask with morphological operations
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Apply Gaussian blur to soften edges
    binary = cv2.GaussianBlur(binary, (5, 5), 0)

    # Apply the mask to the image
    result = cv2.bitwise_and(image, image, mask=binary)

    # Load and resize background
    if not os.path.exists(BACKGROUND_PATH):
        raise FileNotFoundError("Custom background image not found")
    background = cv2.imread(BACKGROUND_PATH)
    background = cv2.resize(background, (image.shape[1], image.shape[0]))

    # Blend background and result
    inverse_mask = cv2.bitwise_not(binary)
    background = cv2.bitwise_and(background, background, mask=inverse_mask)
    final_result = cv2.add(result, background)

    # Add character overlay if available
    if os.path.exists(CHARACTER_PATH):
        character = cv2.imread(CHARACTER_PATH, cv2.IMREAD_UNCHANGED)
        if character is not None and character.shape[2] == 4:
            char_height, char_width = character.shape[:2]
            scale_factor = 0.3
            new_width = int(image.shape[1] * scale_factor)
            new_height = int(char_height * (new_width / char_width))
            character = cv2.resize(character, (new_width, new_height), cv2.INTER_AREA)
            x, y = image.shape[1] - new_width - 20, image.shape[0] - new_height - 20

            alpha = character[:, :, 3] / 255.0
            for c in range(3):
                final_result[y:y+new_height, x:x+new_width, c] = (
                    alpha * character[:, :, c] +
                    (1 - alpha) * final_result[y:y+new_height, x:x+new_width, c]
                )

    return final_result

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
