import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
import uuid
import time
import shutil

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join('/tmp', 'uploads')
OUTPUT_FOLDER = os.path.join('/tmp', 'static/output')
STATIC_ASSETS = os.path.join(BASE_DIR, '../static')
BACKGROUND_PATH = os.path.join(STATIC_ASSETS, 'custom_background.jpg')
CHARACTER_PATH = os.path.join(STATIC_ASSETS, 'character.png')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def cleanup_old_files(folder, max_age_seconds=3600):
    now = time.time()
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and os.path.getmtime(file_path) < now - max_age_seconds:
            os.remove(file_path)

def remove_background(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image.")

    # Clean mask for better separation
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    mask = cv2.GaussianBlur(binary, (5,5), 0)

    fg = cv2.bitwise_and(image, image, mask=mask)

    if not os.path.exists(BACKGROUND_PATH):
        raise FileNotFoundError("Background not found.")
    background = cv2.imread(BACKGROUND_PATH)
    background = cv2.resize(background, (image.shape[1], image.shape[0]))
    background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))

    final = cv2.add(fg, background)

    if os.path.exists(CHARACTER_PATH):
        character = cv2.imread(CHARACTER_PATH, cv2.IMREAD_UNCHANGED)
        if character is not None and character.shape[2] == 4:
            h, w = character.shape[:2]
            scale = 0.3
            new_w = int(final.shape[1] * scale)
            new_h = int(h * (new_w / w))
            character = cv2.resize(character, (new_w, new_h), interpolation=cv2.INTER_AREA)
            x, y = final.shape[1] - new_w - 20, final.shape[0] - new_h - 20
            alpha = character[:, :, 3] / 255.0
            for c in range(3):
                final[y:y+new_h, x:x+new_w, c] = (
                    (1 - alpha) * final[y:y+new_h, x:x+new_w, c] +
                    alpha * character[:, :, c]
                )

    return final

@app.route('/')
def home():
    return jsonify({"message": "Mira Background API is running ✅"}), 200

@app.route('/api/process', methods=['POST'])
def process():
    try:
        cleanup_old_files(UPLOAD_FOLDER)
        cleanup_old_files(OUTPUT_FOLDER)

        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({"error": "Empty filename"}), 400

        if file.mimetype not in ['image/jpeg', 'image/png']:
            return jsonify({"error": "Unsupported image format"}), 400

        file.seek(0, os.SEEK_END)
        if file.tell() > 2 * 1024 * 1024:
            return jsonify({"error": "Image must be under 2MB"}), 400
        file.seek(0)

        tmp = NamedTemporaryFile(delete=False, dir=UPLOAD_FOLDER)
        file.save(tmp.name)

        output_image = remove_background(tmp.name)
        output_filename = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, output_image)
        os.remove(tmp.name)

        return jsonify({"image_url": f"/static/output/{output_filename}"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/static/output/<filename>')
def serve_output(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
