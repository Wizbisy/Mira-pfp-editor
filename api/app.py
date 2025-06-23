import os
import uuid
import time
import cv2
import numpy as np
from PIL import Image
from tempfile import NamedTemporaryFile
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import requests

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

REMOVE_BG_API_KEY = os.environ.get("REMOVE_BG_API_KEY")
REMOVE_BG_API_URL = "https://api.remove.bg/v1.0/removebg"

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def cleanup_old_files(folder, max_age_seconds=3600):
    now = time.time()
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path) and os.path.getmtime(path) < now - max_age_seconds:
            os.remove(path)

def remove_background_with_removebg(image_path):
    with open(image_path, 'rb') as image_file:
        response = requests.post(
            REMOVE_BG_API_URL,
            files={"image_file": image_file},
            data={"size": "preview"},
            headers={"X-Api-Key": REMOVE_BG_API_KEY},
        )
        if response.status_code != 200:
            raise Exception(f"Remove.bg API error: {response.status_code} {response.text}")

        with NamedTemporaryFile(delete=False, suffix=".png") as temp_result:
            temp_result.write(response.content)
            result_path = temp_result.name

    return result_path

def blend_with_background(foreground_path):
    fg_image = Image.open(foreground_path).convert("RGBA")
    fg_np = np.array(fg_image)
    h, w = fg_np.shape[:2]

    # Load and resize background
    if not os.path.exists(BACKGROUND_PATH):
        raise FileNotFoundError("custom_background.jpg not found")
    background = cv2.imread(BACKGROUND_PATH)
    background = cv2.resize(background, (w, h))

    alpha = fg_np[:, :, 3] / 255.0
    alpha_3c = cv2.merge([alpha] * 3)
    foreground_rgb = fg_np[:, :, :3]

    blended = np.clip(alpha_3c * foreground_rgb + (1 - alpha_3c) * background, 0, 255).astype(np.uint8)

    # Add character overlay
    if os.path.exists(CHARACTER_PATH):
        character = cv2.imread(CHARACTER_PATH, cv2.IMREAD_UNCHANGED)
        if character is not None and character.shape[2] == 4:
            scale = 0.3
            new_width = int(w * scale)
            new_height = int(character.shape[0] * (new_width / character.shape[1]))
            character = cv2.resize(character, (new_width, new_height))

            x = w - new_width - 20
            y = h - new_height - 20
            alpha_c = character[:, :, 3] / 255.0
            alpha_c_3c = cv2.merge([alpha_c] * 3)

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

        removed_path = remove_background_with_removebg(temp_file.name)
        final_image = blend_with_background(removed_path)

        os.remove(temp_file.name)
        os.remove(removed_path)

        filename = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        Image.fromarray(final_image).save(output_path, format="JPEG")

        return jsonify({"image_url": f"/static/output/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/hello")
def hello():
    return jsonify({"message": "Mira API is live ✅"}), 200

if __name__ == "__main__":
    app.run(debug=True)
