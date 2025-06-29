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
from itertools import cycle

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
UPLOAD_FOLDER = os.path.join("/tmp", "uploads")
OUTPUT_FOLDER = os.path.join("/tmp", "static", "output")
STATIC_FOLDER = os.path.join(ROOT_DIR, "static")
INDEX_HTML_PATH = os.path.join(ROOT_DIR, "index.html")
BACKGROUND_PATH = os.path.join(STATIC_FOLDER, "custom_background.jpg")
CHARACTER_PATH = os.path.join(STATIC_FOLDER, "character.png")

REMOVE_BG_KEYS = [os.getenv(f"REMOVE_BG_KEY_{i}") for i in range(1, 21)]
REMOVE_BG_KEYS = [key for key in REMOVE_BG_KEYS if key]
key_cycle = cycle(REMOVE_BG_KEYS)
REMOVE_BG_API_URL = "https://api.remove.bg/v1.0/removebg"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def cleanup_old_files(folder, max_age_seconds=3600):
    now = time.time()
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path) and os.path.getmtime(path) < now - max_age_seconds:
            os.remove(path)

def remove_background_with_removebg(image_path):
    for _ in range(len(REMOVE_BG_KEYS)):
        current_key = next(key_cycle)
        try:
            with open(image_path, 'rb') as image_file:
                response = requests.post(
                    REMOVE_BG_API_URL,
                    files={"image_file": image_file},
                    data={"size": "preview"},
                    headers={"X-Api-Key": current_key},
                )
                if response.status_code == 200:
                    with NamedTemporaryFile(delete=False, suffix=".png") as temp_result:
                        temp_result.write(response.content)
                        return temp_result.name
                elif response.status_code == 402:
                    continue
                else:
                    raise Exception(f"Remove.bg API error: {response.status_code} {response.text}")
        except Exception:
            continue
    raise Exception("All API keys exhausted or failed.")

def blend_with_background(foreground_path):
    fg_image = Image.open(foreground_path).convert("RGBA")
    background = Image.open(BACKGROUND_PATH).convert("RGBA").resize(fg_image.size)
    combined = Image.alpha_composite(background, fg_image)

    if os.path.exists(CHARACTER_PATH):
        character = Image.open(CHARACTER_PATH).convert("RGBA")
        scale = 0.35
        new_width = int(fg_image.width * scale)
        new_height = int(character.height * (new_width / character.width))
        character = character.resize((new_width, new_height))
        x = fg_image.width - new_width - 10
        y = fg_image.height - new_height - 10
        combined.paste(character, (x, y), character)

    return combined.convert("RGB")

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
        final_image.save(output_path, format="JPEG")

        return jsonify({"image_url": f"/static/output/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/hello")
def hello():
    return jsonify({"message": "Mira API is live ✅"}), 200

if __name__ == "__main__":
    app.run(debug=True)
