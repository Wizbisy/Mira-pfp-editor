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
REMOVE_BG_KEYS = [k for k in REMOVE_BG_KEYS if k]
key_cycle = cycle(REMOVE_BG_KEYS)
REMOVE_BG_API_URL = "https://api.remove.bg/v1.0/removebg"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def cleanup_old_files(folder, max_age=3600):
    now = time.time()
    for f in os.listdir(folder):
        p = os.path.join(folder, f)
        if os.path.isfile(p) and os.path.getmtime(p) < now - max_age:
            os.remove(p)

def remove_background(image_path):
    for _ in range(len(REMOVE_BG_KEYS)):
        key = next(key_cycle)
        with open(image_path, "rb") as img:
            res = requests.post(
                REMOVE_BG_API_URL,
                files={"image_file": img},
                data={"size": "preview"},
                headers={"X-Api-Key": key},
            )
        if res.status_code == 200:
            with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(res.content)
                return tmp.name
        if res.status_code != 402:
            break
    raise Exception("remove.bg failed or keys exhausted")

def blend_with_background(foreground_path):
    fg = Image.open(foreground_path).convert("RGBA")
    bg = Image.open(BACKGROUND_PATH).convert("RGBA")

    bw, bh = bg.size
    fw, fh = fg.size

    ratio = max(fw / bw, fh / bh)
    target = 0.75
    tol = 0.1

    if ratio < target - tol:
        scale = min(target / ratio, 3.0)
        fg = fg.resize((int(fw * scale), int(fh * scale)), Image.LANCZOS)
    elif ratio > target + tol:
        scale = target / ratio
        fg = fg.resize((int(fw * scale), int(fh * scale)), Image.LANCZOS)

    fw, fh = fg.size
    x = (bw - fw) // 2
    y = (bh - fh) // 2

    out = bg.copy()
    out.paste(fg, (x, y), fg)

    if os.path.exists(CHARACTER_PATH):
        char = Image.open(CHARACTER_PATH).convert("RGBA")
        c_scale = 0.35
        cw = int(bw * c_scale)
        ch = int(char.height * (cw / char.width))
        char = char.resize((cw, ch))
        out.paste(char, (bw - cw - 10, bh - ch - 10), char)

    return out.convert("RGB")

@app.route("/")
def index():
    return send_file(INDEX_HTML_PATH)

@app.route("/static/<path:path>")
def static_proxy(path):
    return send_from_directory(STATIC_FOLDER, path)

@app.route("/static/output/<filename>")
def output_proxy(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype="image/jpeg")

@app.route("/api/process", methods=["POST"])
def process_image():
    try:
        cleanup_old_files(UPLOAD_FOLDER)
        cleanup_old_files(OUTPUT_FOLDER)

        if "file" not in request.files:
            return jsonify(error="No file uploaded"), 400

        file = request.files["file"]
        if not file.filename:
            return jsonify(error="Empty filename"), 400

        file.seek(0, os.SEEK_END)
        if file.tell() > 2 * 1024 * 1024:
            return jsonify(error="Image must be under 2 MB"), 400
        file.seek(0)

        tmp = NamedTemporaryFile(delete=False, dir=UPLOAD_FOLDER)
        file.save(tmp.name)

        fg_path = remove_background(tmp.name)
        final_img = blend_with_background(fg_path)

        os.remove(tmp.name)
        os.remove(fg_path)

        out_name = f"{uuid.uuid4().hex}.jpg"
        out_path = os.path.join(OUTPUT_FOLDER, out_name)
        final_img.save(out_path, format="JPEG")

        return jsonify(image_url=f"/static/output/{out_name}")
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route("/api/hello")
def hello():
    return jsonify(message="Mira API is live ✅")

if __name__ == "__main__":
    app.run(debug=True)
