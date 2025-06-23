import os
import uuid
import time
import cv2
import numpy as np
from PIL import Image
from tempfile import NamedTemporaryFile
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import torch
from torchvision import transforms
from u2net import U2NETP  # Ensure this file is in your directory

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
MODEL_PATH = os.path.join(BASE_DIR, "models", "u2netp.pth")

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load U2Net model
def load_u2netp_model():
    model = U2NETP(3, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    return model

u2net_model = load_u2netp_model()

def cleanup_old_files(folder, max_age_seconds=3600):
    now = time.time()
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path) and os.path.getmtime(path) < now - max_age_seconds:
            os.remove(path)

def remove_background(image_path):
    pil_image = Image.open(image_path).convert("RGB")
    original_np = np.array(pil_image)
    w, h = pil_image.size

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        d1, *_ = u2net_model(input_tensor)
        mask = d1.squeeze().cpu().numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min())
        mask = cv2.resize(mask, (w, h))
        alpha = mask.astype(np.float32)
        alpha_3c = cv2.merge([alpha, alpha, alpha])

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(float)

    if not os.path.exists(BACKGROUND_PATH):
        raise FileNotFoundError("custom_background.jpg not found")
    background = cv2.imread(BACKGROUND_PATH)
    background = cv2.resize(background, (w, h)).astype(float)

    fg = cv2.multiply(alpha_3c, image)
    bg = cv2.multiply(1.0 - alpha_3c, background)
    result = cv2.add(fg, bg).astype(np.uint8)

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
            alpha_c_3c = cv2.merge([alpha_c, alpha_c, alpha_c])

            for c in range(3):
                result[y:y+new_height, x:x+new_width, c] = (
                    (1 - alpha_c_3c[:, :, c]) * result[y:y+new_height, x:x+new_width, c] +
                    alpha_c_3c[:, :, c] * character[:, :, c]
                )

    return result

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
