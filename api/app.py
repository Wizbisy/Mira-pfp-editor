import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tempfile import NamedTemporaryFile
import uuid
import time

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join('/tmp', 'uploads')
OUTPUT_FOLDER = os.path.join('/tmp', 'static/output')
BACKGROUND_PATH = os.path.join(BASE_DIR, 'static/custom_background.jpg')
CHARACTER_PATH = os.path.join(BASE_DIR, 'static/character.png')

for folder in (UPLOAD_FOLDER, OUTPUT_FOLDER):
    os.makedirs(folder, exist_ok=True)

def cleanup_old_files(folder, max_age_seconds=3600):
    current_time = time.time()
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path) and os.path.getmtime(file_path) < current_time - max_age_seconds:
            try:
                os.remove(file_path)
                print(f"Deleted old file: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def remove_background(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    result = cv2.bitwise_and(image, image, mask=mask)

    if not os.path.exists(BACKGROUND_PATH):
        raise FileNotFoundError("Custom background image not found")
    background = cv2.imread(BACKGROUND_PATH)
    background = cv2.resize(background, (image.shape[1], image.shape[0]))

    background = cv2.bitwise_and(background, background, mask=cv2.bitwise_not(mask))
    final_result = cv2.add(result, background)

    if os.path.exists(CHARACTER_PATH):
        character = cv2.imread(CHARACTER_PATH, cv2.IMREAD_UNCHANGED)
        if character is not None:
            char_height, char_width = character.shape[:2]
            char_scale = 0.3
            char_width = int(image.shape[1] * char_scale)
            char_height = int(char_height * (char_width / char_width))
            character = cv2.resize(character, (char_width, char_height), cv2.INTER_AREA)
            x, y = image.shape[1] - char_width - 20, image.shape[0] - char_height - 20
            if character.shape[2] == 4:
                alpha_mask = character[:, :, 3] / 255.0
                for c in range(0, 3):
                    final_result[y:y+char_height, x:x+char_width, c] = (
                        (1.0 - alpha_mask) * final_result[y:y+char_height, x:x+char_width, c] +
                        alpha_mask * character[:, :, c]
                    )

    return final_result

@app.route('/')
def home():
    return jsonify({"message": "Mira Background API is running ✅"}), 200

@app.route('/api/process', methods=['POST'])
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
        temp_file_path = temp_file.name
        file.save(temp_file_path)

        output_image = remove_background(temp_file_path)

        unique_name = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, unique_name)
        cv2.imwrite(output_path, output_image)

        os.remove(temp_file_path)

        return jsonify({"image_url": f"/static/output/{unique_name}"})

    except Exception as e:
        print(f"🔥 Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/static/output/<filename>')
def serve_image(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename), mimetype='image/jpeg')

def handler(event, context):
    from wsgi import WSGIHandler
    wsgi_handler = WSGIHandler()
    return wsgi_handler(event, context)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
