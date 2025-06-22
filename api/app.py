import os
from flask import Flask, request, jsonify
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
PUBLIC_STATIC_DIR = os.path.join('/tmp', 'public/static')
BACKGROUND_PATH = os.path.join(PUBLIC_STATIC_DIR, 'custom_background.jpg')
CHARACTER_PATH = os.path.join(PUBLIC_STATIC_DIR, 'character.png')

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

def copy_static_files():
    if not os.path.exists(PUBLIC_STATIC_DIR):
        os.makedirs(PUBLIC_STATIC_DIR)
    source_static_dir = os.path.join(BASE_DIR, '../../public/static')
    for file in ['custom_background.jpg', 'character.png']:
        source = os.path.join(source_static_dir, file)
        dest = os.path.join(PUBLIC_STATIC_DIR, file)
        if os.path.exists(source):
            shutil.copy2(source, dest)
            print(f"Copied {file} to {dest}")
        else:
            print(f"Source {source} not found")

def remove_background(image_path):
    copy_static_files()
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
    import io
    from urllib.parse import parse_qs

    if event.get('httpMethod') == 'GET':
        if event.get('path') == '/':
            return {
                'statusCode': 200,
                'body': jsonify({"message": "Mira Background API is running ✅"}).data.decode('utf-8'),
                'headers': {'Content-Type': 'application/json'}
            }
        elif event.get('path').startswith('/static/output/'):
            filename = event['path'].replace('/static/output/', '')
            file_path = os.path.join(OUTPUT_FOLDER, filename)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    return {
                        'statusCode': 200,
                        'body': f.read().decode('utf-8') if filename.endswith('.txt') else f.read(),
                        'headers': {'Content-Type': 'image/jpeg' if filename.endswith('.jpg') else 'application/octet-stream'}
                    }
            return {'statusCode': 404, 'body': 'File not found'}

    elif event.get('httpMethod') == 'POST' and event.get('path') == '/api/process':
        try:
            cleanup_old_files(UPLOAD_FOLDER)
            cleanup_old_files(OUTPUT_FOLDER)

            if 'file' not in event.get('multiValueHeaders', {}):
                return {'statusCode': 400, 'body': jsonify({"error": "No file uploaded"}).data.decode('utf-8'), 'headers': {'Content-Type': 'application/json'}}

            body = event.get('body')
            if not body:
                return {'statusCode': 400, 'body': jsonify({"error": "Empty filename"}).data.decode('utf-8'), 'headers': {'Content-Type': 'application/json'}}

            import base64
            file_data = base64.b64decode(parse_qs(body)['file'][0])
            file.seek(0, os.SEEK_END)
            if file.tell() > 2 * 1024 * 1024:
                return {'statusCode': 400, 'body': jsonify({"error": "Image must be under 2MB"}).data.decode('utf-8'), 'headers': {'Content-Type': 'application/json'}}
            file.seek(0)

            temp_file = NamedTemporaryFile(delete=False, dir=UPLOAD_FOLDER)
            temp_file_path = temp_file.name
            with open(temp_file_path, 'wb') as f:
                f.write(file_data)

            output_image = remove_background(temp_file_path)

            unique_name = f"{uuid.uuid4().hex}.jpg"
            output_path = os.path.join(OUTPUT_FOLDER, unique_name)
            cv2.imwrite(output_path, output_image)

            os.remove(temp_file_path)

            return {'statusCode': 200, 'body': jsonify({"image_url": f"/static/output/{unique_name}"}).data.decode('utf-8'), 'headers': {'Content-Type': 'application/json'}}

        except Exception as e:
            print(f"🔥 Error: {e}")
            return {'statusCode': 500, 'body': jsonify({"error": str(e)}).data.decode('utf-8'), 'headers': {'Content-Type': 'application/json'}}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
