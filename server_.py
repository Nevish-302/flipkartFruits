import tensorflow as tf
from flask import Flask, request, jsonify
import cv2
import requests
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
import traceback
import logging
import sys
from functools import lru_cache
import os
import gdown


def check_and_download_file(file_path, google_drive_url):
    # Check if the file exists
    if not os.path.isfile(file_path):
        print(f"{file_path} does not exist. Downloading from Google Drive...")

        # Download the file
        gdown.download(google_drive_url, file_path, quiet=False)
        print(f"Downloaded {file_path}.")
    else:
        print(f"{file_path} already exists.")


# Example usage
file_name = "fruit_model.h5"  # Replace with your desired file name
google_drive_url = "https://drive.google.com/uc?id=16tQdzd_hpU3veCnQSzSo5xFeTGZ7qcvs"  # Replace with your file's Google Drive URL
check_and_download_file(file_name, google_drive_url)

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables
MODEL_PATH = "fruit_model.h5"
img_size = (224, 224)
model = None

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU memory growth enabled")
    except RuntimeError as e:
        logger.error(f"Error configuring GPU: {e}")

# Load model
try:
    logger.info("Loading model...")
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

class_names = [
    "Apple_Bad", "Apple_Good", "Apple_mixed",
    "Banana_Bad", "Banana_Good", "Banana_mixed",
    "Guava_Bad", "Guava_Good", "Guava_mixed",
    "Lemon_mixed",
    "Lime_Bad", "Lime_Good",
    "Orange_Bad", "Orange_Good", "Orange_mixed",
    "Pomegranate_Bad", "Pomegranate_Good", "Pomegranate_mixed"
]

shelf_life = {
    "Apple": {"shelf": "1-2 days", "refrigerator": "3 weeks", "freezer": "8 months"},
    "Banana": {"shelf": "Until ripe", "refrigerator": "2 days (skin will blacken)",
               "freezer": "1 month (whole peeled)"},
    "Guava": {"shelf": "3-5 days", "refrigerator": "1 week", "freezer": "Do not freeze"},
    "Lemon": {"shelf": "10 days", "refrigerator": "1-2 weeks", "freezer": "Do not freeze"},
    "Lime": {"shelf": "10 days", "refrigerator": "1-2 weeks", "freezer": "Do not freeze"},
    "Orange": {"shelf": "10 days", "refrigerator": "1-2 weeks", "freezer": "Do not freeze"},
    "Pomegranate": {"shelf": "1-2 days", "refrigerator": "3-4 days", "freezer": "Balls, 1 month"}
}


@lru_cache(maxsize=32)
def preprocess_for_model(image_bytes):
    try:
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        image = cv2.resize(image, img_size)
        image = preprocess_input(image)
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logger.error(f"Error in preprocess_for_model: {e}")
        raise


@tf.function
def predict(input_tensor):
    return model(input_tensor, training=False)


def get_shelf_life_info(fruit_class):
    try:
        fruit_type = fruit_class.split('_')[0]
        condition = fruit_class.split('_')[1]

        if fruit_type in shelf_life:
            info = shelf_life[fruit_type]
            if condition == "Bad":
                shelf_days = min(0, int(info["refrigerator"].split()[0]))
            elif condition == "mixed":
                shelf_days = max(1, int(int(info["refrigerator"].split()[0]) * 0.7))
            else:  # "Good"
                shelf_days = int(info["refrigerator"].split()[0])

            return {
                "shelf": info["shelf"],
                "refrigerator": f"{shelf_days} days",
                "freezer": info["freezer"],
                "estimated_days": shelf_days
            }
    except Exception as e:
        logger.error(f"Error in get_shelf_life_info: {e}")

    return {
        "shelf": "Inedible",
        "refrigerator": "Inedible",
        "freezer": "Inedible",
        "estimated_days": 0
    }


def process_image(image_bytes):
    try:
        preprocessed_img = preprocess_for_model(image_bytes)
        predictions = predict(preprocessed_img)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        shelf_life_info = get_shelf_life_info(predicted_class)
        expiry_date = datetime.now() + timedelta(days=shelf_life_info["estimated_days"])

        return {
            "fruit_class": predicted_class,
            "confidence": confidence,
            "shelf_life": shelf_life_info,
            "expiry_date": expiry_date.strftime('%Y-%m-%d')
        }
    except Exception as e:
        logger.error(f"Error in process_image: {e}")
        raise


@app.route('/test', methods=['POST'])
def test_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        return jsonify({
            "message": "Image received successfully",
            "filename": file.filename,
            "model_loaded": model is not None
        })
    except Exception as e:
        logger.error(f"Error in test_endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():

    return jsonify({
        "message": "Server running successfully",
         })



@app.route('/detect_fruit', methods=['POST'])
def detect_fruit():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # if 'image' not in request.files:
        #     return jsonify({"error": "No image file provided"}), 400

        # file = request.files['image']
        formdata = request.form  # This gets the form data from the incoming request
        image_url = formdata.get('image_url')  # Extract image_url from formdata
        img_response = requests.get(image_url, timeout=30)
        if img_response.status_code != 200:
            return {
                "status": "failure",
                "status_code": img_response.status_code,
                "error": img_response.text
            }
        image_bytes = img_response.content

        # You don't need to check for `file.filename` because you're not dealing with a file object
        # Just ensure that `image_bytes` has content
        if not image_bytes:
            return jsonify({"error": "No image data received."}), 400

        # Process the image bytes
        result = process_image(image_bytes)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in detect_fruit: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(threaded=True)