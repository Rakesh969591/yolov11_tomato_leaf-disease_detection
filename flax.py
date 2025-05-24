from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import base64
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import io

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load model
model = YOLO("models/tomato_disease_yolo11.pt")

# Disease descriptions dictionary (based on YOLO class names)
disease_info = {
    "Late Blight": "Caused by Phytophthora infestans, resulting in dark lesions on leaves.",
    "Early Blight": "Caused by Alternaria solani, leading to brown concentric rings on older leaves.",
    "Leaf Mold": "Fungal infection causing pale green or yellowish patches on the leaves.",
    "Bacterial Spot": "Bacterial disease characterized by dark spots with yellow halos.",
    "Healthy": "No signs of disease detected. Plant appears healthy."
    # Add more labels as per your dataset
}


def read_image_from_request(image_file):
    image = Image.open(image_file).convert("RGB")
    return np.array(image)


def encode_image_to_base64(image_np):
    _, buffer = cv2.imencode(".jpg", image_np)
    return base64.b64encode(buffer).decode("utf-8")


@app.route("/api/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image_np = read_image_from_request(image_file)

    # Run inference
    results = model(image_np)[0]

    # Draw bounding boxes and collect class names
    for box in results.boxes:
        cls_id = int(box.cls)
        class_name = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image_np, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Get first detected class for info (you can expand to multi-class if needed)
    detected_class = model.names[int(results.boxes[0].cls)] if results.boxes else "Healthy"
    info = disease_info.get(detected_class, "No information available.")

    # Encode final image to base64
    encoded_img = encode_image_to_base64(image_np)

    return jsonify({
        "resultImage": encoded_img,
        "diseaseInfo": f"Detected: {detected_class}. Description: {info}"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
