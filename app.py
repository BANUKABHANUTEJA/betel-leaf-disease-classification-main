import os
import io
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# Configuration
IMG_SIZE = 224
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "resnet50_betel_gabor.h5")

# Load model globally
print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Class names corresponding to the output neurons
CLASS_NAMES = [
    "Anthracnose",
    "Healthy_Leaf",
    "Leaf_Rot",
    "Leaf_Spot",
    "bacterial_leaf"
]

# Disease treatments dictionary
TREATMENTS = {
    "Anthracnose": {
        "description": "Fungal disease causing dark, sunken lesions on leaves.",
        "treatment": "Remove and destroy infected leaves. Ensure good air circulation and avoid overhead watering. Apply copper-based or systemic fungicides like Carbendazim or Mancozeb as recommended."
    },
    "Healthy_Leaf": {
        "description": "The leaf appears healthy and free of known diseases.",
        "treatment": "Maintain current care routine: balanced watering, adequate sunlight, and regular monitoring for early signs of pests or disease."
    },
    "Leaf_Rot": {
        "description": "A severe fungal/bacterial condition causing rapid decay and rotting of leaf tissue, often favored by excessive moisture.",
        "treatment": "Immediately prune rotting plant parts. Improve drainage and reduce humidity/watering. Apply appropriate bactericides or fungicides based on severity."
    },
    "Leaf_Spot": {
        "description": "Characterized by brown or black spots on the leaves, often leading to yellowing and leaf drop.",
        "treatment": "Pick off affected leaves. Avoid wetting the foliage when watering. Neem oil or a broad-spectrum fungicide can help control the spread."
    },
    "bacterial_leaf": {
        "description": "Bacterial infection characterized by water-soaked, angular spots sometimes surrounded by a yellow halo.",
        "treatment": "Isolate the plant to prevent spread. Prune diseased areas using sterilized tools. Copper-based bactericides can offer some control if caught early."
    }
}

def gabor_enhance(rgb_img):
    rgb = rgb_img.astype(np.float32)
    gray = cv2.cvtColor(rgb_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gabor_sum = np.zeros_like(gray, dtype=np.float32)
    for theta in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        kernel = cv2.getGaborKernel((21, 21), 5.0, theta, 10.0, 0.5, 0)
        gabor_sum += cv2.filter2D(gray, cv2.CV_32F, kernel)
    gabor_norm = cv2.normalize(gabor_sum, None, 0, 255, cv2.NORM_MINMAX)
    gabor_3ch = np.stack([gabor_norm] * 3, axis=-1)
    combined = np.clip(0.7 * rgb + 0.3 * gabor_3ch, 0, 255)
    return combined.astype(np.float32)

def prepare_tensor(rgb_img):
    resized = cv2.resize(rgb_img, (IMG_SIZE, IMG_SIZE))
    enhanced = gabor_enhance(resized)
    processed = preprocess_input(enhanced)
    return np.expand_dims(processed, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load on the server."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No image file provided."}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Read image to memory
        in_memory_file = io.BytesIO()
        file.save(in_memory_file)
        data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
        
        # Decode and convert to RGB
        bgr_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr_image is None:
            return jsonify({"error": "Could not decode the image. Please upload a valid image file."}), 400
            
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        
        # Prepare for prediction
        tensor = prepare_tensor(rgb_image)
        
        # Predict
        probs = model.predict(tensor, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        
        # Ensure we don't go out of bounds if model outputs differently
        if pred_idx < len(CLASS_NAMES):
            pred_class = CLASS_NAMES[pred_idx]
        else:
            pred_class = f"Unknown Class {pred_idx}"
            
        confidence = float(probs[pred_idx]) * 100.0
        
        treatment_info = TREATMENTS.get(pred_class, {
            "description": "Information not available.",
            "treatment": "Consult an agricultural expert."
        })

        # Prepare probabilities dictionary
        probs_dict = {
            CLASS_NAMES[i].replace("_", " ").title(): float(probs[i])
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            "success": True,
            "prediction": pred_class.replace("_", " ").title(),
            "confidence": f"{confidence:.2f}%",
            "description": treatment_info["description"],
            "treatment": treatment_info["treatment"],
            "all_predictions": probs_dict
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure templates and static folders exist
    os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static", "css"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static", "js"), exist_ok=True)
    
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
