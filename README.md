# Betel Leaf Disease Classification

This project is a Flask-based web application for classifying betel leaf diseases using a pre-trained ResNet50 deep learning model. 

## Features
- **Disease Classification:** Identifies conditions like Anthracnose, Leaf Rot, Leaf Spot, Bacterial Leaf, and accurately detects Healthy Leaves.
- **Treatment Suggestions:** Provides detailed symptom descriptions and practical treatment methods for the identified disease.
- **Image Enhancement:** Uses Gabor filters combined with RGB preprocessing to enhance image features before feeding them into the model for higher accuracy.

## Project Structure
- `app.py`: The main Flask server application handling routes and model inference.
- `resnet50_betel_gabor.h5`: The pre-trained Keras model (ResNet50).
- `templates/`: HTML templates for the web interface.
- `static/`: CSS and JavaScript files for frontend styling and logic.
- `notebook1d62d08d60.ipynb`: Jupyter notebook containing model training and evaluation code.
- `betel_dataset/`: Image dataset of betel leaves used for training the model.

## Prerequisites
Ensure you have Python installed, then install the required dependencies:
```bash
pip install flask werkzeug tensorflow opencv-python-headless numpy
```

## How to Run Locally
1. Clone the repository natively or download the project files.
2. Navigate to the project directory.
3. Start the Flask server:
   ```bash
   python app.py
   ```
4. Open your web browser and navigate to `http://localhost:5000`. Upload an image of a betel leaf to get disease predictions and treatment recommendations.
