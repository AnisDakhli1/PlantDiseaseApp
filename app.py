from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import os

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Device: GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Path to your saved model checkpoint
MODEL_PATH = 'model/plant_disease_model.pth'  # update path if needed

# List of class names
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy',
    # add all classes in same order as training
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Global model variable
model = None

# Define the model architecture (match your training)
class PlantModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=None)  # replace with your architecture if different
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def load_model():
    """Load the checkpoint into the model architecture"""
    global model
    try:
        model = PlantModel(num_classes=len(CLASS_NAMES))
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        print(f"Model loaded successfully from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")

def preprocess_image(image_bytes):
    """Convert uploaded image bytes into a torch tensor"""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_tensor = transform(image)
    return img_tensor.unsqueeze(0)  # add batch dimension

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        image_file = request.files['image']
        image_bytes = image_file.read()
        processed_image = preprocess_image(image_bytes).to(device)

        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)

        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class_idx].item()

        top_3_probs, top_3_idx = torch.topk(probabilities, 3)
        top_3_predictions = [
            {'class': CLASS_NAMES[idx], 'confidence': top_3_probs[0][i].item()}
            for i, idx in enumerate(top_3_idx[0])
        ]

        return jsonify({
            'success': True,
            'predicted_class': CLASS_NAMES[predicted_class_idx],
            'confidence': confidence,
            'top_predictions': top_3_predictions
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    load_model()
    # Local development
    app.run(debug=True, host='0.0.0.0', port=5000)
    # Production: use gunicorn (Linux)
    # gunicorn -w 4 -b 0.0.0.0:5000 app:app