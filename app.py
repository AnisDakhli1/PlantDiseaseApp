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

# Device: Force GPU usage
device = torch.device('cuda')
print(f"Using device: {device}")
if not torch.cuda.is_available():
    print("WARNING: CUDA not available! Falling back to CPU")
    device = torch.device('cpu')
print(f"GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

# Path to your saved model checkpoint
MODEL_PATH = 'model/model_checkpoint.pth'  # update path if needed

# List of class names (38 classes)
CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy',
]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Global model variable
model = None

# Define the model architecture (match your training)
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        IMG_SIZE = 224

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

def load_model():
    """Load the checkpoint into the model architecture"""
    global model
    try:
        model = CNNModel(num_classes=len(CLASS_NAMES))
        model.to(device)  # Move model to device BEFORE loading state_dict
        
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)  # Ensure model is on device after loading
        model.eval()
        print(f"Model loaded successfully from {MODEL_PATH}")
        print(f"Model device: {next(model.parameters()).device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()

def preprocess_image(image_bytes):
    """Convert uploaded image bytes into a torch tensor"""
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_tensor = transform(image)
    return img_tensor.unsqueeze(0)  # add batch dimension

@app.route('/')
def index():
    return send_from_directory('.', 'templates/index.html')

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