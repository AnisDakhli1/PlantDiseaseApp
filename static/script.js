// Global variables
let uploadedImage = null;
let model = null;

// Class names - UPDATE THESE based on your model's classes
const CLASS_NAMES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___healthy',
    // Add all your disease classes here
];

// DOM elements
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const predictBtn = document.getElementById('predictBtn');
const resultSection = document.getElementById('resultSection');
const loading = document.getElementById('loading');
const diseaseName = document.getElementById('diseaseName');
const confidenceFill = document.getElementById('confidenceFill');
const confidenceText = document.getElementById('confidenceText');
const topPredictions = document.getElementById('topPredictions');

// Load TensorFlow.js model
async function loadModel() {
    try {
        // For TensorFlow.js model (converted from Keras)
        model = await tf.loadLayersModel('model/model.json');
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading model:', error);
        alert('Failed to load model. Please ensure model files are in the correct location.');
    }
}

// Initialize the application
window.addEventListener('DOMContentLoaded', () => {
    loadModel();
    setupEventListeners();
});

function setupEventListeners() {
    // Click to upload
    uploadArea.addEventListener('click', () => imageInput.click());
    
    // File input change
    imageInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // Remove image
    removeBtn.addEventListener('click', removeImage);
    
    // Predict button
    predictBtn.addEventListener('click', predictDisease);
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    if (!file.type.match('image.*')) {
        alert('Please select an image file');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        uploadedImage = e.target.result;
        displayPreview(uploadedImage);
    };
    reader.readAsDataURL(file);
}

function displayPreview(imageSrc) {
    previewImage.src = imageSrc;
    uploadArea.style.display = 'none';
    previewSection.style.display = 'block';
    predictBtn.disabled = false;
    resultSection.style.display = 'none';
}

function removeImage() {
    uploadedImage = null;
    previewImage.src = '';
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
    predictBtn.disabled = true;
    resultSection.style.display = 'none';
    imageInput.value = '';
}

// Preprocess image to 224x224 and normalize
async function preprocessImage(imageSrc) {
    return new Promise((resolve) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.src = imageSrc;
        
        img.onload = () => {
            // Create a canvas to resize the image to 224x224
            const canvas = document.createElement('canvas');
            canvas.width = 224;
            canvas.height = 224;
            const ctx = canvas.getContext('2d');
            
            // Draw and resize image to 224x224
            ctx.drawImage(img, 0, 0, 224, 224);
            
            // Convert to tensor
            const tensor = tf.browser.fromPixels(canvas)
                .toFloat()
                .div(255.0) // Normalize to [0, 1]
                .expandDims(0); // Add batch dimension
            
            resolve(tensor);
        };
    });
}

async function predictDisease() {
    if (!model) {
        alert('Model is still loading. Please wait...');
        return;
    }
    
    if (!uploadedImage) {
        alert('Please upload an image first');
        return;
    }
    
    // Show loading
    loading.style.display = 'block';
    resultSection.style.display = 'none';
    predictBtn.disabled = true;
    
    try {
        // Preprocess the image
        const tensor = await preprocessImage(uploadedImage);
        
        // Make prediction
        const predictions = await model.predict(tensor);
        const predArray = await predictions.data();
        
        // Clean up tensor
        tensor.dispose();
        predictions.dispose();
        
        // Get top 3 predictions
        const topK = getTopK(predArray, 3);
        
        // Display results
        displayResults(topK);
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error making prediction. Please try again.');
    } finally {
        loading.style.display = 'none';
        predictBtn.disabled = false;
    }
}

function getTopK(predictions, k) {
    const indexed = predictions.map((prob, index) => ({ prob, index }));
    indexed.sort((a, b) => b.prob - a.prob);
    return indexed.slice(0, k);
}

function displayResults(topK) {
    // Display main prediction
    const topPrediction = topK[0];
    const className = CLASS_NAMES[topPrediction.index] || `Class ${topPrediction.index}`;
    const confidence = (topPrediction.prob * 100).toFixed(2);
    
    diseaseName.textContent = formatClassName(className);
    confidenceFill.style.width = `${confidence}%`;
    confidenceText.textContent = `${confidence}%`;
    
    // Display top 3 predictions
    topPredictions.innerHTML = '<h3>Other Possibilities:</h3>';
    topK.slice(1).forEach(pred => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        
        const name = document.createElement('span');
        name.textContent = formatClassName(CLASS_NAMES[pred.index] || `Class ${pred.index}`);
        
        const prob = document.createElement('span');
        prob.textContent = `${(pred.prob * 100).toFixed(2)}%`;
        
        item.appendChild(name);
        item.appendChild(prob);
        topPredictions.appendChild(item);
    });
    
    resultSection.style.display = 'block';
}

function formatClassName(className) {
    // Format class name for display (replace underscores with spaces)
    return className.replace(/_/g, ' ');
}