// Global variables
let uploadedImage = null;
let uploadedFile = null;

// API endpoint
const API_URL = 'http://localhost:5000';

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

// Check backend health
async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        console.log('Backend status:', data);
        if (!data.model_loaded) {
            console.warn('Model not loaded on backend yet');
        }
    } catch (error) {
        console.error('Cannot connect to backend:', error);
        alert('Cannot connect to backend server. Make sure Flask app is running on port 5000.');
    }
}

// Initialize the application
window.addEventListener('DOMContentLoaded', () => {
    checkBackendHealth();
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
    
    uploadedFile = file;
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
    uploadedFile = null;
    uploadedImage = null;
    previewImage.src = '';
    uploadArea.style.display = 'block';
    previewSection.style.display = 'none';
    predictBtn.disabled = true;
    resultSection.style.display = 'none';
    imageInput.value = '';
}
async function predictDisease() {
    if (!uploadedFile) {
        alert('Please upload an image first');
        return;
    }
    
    // Show loading
    loading.style.display = 'block';
    resultSection.style.display = 'none';
    predictBtn.disabled = true;
    
    try {
        // Create FormData to send image to backend
        const formData = new FormData();
        formData.append('image', uploadedFile);
        
        console.log('Sending request to:', `${API_URL}/predict`);
        
        // Send to backend API
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });
        
        console.log('Response status:', response.status);
        const responseText = await response.text();
        console.log('Response text:', responseText);
        
        let result;
        try {
            result = JSON.parse(responseText);
        } catch (e) {
            console.error('Failed to parse response as JSON:', e);
            alert(`Server error: ${responseText}`);
            return;
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}, message: ${result.error || 'Unknown error'}`);
        }
        
        if (result.error) {
            alert(`Backend error: ${result.error}`);
            return;
        }
        
        console.log('Prediction result:', result);
        
        // Display results from backend
        displayResults(result);
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert(`Error making prediction: ${error.message}`);
    } finally {
        loading.style.display = 'none';
        predictBtn.disabled = false;
    }
}

function displayResults(result) {
    // Display main prediction
    const confidence = (result.confidence * 100).toFixed(2);
    
    diseaseName.textContent = formatClassName(result.predicted_class);
    confidenceFill.style.width = `${confidence}%`;
    confidenceText.textContent = `${confidence}%`;
    
    // Display top 3 predictions
    topPredictions.innerHTML = '<h3>Other Possibilities:</h3>';
    result.top_predictions.slice(1).forEach(pred => {
        const item = document.createElement('div');
        item.className = 'prediction-item';
        
        const name = document.createElement('span');
        name.textContent = formatClassName(pred.class);
        
        const prob = document.createElement('span');
        prob.textContent = `${(pred.confidence * 100).toFixed(2)}%`;
        
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