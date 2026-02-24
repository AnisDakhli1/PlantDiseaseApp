// Global variables
let uploadedImage = null;
let uploadedFile = null;

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

// API endpoint (change this based on your deployment)
const API_URL = '/predict';  // Local development
// const API_URL = 'https://your-domain.com/predict';  // Production

// Initialize the application
window.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkServerHealth();
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

async function checkServerHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        if (!data.model_loaded) {
            console.warn('Model is not loaded on the server');
        }
    } catch (error) {
        console.error('Cannot connect to server:', error);
    }
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
    uploadedImage = null;
    uploadedFile = null;
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
        // Create FormData to send the image
        const formData = new FormData();
        formData.append('image', uploadedFile);
        
        // Send request to backend
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error('Prediction request failed');
        }
        
        const result = await response.json();
        
        if (result.success) {
            // Display results
            displayResults(result);
        } else {
            throw new Error(result.error || 'Prediction failed');
        }
        
    } catch (error) {
        console.error('Prediction error:', error);
        alert('Error making prediction. Please ensure the server is running and try again.');
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
    
    // Skip the first one (already shown as main prediction)
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