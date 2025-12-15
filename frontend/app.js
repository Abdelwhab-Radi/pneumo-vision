/**
 * Pneumonia Detection AI - Frontend Application
 * Handles image upload, API communication, and result display
 */

// Configuration
const CONFIG = {
    API_BASE_URL: 'http://localhost:8000',
    ENDPOINTS: {
        HEALTH: '/health',
        PREDICT: '/predict',
        MODEL_INFO: '/model/info',
        ROOT: '/'
    },
    MAX_FILE_SIZE: 10 * 1024 * 1024, // 10MB
    ALLOWED_TYPES: ['image/jpeg', 'image/png', 'image/jpg', 'image/webp']
};

// DOM Elements
const elements = {
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    previewContainer: document.getElementById('previewContainer'),
    previewImage: document.getElementById('previewImage'),
    removeImage: document.getElementById('removeImage'),
    analyzeBtn: document.getElementById('analyzeBtn'),
    resultsSection: document.getElementById('resultsSection'),
    loadingState: document.getElementById('loadingState'),
    resultsContent: document.getElementById('resultsContent'),
    diagnosisBadge: document.getElementById('diagnosisBadge'),
    diagnosisText: document.getElementById('diagnosisText'),
    confidenceBar: document.getElementById('confidenceBar'),
    confidenceValue: document.getElementById('confidenceValue'),
    probabilityItems: document.getElementById('probabilityItems'),
    analysisTime: document.getElementById('analysisTime'),
    apiStatus: document.getElementById('apiStatus'),
    modelStatus: document.getElementById('modelStatus'),
    inputSize: document.getElementById('inputSize'),
    modelClasses: document.getElementById('modelClasses')
};

// State
let currentFile = null;

/**
 * Initialize the application
 */
async function init() {
    setupEventListeners();
    await checkApiStatus();
    await loadModelInfo();
}

/**
 * Set up all event listeners
 */
function setupEventListeners() {
    // File input change
    elements.fileInput.addEventListener('change', handleFileSelect);

    // Upload area click
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());

    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);

    // Remove image button
    elements.removeImage.addEventListener('click', handleRemoveImage);

    // Analyze button
    elements.analyzeBtn.addEventListener('click', handleAnalyze);
}

/**
 * Check API health status
 */
async function checkApiStatus() {
    const statusIndicator = elements.apiStatus;
    const statusText = statusIndicator.querySelector('.status-text');

    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.HEALTH}`);

        if (response.ok) {
            statusIndicator.classList.remove('offline');
            statusIndicator.classList.add('online');
            statusText.textContent = 'API Online';
            elements.modelStatus.textContent = 'Ready';
            elements.modelStatus.style.color = 'var(--color-success)';
            return true;
        } else {
            throw new Error('API not healthy');
        }
    } catch (error) {
        statusIndicator.classList.remove('online');
        statusIndicator.classList.add('offline');
        statusText.textContent = 'API Offline';
        elements.modelStatus.textContent = 'Unavailable';
        elements.modelStatus.style.color = 'var(--color-danger)';
        console.error('API health check failed:', error);
        return false;
    }
}

/**
 * Load model information from API
 */
async function loadModelInfo() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.MODEL_INFO}`);

        if (response.ok) {
            const data = await response.json();
            elements.inputSize.textContent = `${data.input_size}x${data.input_size}px`;
            elements.modelClasses.textContent = data.classes.join(', ');
        }
    } catch (error) {
        console.error('Failed to load model info:', error);
    }
}

/**
 * Handle file selection
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

/**
 * Handle drag over
 */
function handleDragOver(event) {
    event.preventDefault();
    event.stopPropagation();
    elements.uploadArea.classList.add('drag-over');
}

/**
 * Handle drag leave
 */
function handleDragLeave(event) {
    event.preventDefault();
    event.stopPropagation();
    elements.uploadArea.classList.remove('drag-over');
}

/**
 * Handle file drop
 */
function handleDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    elements.uploadArea.classList.remove('drag-over');

    const file = event.dataTransfer.files[0];
    if (file) {
        processFile(file);
    }
}

/**
 * Process the selected file
 */
function processFile(file) {
    // Validate file type
    if (!CONFIG.ALLOWED_TYPES.includes(file.type)) {
        showError('Please select a valid image file (JPEG, PNG, or WebP)');
        return;
    }

    // Validate file size
    if (file.size > CONFIG.MAX_FILE_SIZE) {
        showError('File size exceeds 10MB limit');
        return;
    }

    currentFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        elements.previewImage.src = e.target.result;
        elements.previewContainer.style.display = 'flex';
        elements.uploadArea.style.display = 'none';

        // Hide results if showing
        elements.resultsSection.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

/**
 * Handle remove image
 */
function handleRemoveImage(event) {
    event.stopPropagation();
    resetUpload();
}

/**
 * Reset the upload state
 */
function resetUpload() {
    currentFile = null;
    elements.fileInput.value = '';
    elements.previewContainer.style.display = 'none';
    elements.uploadArea.style.display = 'block';
    elements.resultsSection.style.display = 'none';
}

/**
 * Handle analyze button click
 */
async function handleAnalyze() {
    if (!currentFile) {
        showError('Please select an image first');
        return;
    }

    // Check API status first
    const isOnline = await checkApiStatus();
    if (!isOnline) {
        showError('API is not available. Please make sure the server is running.');
        return;
    }

    // Show results section with loading state
    elements.resultsSection.style.display = 'block';
    elements.loadingState.style.display = 'flex';
    elements.resultsContent.style.display = 'none';
    elements.analyzeBtn.disabled = true;

    const startTime = performance.now();

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', currentFile);

        // Send prediction request
        const response = await fetch(`${CONFIG.API_BASE_URL}${CONFIG.ENDPOINTS.PREDICT}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `HTTP ${response.status}`);
        }

        const result = await response.json();
        const endTime = performance.now();
        const duration = ((endTime - startTime) / 1000).toFixed(2);

        // Display results
        displayResults(result, duration);

    } catch (error) {
        console.error('Prediction error:', error);
        showError(`Analysis failed: ${error.message}`);
        elements.resultsSection.style.display = 'none';
    } finally {
        elements.analyzeBtn.disabled = false;
    }
}

/**
 * Display prediction results
 */
function displayResults(result, duration) {
    // Hide loading, show results
    elements.loadingState.style.display = 'none';
    elements.resultsContent.style.display = 'block';

    // Set analysis time
    elements.analysisTime.textContent = `Analyzed in ${duration}s`;

    // Set diagnosis
    const isPneumonia = result.prediction === 'PNEUMONIA';
    elements.diagnosisBadge.className = 'diagnosis-badge ' + (isPneumonia ? 'pneumonia' : 'normal');
    elements.diagnosisBadge.querySelector('.diagnosis-icon').textContent = isPneumonia ? '⚠️' : '✅';
    elements.diagnosisText.textContent = result.prediction;

    // Set confidence
    const confidencePercent = (result.confidence * 100).toFixed(1);
    elements.confidenceBar.style.width = `${confidencePercent}%`;
    elements.confidenceValue.textContent = `${confidencePercent}%`;

    // Set probability breakdown
    if (result.probabilities) {
        elements.probabilityItems.innerHTML = '';

        for (const [className, probability] of Object.entries(result.probabilities)) {
            const percent = (probability * 100).toFixed(1);
            const isNormal = className === 'NORMAL';

            const itemHtml = `
                <div class="probability-item">
                    <span class="probability-item-label">${className}</span>
                    <div class="probability-item-bar-container">
                        <div class="probability-item-bar ${isNormal ? 'normal' : 'pneumonia'}" 
                             style="width: ${percent}%"></div>
                    </div>
                    <span class="probability-item-value">${percent}%</span>
                </div>
            `;

            elements.probabilityItems.innerHTML += itemHtml;
        }
    }

    // Scroll to results
    elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Show error message
 */
function showError(message) {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = 'toast-error';
    toast.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"/>
            <line x1="15" y1="9" x2="9" y2="15"/>
            <line x1="9" y1="9" x2="15" y2="15"/>
        </svg>
        <span>${message}</span>
    `;

    // Add styles if not exists
    if (!document.querySelector('style[data-toast]')) {
        const style = document.createElement('style');
        style.setAttribute('data-toast', 'true');
        style.textContent = `
            .toast-error {
                position: fixed;
                bottom: 24px;
                right: 24px;
                display: flex;
                align-items: center;
                gap: 12px;
                padding: 16px 24px;
                background: rgba(239, 68, 68, 0.95);
                color: white;
                border-radius: 12px;
                font-size: 14px;
                font-weight: 500;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
                z-index: 9999;
                animation: toastIn 0.3s ease, toastOut 0.3s ease 3s forwards;
            }
            .toast-error svg {
                width: 20px;
                height: 20px;
            }
            @keyframes toastIn {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            @keyframes toastOut {
                from {
                    opacity: 1;
                    transform: translateY(0);
                }
                to {
                    opacity: 0;
                    transform: translateY(20px);
                }
            }
        `;
        document.head.appendChild(style);
    }

    document.body.appendChild(toast);

    // Remove after animation
    setTimeout(() => {
        toast.remove();
    }, 3500);
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', init);
