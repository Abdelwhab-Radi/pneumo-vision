# Pneumonia Detection - Model Evaluation and Production API Guide

This guide explains how to:
1. ✅ Check your model's accuracy
2. 🚀 Deploy your model in a production API
3. 📸 Use the API to make predictions on X-ray images

---

## 🎁 NEW: Automatic Deployment Preparation

**The training script now automatically prepares your model for deployment!**

When you run `python model.py`, it will:
1. ✅ Train your model (initial + fine-tuning stages)
2. ✅ Evaluate performance on test data  
3. ✅ **Automatically create deployment-ready files**:
   - `.keras` model - For standard TensorFlow deployment
   - `.tflite` model - For mobile/edge deployment (optimized & quantized)
   - `deployment_manifest.json` - Model metadata and performance metrics

All deployment files are saved to `./results/models/`

### Creating a Complete Deployment Package

After training completes, create a packaged deployment bundle:

```bash
python deploy_package.py
```

This creates a `deployment/` folder with everything needed:
- All model files (.keras and .tflite)
- Configuration files
- API server code
- Documentation
- Ready-to-deploy structure

### Configuration

The automatic optimization can be controlled in your config file:

```json
{
  "auto_optimize_for_deployment": true,
  "save_deployment_package": true,
  "deployment_formats": ["keras", "tflite"],
  "tflite_quantize": true
}
```

---

### Quick Start

Run the evaluation script to see your model's accuracy and performance metrics:

```powershell
python evaluate_model.py
```

### What You'll Get

The evaluation script will show you:

1. **Overall Performance Metrics**
   - **Accuracy**: Percentage of correct predictions (e.g., 95%)
   - **Precision**: How accurate positive predictions are
   - **Recall**: How many actual positive cases were detected
   - **F1-Score**: Balance between precision and recall
   - **ROC AUC**: Overall model quality (higher is better, max = 1.0)

2. **Medical Metrics** (Important for healthcare!)
   - **Sensitivity**: How good the model is at detecting pneumonia (True Positive Rate)
   - **Specificity**: How good the model is at identifying normal cases (True Negative Rate)
   - **PPV**: If model says "pneumonia", how likely is it correct?
   - **NPV**: If model says "normal", how likely is it correct?

3. **Visualizations**
   - Confusion Matrix (shows correct vs incorrect predictions)
   - ROC Curve (shows model's discrimination ability)

4. **Saved Results**
   All results are saved to `a:/project/results/`:
   - `evaluation_results.json` - All metrics in JSON format
   - `evaluation_confusion_matrix.png` - Visual confusion matrix
   - `evaluation_roc_curve.png` - ROC curve plot

### Expected Output Example

```
EVALUATION METRICS
============================================================

📊 Overall Performance:
   Accuracy:  0.9423 (94.23%)
   Precision: 0.9581
   Recall:    0.9615
   F1-Score:  0.9598
   ROC AUC:   0.9845

🏥 Medical Metrics:
   Sensitivity (True Positive Rate): 0.9615 (96.15%)
   Specificity (True Negative Rate): 0.8974 (89.74%)
   PPV (Positive Predictive Value):  0.9581
   NPV (Negative Predictive Value):  0.9024

📈 Confusion Matrix:
   True Positives:  375
   True Negatives:  210
   False Positives: 24
   False Negatives: 15

💡 Interpretation:
   ✅ Excellent accuracy!
   ✅ High sensitivity - good at detecting positive cases
   ✅ High specificity - good at identifying negative cases
```

### Understanding the Metrics

**What is good accuracy?**
- **90-95%**: Very good! Your model is production-ready
- **85-90%**: Good, but could be improved
- **< 85%**: Needs more training or data

**For Medical Applications:**
- **High Sensitivity (>90%)**: Important! You don't want to miss pneumonia cases
- **High Specificity (>90%)**: Important! You don't want false alarms
- **Both high**: Excellent! Your model is reliable

---

## 🚀 Part 2: Production API Setup

### Step 1: Install Dependencies

First, install the required packages:

```powershell
pip install -r requirements.txt
```

This will install:
- FastAPI (web framework)
- Uvicorn (web server)
- Pillow (image processing)
- And other dependencies

### Step 2: Start the API Server

Run the API server:

```powershell
python api.py
```

You should see:

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Loading model from: a:/project/results/models/model_stage1_frozen.keras
INFO:     ✓ Model loaded successfully
INFO:     Warming up model...
INFO:     ✓ Model warmed up
INFO:     ✓ API ready to serve predictions
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

The API is now running! 🎉

### Step 3: Access the API

**Interactive API Documentation:**
Open your browser and go to:
- http://localhost:8000/docs (Swagger UI - Interactive testing)
- http://localhost:8000/redoc (ReDoc - Clean documentation)
- http://localhost:8000 (API information)

---

## 📸 Part 3: Using the API to Make Predictions

### Method 1: Using Python Client (Easiest)

Use the provided test script:

```powershell
python test_api.py
```

Or use the client programmatically:

```python
from test_api import PneumoniaAPIClient

# Initialize client
client = PneumoniaAPIClient()

# Check if API is running
health = client.health_check()
print(health)

# Predict on a single image
result = client.predict_single("path/to/chest_xray.jpg")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")

# Example output:
# Prediction: PNEUMONIA
# Confidence: 94.23%
# Probabilities: {'NORMAL': 0.0577, 'PNEUMONIA': 0.9423}
```

### Method 2: Using the Web Interface (Browser)

1. Go to http://localhost:8000/docs
2. Click on **POST /predict**
3. Click "Try it out"
4. Click "Choose File" and select an X-ray image
5. Click "Execute"
6. See the prediction result below!

### Method 3: Using cURL (Command Line)

```powershell
curl -X POST "http://localhost:8000/predict" `
  -H "accept: application/json" `
  -H "Content-Type: multipart/form-data" `
  -F "file=@path/to/chest_xray.jpg"
```

### Method 4: Using Python requests

```python
import requests

# Single prediction
with open("path/to/chest_xray.jpg", "rb") as f:
    files = {"file": ("xray.jpg", f, "image/jpeg")}
    response = requests.post("http://localhost:8000/predict", files=files)
    result = response.json()
    
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
print(f"Probabilities: {result['probabilities']}")
```

### Method 5: Batch Predictions (Multiple Images)

```python
from test_api import PneumoniaAPIClient

client = PneumoniaAPIClient()

# Predict on multiple images at once
results = client.predict_batch([
    "path/to/image1.jpg",
    "path/to/image2.jpg",
    "path/to/image3.jpg"
])

for i, pred in enumerate(results['predictions']):
    print(f"Image {i+1}: {pred['prediction']} ({pred['confidence']:.2%})")
```

---

## 📋 API Endpoints Reference

### GET /
- **Description**: API information and available endpoints
- **Response**: JSON with API details

### GET /health
- **Description**: Check if API is running
- **Response**: `{"status": "healthy", "model_loaded": true}`

### POST /predict
- **Description**: Predict pneumonia from a single X-ray image
- **Input**: Image file (JPEG, PNG, etc.)
- **Response**:
  ```json
  {
    "prediction": "PNEUMONIA",
    "confidence": 0.9423,
    "probabilities": {
      "NORMAL": 0.0577,
      "PNEUMONIA": 0.9423
    },
    "timestamp": "2025-12-09T06:20:26"
  }
  ```

### POST /predict/batch
- **Description**: Predict pneumonia from multiple images
- **Input**: List of image files (max 10)
- **Response**: Array of prediction results

### GET /model/info
- **Description**: Get information about the loaded model
- **Response**: Model configuration and metadata

---

## 🔧 Customization

### Change Model or Config Paths

Edit these lines in `api.py`:

```python
# In startup_event() function
model_path = "a:/project/results/models/model_stage1_frozen.keras"
config_path = "a:/project/results/training_config.json"
```

### Change API Port

Edit this line in `api.py`:

```python
# At the bottom of api.py
uvicorn.run(
    app,
    host="0.0.0.0",
    port=8000,  # Change this to your desired port
    log_level="info"
)
```

### Enable HTTPS (Production)

For production, deploy with a proper HTTPS setup:

```powershell
uvicorn api:app --host 0.0.0.0 --port 443 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

---

## 🚀 Deployment Options

### Option 1: Local Server (What we're doing now)
- Run: `python api.py`
- Access: `http://localhost:8000`
- Best for: Development and testing

### Option 2: Deploy to Cloud (Production)

**Heroku:**
```powershell
# Create Procfile
echo "web: uvicorn api:app --host 0.0.0.0 --port $PORT" > Procfile

# Deploy
heroku create your-app-name
git push heroku main
```

**AWS EC2:**
```bash
# On EC2 instance
pip install -r requirements.txt
nohup python api.py &
```

**Docker:**
```dockerfile
FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "api.py"]
```

### Option 3: Serverless (AWS Lambda, Google Cloud Functions)
Use the model with serverless frameworks for auto-scaling

---

## 🧪 Testing Your API

### Basic Health Check
```python
import requests
response = requests.get("http://localhost:8000/health")
print(response.json())
```

### Test with Sample Images
```python
from test_api import PneumoniaAPIClient

client = PneumoniaAPIClient()

# Test with normal X-ray
result = client.predict_single("test_images/normal.jpg")
assert result['prediction'] == 'NORMAL'

# Test with pneumonia X-ray
result = client.predict_single("test_images/pneumonia.jpg")
assert result['prediction'] == 'PNEUMONIA'

print("✅ All tests passed!")
```

---

## 📝 Example Integration

### Integrate into Web App (HTML + JavaScript)

```html
<!DOCTYPE html>
<html>
<body>
    <h1>Pneumonia Detector</h1>
    <input type="file" id="imageInput" accept="image/*">
    <button onclick="predict()">Detect Pneumonia</button>
    <div id="result"></div>

    <script>
        async function predict() {
            const input = document.getElementById('imageInput');
            const formData = new FormData();
            formData.append('file', input.files[0]);

            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            document.getElementById('result').innerHTML = `
                <h2>Result: ${result.prediction}</h2>
                <p>Confidence: ${(result.confidence * 100).toFixed(2)}%</p>
            `;
        }
    </script>
</body>
</html>
```

---

## ❓ Troubleshooting

### API won't start
- **Problem**: `Model not found`
- **Solution**: Check the model path in `api.py` startup_event()

### Low accuracy
- **Problem**: Model accuracy < 85%
- **Solutions**: 
  - Train for more epochs
  - Use more data
  - Try different model variants (B3, B4 instead of B0)

### API is slow
- **Problem**: Predictions take too long
- **Solutions**:
  - Run on GPU (make sure TensorFlow GPU is installed)
  - Use model quantization (convert to TFLite)
  - Enable batch predictions

### CORS errors in browser
- **Problem**: Can't access API from web app
- **Solution**: The API already has CORS enabled. Check browser console for details.

---

## 📚 Summary

**To check model accuracy:**
```powershell
python evaluate_model.py
```

**To start production API:**
```powershell
python api.py
```

**To test the API:**
```powershell
python test_api.py
```

**To make predictions:**
1. Upload image to http://localhost:8000/docs
2. Or use Python client:
   ```python
   from test_api import PneumoniaAPIClient
   client = PneumoniaAPIClient()
   result = client.predict_single("xray.jpg")
   print(result)
   ```

---

## 🎯 Next Steps

1. ✅ Run `python evaluate_model.py` to see your model's accuracy
2. ✅ Run `python api.py` to start the API server
3. ✅ Test predictions at http://localhost:8000/docs
4. ✅ Integrate the API into your application
5. 🚀 Deploy to production (cloud server, Docker, etc.)

**Need help?** Check the API documentation at http://localhost:8000/docs when the server is running!
