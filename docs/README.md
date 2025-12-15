# 🫁 Pneumonia Detection from Chest X-Ray Images

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.20](https://img.shields.io/badge/tensorflow-2.20-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

A deep learning-powered REST API for detecting **pneumonia** from chest X-ray images using **EfficientNet** architecture. Upload a chest X-ray image and receive an instant diagnosis with confidence scores.

---

## ✨ Features

- 🔬 **High Accuracy**: EfficientNet-based model trained on chest X-ray dataset
- ⚡ **Fast Inference**: GPU-accelerated predictions in milliseconds
- 🌐 **REST API**: Clean, documented endpoints with FastAPI
- 🐳 **Docker Ready**: One-command deployment with Docker Compose
- 📊 **Confidence Scores**: Get probability scores for diagnosis reliability
- 📦 **Batch Processing**: Analyze multiple images in a single request
- 📖 **Interactive Docs**: Built-in Swagger UI and ReDoc documentation

---

## 🚀 Quick Start

### Option 1: Run Locally (3 Steps)

```bash
# 1. Clone and enter the project
cd /path/to/project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API
python api.py
```

**That's it!** Access the API at `http://localhost:8000`

### Option 2: Run with Docker

```bash
# Build and run
docker compose up -d

# Check status
docker compose logs -f
```

---

## 📡 API Usage

### Single Image Prediction

```bash
# Using curl
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"
```

**Response:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9234,
  "probabilities": {
    "NORMAL": 0.0766,
    "PNEUMONIA": 0.9234
  },
  "timestamp": "2024-12-09T12:00:00.000000"
}
```

### Python Client

```python
import requests

# Single prediction
with open("chest_xray.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    
result = response.json()
print(f"Diagnosis: {result['prediction']}")
print(f"Confidence: {result['confidence']:.1%}")
```

### Batch Prediction

```python
# Multiple images at once
files = [
    ("files", open("xray1.jpg", "rb")),
    ("files", open("xray2.jpg", "rb")),
    ("files", open("xray3.jpg", "rb")),
]
response = requests.post(
    "http://localhost:8000/predict/batch",
    files=files
)
results = response.json()
```

---

## 📖 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI documentation |
| `GET` | `/model/info` | Model configuration |
| `POST` | `/predict` | Single image prediction |
| `POST` | `/predict/batch` | Batch prediction (max 10 images) |

---

## ⚙️ Configuration

Configuration is managed via environment variables. Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./results/models/model_stage1_frozen.keras` | Path to trained model |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |

---

## 📊 Model Performance

The model was trained on the [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) with the following performance:

| Metric | Value |
|--------|-------|
| Accuracy | ~95% |
| AUC | ~0.98 |
| Sensitivity | ~96% |
| Specificity | ~93% |

*Note: Run `python evaluate_model.py` for detailed metrics on your trained model.*

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -t pneumonia-api .

# Run the container
docker run -d -p 8000:8000 --name pneumonia-api pneumonia-api

# View logs
docker logs -f pneumonia-api
```

### Docker Compose (Recommended)

```bash
# Start
docker compose up -d

# Stop
docker compose down

# View logs
docker compose logs -f
```

### GPU Support (Optional)

Uncomment the GPU section in `docker-compose.yml` for NVIDIA GPU acceleration.

---

## 📁 Project Structure

```
project/
├── api.py                 # FastAPI application
├── config.py              # Configuration management
├── model.py               # Training pipeline
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container image
├── docker-compose.yml     # Container orchestration
├── .env.example           # Environment template
│
├── results/
│   └── models/            # Trained model files
│       └── model_stage1_frozen.keras
│
├── test_api.py            # API test client
├── evaluate_model.py      # Model evaluation
└── verify_gpu.py          # GPU verification
```

---

## 🔧 Development

### Training a New Model

```bash
# Configure training parameters
nano results/training_config.json

# Run training
python model.py
```

### Testing the API

```bash
# Start the API in background
python api.py &

# Run tests
python test_api.py
```

---

## ⚠️ Medical Disclaimer

> **This tool is for research and educational purposes only.** It is NOT intended for clinical diagnosis. Always consult a qualified healthcare professional for medical advice.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Model: EfficientNet architecture from TensorFlow
- Framework: FastAPI for the REST API

---

## 📞 Support

For issues or questions:
1. Check existing documentation in `/docs`
2. Run diagnostic scripts (`diagnose_gpu.py`, `verify_gpu.py`)
3. Open an issue with error logs
