---
title: Pneumonia Detection AI
emoji: 🫁
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# 🫁 Pneumonia Detection AI

AI-powered pneumonia detection from chest X-ray images using deep learning.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

## 🚀 Live Demo

**Streamlit Cloud**: [Launch App](https://your-app.streamlit.app)
- **Confidence Scores** - Get probability scores for each prediction
- **REST API** - Easy integration with any application

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API information |
| `/health` | GET | Health check |
| `/predict` | POST | Single image prediction |
| `/predict/batch` | POST | Batch prediction |
| `/model/info` | GET | Model information |
| `/docs` | GET | Swagger documentation |

## Usage

### Web Interface
Simply access the Space URL to use the built-in frontend.

### API Call
```bash
curl -X POST https://YOUR_SPACE.hf.space/predict \
  -F "file=@chest_xray.jpg"
```

### Response Example
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 0.9542,
  "probabilities": {
    "NORMAL": 0.0458,
    "PNEUMONIA": 0.9542
  }
}
```

## Model

- **Architecture**: EfficientNet (Transfer Learning)
- **Input Size**: 224x224 RGB images
- **Classes**: NORMAL, PNEUMONIA
- **Framework**: TensorFlow/Keras

## Disclaimer

⚠️ This is a demonstration project for educational purposes. It should NOT be used for actual medical diagnosis. Always consult qualified healthcare professionals.
