# Pneumonia Detection API Reference

## Base URL
```
http://localhost:8000
```

---

## Authentication
Currently no authentication required. For production, consider adding API keys or OAuth2.

---

## Endpoints

### GET /
**Description:** Get API information

**Response:**
```json
{
  "name": "Pneumonia Detection API",
  "version": "1.0.0",
  "status": "running",
  "endpoints": {
    "/": "API information",
    "/health": "Health check",
    "/predict": "Single image prediction (POST)",
    "/predict/batch": "Batch prediction (POST)",
    "/model/info": "Model information"
  }
}
```

---

### GET /health
**Description:** Health check for monitoring

**Response (200):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-12-09T12:00:00.000000"
}
```

**Response (503):** Model not loaded

---

### POST /predict
**Description:** Predict pneumonia from a single chest X-ray image

**Request:**
- Content-Type: `multipart/form-data`
- Body: `file` - Image file (JPEG, PNG, etc.)

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@chest_xray.jpg"
```

**Example (Python):**
```python
import requests

with open("chest_xray.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
result = response.json()
```

**Response (200):**
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

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | "NORMAL" or "PNEUMONIA" |
| `confidence` | float | Confidence score (0.0 - 1.0) |
| `probabilities` | object | Probability for each class |
| `timestamp` | string | ISO 8601 timestamp |

**Errors:**
- `400`: Invalid file type (not an image)
- `500`: Internal server error
- `503`: Model not initialized

---

### POST /predict/batch
**Description:** Predict pneumonia from multiple images (max 10)

**Request:**
- Content-Type: `multipart/form-data`
- Body: `files` - Multiple image files

**Example (curl):**
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "accept: application/json" \
  -F "files=@xray1.jpg" \
  -F "files=@xray2.jpg" \
  -F "files=@xray3.jpg"
```

**Response (200):**
```json
{
  "predictions": [
    {
      "prediction": "NORMAL",
      "confidence": 0.8765,
      "probabilities": {"NORMAL": 0.8765, "PNEUMONIA": 0.1235},
      "timestamp": "..."
    },
    {
      "prediction": "PNEUMONIA",
      "confidence": 0.9234,
      "probabilities": {"NORMAL": 0.0766, "PNEUMONIA": 0.9234},
      "timestamp": "..."
    }
  ]
}
```

**Limits:**
- Maximum 10 images per request

---

### GET /model/info
**Description:** Get information about the loaded model

**Response:**
```json
{
  "model_path": "./results/models/model_stage1_frozen.keras",
  "input_size": 256,
  "classes": ["NORMAL", "PNEUMONIA"],
  "config": {...}
}
```

---

## Error Responses

All errors follow this format:
```json
{
  "detail": "Error message describing the issue"
}
```

| Status Code | Description |
|-------------|-------------|
| `400` | Bad request (invalid input) |
| `500` | Internal server error |
| `503` | Service unavailable (model not loaded) |

---

## Response Headers

| Header | Description |
|--------|-------------|
| `X-Process-Time` | Request processing time in seconds |
| `Content-Type` | `application/json` |

---

## Rate Limits
No rate limits currently configured. For production, consider adding rate limiting middleware.

---

## Interactive Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
