"""
Production API for Pneumonia Detection
This FastAPI application provides REST endpoints for making predictions using the trained model.

Usage:
    python3 api.py                    # Run with default settings
    PORT=8080 python3 api.py          # Run on custom port
    
API Documentation:
    http://localhost:8000/docs       # Swagger UI
    http://localhost:8000/redoc      # ReDoc
"""

import os
import time
from contextlib import asynccontextmanager

# Configure TensorFlow before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU mode to avoid cuDNN issues

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf

# Configure GPU memory growth to avoid cuDNN issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU config error: {e}")

from tensorflow import keras
import numpy as np
from PIL import Image
import io
from pathlib import Path
from typing import Dict, List
import json
import logging
from datetime import datetime

# Import X-ray validator
try:
    from xray_validator import XrayValidator, get_validator
    XRAY_VALIDATION_AVAILABLE = True
except ImportError:
    XRAY_VALIDATION_AVAILABLE = False
    print("Warning: xray_validator not found. Image validation disabled.")

# Import configuration
try:
    from config import settings
except ImportError:
    # Fallback if config.py not found
    class settings:
        MODEL_PATH = os.getenv("MODEL_PATH", "./results/models/model_stage1_frozen.keras")
        CONFIG_PATH = os.getenv("CONFIG_PATH", "./results/training_config.json")
        HOST = os.getenv("HOST", "0.0.0.0")
        PORT = int(os.getenv("PORT", "8000"))
        LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
        API_TITLE = "Pneumonia Detection API"
        API_VERSION = "1.0.0"
        API_DESCRIPTION = "REST API for detecting pneumonia from chest X-ray images"

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PneumoniaDetector:
    """Pneumonia detection model wrapper for production"""
    
    # Confidence threshold for valid predictions
    PREDICTION_CONFIDENCE_THRESHOLD = 0.55
    
    def __init__(self, model_path: str, config_path: str = None):
        """
        Initialize the detector
        
        Args:
            model_path: Path to the trained Keras model
            config_path: Path to training configuration (optional)
        """
        self.model_path = model_path
        
        # Load configuration
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                'img_size': 224,
                'color_mode': 'rgb',
                'class_names': ['NORMAL', 'PNEUMONIA']
            }
        
        # Extract important config values
        self.img_size = self.config.get('img_size', 224)
        self.class_names = self.config.get('class_names', ['NORMAL', 'PNEUMONIA'])
        
        # Load model
        logger.info(f"Loading model from: {model_path}")
        try:
            self.model = keras.models.load_model(model_path)
            logger.info("✓ Model loaded successfully")
            
            # Auto-detect input size from model
            try:
                model_input_shape = self.model.input_shape
                if model_input_shape and len(model_input_shape) >= 3:
                    model_img_size = model_input_shape[1]  # (batch, height, width, channels)
                    if model_img_size and model_img_size != self.img_size:
                        logger.warning(f"Config img_size ({self.img_size}) differs from model input ({model_img_size}). Using model input size.")
                        self.img_size = model_img_size
            except Exception as e:
                logger.warning(f"Could not auto-detect model input size: {e}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Warm up the model
        self._warmup()
    
    def _warmup(self):
        """Warm up the model with a dummy prediction"""
        logger.info("Warming up model...")
        dummy_input = np.random.rand(1, self.img_size, self.img_size, 3).astype(np.float32)
        _ = self.model.predict(dummy_input, verbose=0)
        logger.info("✓ Model warmed up")
    
    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """
        Preprocess image for prediction
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Preprocessed image array
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((self.img_size, self.img_size))
            
            # Convert to array
            img_array = np.array(image, dtype=np.float32)
            
            # Add batch dimension
            img_array = np.expand_dims(img_array, axis=0)
            
            # EfficientNet preprocessing is already in the model
            # So we just need to normalize to [0, 1] or [0, 255] depending on model
            # The model handles preprocessing internally
            
            return img_array
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise ValueError(f"Invalid image: {e}")
    
    def predict(self, image_bytes: bytes, return_confidence: bool = True) -> Dict:
        """
        Make prediction on an image
        
        Args:
            image_bytes: Raw image bytes
            return_confidence: Whether to return confidence scores
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess image
            img_array = self.preprocess_image(image_bytes)
            
            # Make prediction
            prediction = self.model.predict(img_array, verbose=0)[0][0]
            
            # Convert to prediction
            predicted_class_idx = int(prediction >= 0.5)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(prediction if predicted_class_idx == 1 else 1 - prediction)
            
            # Prepare result
            result = {
                'prediction': predicted_class,
                'confidence': round(confidence, 4),
                'timestamp': datetime.now().isoformat()
            }
            
            if return_confidence:
                result['probabilities'] = {
                    self.class_names[0]: round(float(1 - prediction), 4),
                    self.class_names[1]: round(float(prediction), 4)
                }
            
            logger.info(f"Prediction: {predicted_class} (confidence: {confidence:.4f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def predict_batch(self, images_bytes: List[bytes]) -> List[Dict]:
        """
        Make predictions on multiple images
        
        Args:
            images_bytes: List of raw image bytes
            
        Returns:
            List of prediction results
        """
        results = []
        for img_bytes in images_bytes:
            try:
                result = self.predict(img_bytes)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results


# Global instances
detector = None
xray_validator = None
start_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    global detector, xray_validator, start_time
    start_time = datetime.now()
    
    # Use configuration-based paths
    model_path = settings.MODEL_PATH
    config_path = settings.CONFIG_PATH
    
    logger.info(f"Starting Pneumonia Detection API v{settings.API_VERSION}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Config path: {config_path}")
    
    try:
        detector = PneumoniaDetector(model_path, config_path)
        logger.info("✓ Pneumonia detector ready")
        
        # Initialize X-ray validator
        if XRAY_VALIDATION_AVAILABLE:
            xray_validator = get_validator()
            logger.info("✓ X-ray image validator ready")
        else:
            xray_validator = None
            logger.warning("⚠ X-ray validation disabled (module not found)")
        
        logger.info("✓ API ready to serve predictions")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        logger.error("Please ensure the model file exists or set MODEL_PATH environment variable")
        raise
    except Exception as e:
        logger.error(f"Failed to initialize detector: {e}")
        raise
    
    yield  # Application runs here
    
    # Cleanup (if needed)
    logger.info("Shutting down API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title=settings.API_TITLE,
    description=getattr(settings, 'API_DESCRIPTION', 'REST API for detecting pneumonia from chest X-ray images'),
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS if isinstance(settings.ALLOWED_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add response timing header for monitoring"""
    request_start = time.time()
    response = await call_next(request)
    process_time = time.time() - request_start
    response.headers["X-Process-Time"] = f"{process_time:.4f}"
    return response


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if detector is None or detector.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict pneumonia from a single chest X-ray image
    
    This endpoint now includes validation to ensure the uploaded image
    is a valid chest X-ray before making predictions.
    
    Args:
        file: Image file (JPEG, PNG, etc.) - must be a chest X-ray
        
    Returns:
        Prediction result with confidence scores, or validation error
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Expected image file."
        )
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Step 1 & 2: Validate image is a chest X-ray
        if xray_validator is not None:
            logger.info("Running X-ray validation...")
            validation_result = xray_validator.validate(image_bytes)
            logger.info(f"Validation result: is_valid={validation_result.is_valid}, confidence={validation_result.confidence}")
            logger.info(f"Characteristics: {validation_result.validation_details.get('characteristics', {})}")
            
            if not validation_result.is_valid:
                logger.warning(f"Image validation failed: {validation_result.message_en}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": "invalid_image",
                        "is_valid_xray": False,
                        "message": validation_result.message_en,
                        "message_ar": validation_result.message_ar,
                        "validation_confidence": validation_result.confidence,
                        "details": validation_result.validation_details
                    }
                )
        else:
            logger.warning("X-ray validator is None!")
        
        # Make prediction
        result = detector.predict(image_bytes)
        
        # Step 3: Check prediction confidence (Layer 3 validation)
        if xray_validator is not None:
            prediction_confidence = result.get('confidence', 0)
            is_confident, conf_msg_en, conf_msg_ar = xray_validator.validate_prediction_confidence(
                prediction_confidence,
                threshold=detector.PREDICTION_CONFIDENCE_THRESHOLD
            )
            
            if not is_confident:
                result['confidence_warning'] = True
                result['confidence_warning_message'] = conf_msg_en
                result['confidence_warning_message_ar'] = conf_msg_ar
        
        # Add validation status to response
        result['is_valid_xray'] = True
        
        return JSONResponse(content=result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict pneumonia from multiple chest X-ray images
    
    Args:
        files: List of image files
        
    Returns:
        List of prediction results
    """
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images allowed per batch request"
        )
    
    try:
        # Read all image bytes
        images_bytes = []
        for file in files:
            if not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type: {file.content_type}"
                )
            images_bytes.append(await file.read())
        
        # Make predictions
        results = detector.predict_batch(images_bytes)
        
        return JSONResponse(content={"predictions": results})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    return {
        "model_path": detector.model_path,
        "input_size": detector.img_size,
        "classes": detector.class_names,
        "config": detector.config
    }


if __name__ == "__main__":
    import uvicorn
    
    # Print startup information
    print("\n" + "="*60)
    print("  🫁 PNEUMONIA DETECTION API")
    print("="*60)
    print(f"  Server: http://{settings.HOST}:{settings.PORT}")
    print(f"  API Docs: http://localhost:{settings.PORT}/docs")
    print(f"  Model: {settings.MODEL_PATH}")
    print("="*60 + "\n")
    
    # Run the API
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower()
    )
