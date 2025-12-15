"""
Production Configuration for Pneumonia Detection API
Environment-based configuration management
"""

import os
from pathlib import Path
from typing import List
from dataclasses import dataclass, field

# Base directory
BASE_DIR = Path(__file__).parent.resolve()


def _get_allowed_origins() -> List[str]:
    """Factory function to get allowed origins from environment"""
    return os.getenv("ALLOWED_ORIGINS", "*").split(",")


def _get_class_names() -> List[str]:
    """Factory function to get class names"""
    return ["NORMAL", "PNEUMONIA"]


@dataclass
class Settings:
    """Application settings loaded from environment variables"""
    
    # Model Configuration
    MODEL_PATH: str = field(default_factory=lambda: os.getenv(
        "MODEL_PATH", 
        str(BASE_DIR / "results" / "models" / "model_final.keras")
    ))
    CONFIG_PATH: str = field(default_factory=lambda: os.getenv(
        "CONFIG_PATH",
        str(BASE_DIR / "results" / "training_config.json")
    ))
    
    # Server Configuration
    HOST: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    WORKERS: int = field(default_factory=lambda: int(os.getenv("WORKERS", "1")))
    
    # Image Processing
    IMG_SIZE: int = field(default_factory=lambda: int(os.getenv("IMG_SIZE", "256")))
    
    # Logging
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = field(default_factory=_get_allowed_origins)
    
    # API Configuration
    API_TITLE: str = "Pneumonia Detection API"
    API_DESCRIPTION: str = """
    🫁 **Pneumonia Detection from Chest X-Ray Images**
    
    This API uses a deep learning model (EfficientNet) trained on chest X-ray images 
    to detect pneumonia. Upload a chest X-ray image and receive a diagnosis prediction.
    
    ## Features
    - Single image prediction
    - Batch prediction for multiple images
    - Confidence scores and probabilities
    - Health monitoring endpoint
    
    ## Usage
    1. Upload a chest X-ray image to `/predict`
    2. Receive prediction: PNEUMONIA or NORMAL
    3. Check confidence scores for reliability
    """
    API_VERSION: str = "1.0.0"
    
    # Model Classes
    CLASS_NAMES: List[str] = field(default_factory=_get_class_names)
    
    # Deployment Settings
    DEPLOYMENT_DIR: str = field(default_factory=lambda: os.getenv("DEPLOYMENT_DIR", str(BASE_DIR / "deployment")))
    AUTO_OPTIMIZE: bool = field(default_factory=lambda: os.getenv("AUTO_OPTIMIZE", "true").lower() == "true")
    TFLITE_QUANTIZE: bool = field(default_factory=lambda: os.getenv("TFLITE_QUANTIZE", "true").lower() == "true")
    MAX_BATCH_SIZE: int = field(default_factory=lambda: int(os.getenv("MAX_BATCH_SIZE", "10")))
    
    def validate(self) -> bool:
        """Validate configuration"""
        model_path = Path(self.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.MODEL_PATH}")
        return True


# Global settings instance
settings = Settings()
