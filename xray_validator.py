"""
Chest X-ray Image Validator

This module provides multi-layer validation to ensure uploaded images
are valid chest X-ray images before processing them for pneumonia detection.

Validation Layers:
1. Image Characteristics: Grayscale detection, histogram analysis, aspect ratio
2. Pre-trained Model: MobileNetV2 classification to detect medical imagery
3. Confidence Threshold: Additional safety net based on prediction confidence

Author: Pneumonia Detection Team
Version: 1.0.0
"""

import numpy as np
from PIL import Image
import io
import logging
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

# TensorFlow imports with lazy loading
_tf = None
_keras = None
_mobilenet = None

def _load_tensorflow():
    """Lazy load TensorFlow to avoid import delays"""
    global _tf, _keras, _mobilenet
    if _tf is None:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras.applications import mobilenet_v2
        _tf = tf
        _keras = keras
        _mobilenet = mobilenet_v2
    return _tf, _keras, _mobilenet


@dataclass
class ValidationResult:
    """Result of image validation"""
    is_valid: bool
    confidence: float
    message_en: str
    message_ar: str
    validation_details: Dict
    
    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "confidence": self.confidence,
            "message": self.message_en,
            "message_ar": self.message_ar,
            "details": self.validation_details
        }


class XrayValidator:
    """
    Multi-layer validator for chest X-ray images.
    
    This validator combines multiple techniques to ensure that only
    valid chest X-ray images are processed for pneumonia detection.
    """
    
    # Validation thresholds
    GRAYSCALE_THRESHOLD = 0.85  # How close to grayscale the image should be
    MIN_CONTRAST_THRESHOLD = 30  # Minimum standard deviation in pixel values
    MAX_CONTRAST_THRESHOLD = 120  # Maximum standard deviation (avoid over-saturated)
    ASPECT_RATIO_MIN = 0.7  # Minimum aspect ratio for medical images
    ASPECT_RATIO_MAX = 1.5  # Maximum aspect ratio for medical images
    XRAY_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for X-ray classification
    
    # ImageNet classes related to medical/X-ray imagery
    MEDICAL_RELATED_CLASSES = {
        # These are ImageNet class indices that might indicate medical imagery
        # or patterns similar to X-rays
        'radiograph': True,
        'x-ray': True,
        'magnetic_resonance': True,
        'ct_scan': True,
    }
    
    # Classes that definitely indicate non-X-ray images
    REJECT_KEYWORDS = [
        'person', 'man', 'woman', 'boy', 'girl', 'face', 'head',
        'dog', 'cat', 'car', 'house', 'tree', 'flower', 'food',
        'phone', 'computer', 'building', 'landscape', 'selfie',
        'portrait', 'animal', 'pet', 'outdoor', 'indoor'
    ]
    
    def __init__(self, use_pretrained_model: bool = True):
        """
        Initialize the validator.
        
        Args:
            use_pretrained_model: Whether to use MobileNetV2 for additional validation
        """
        self.use_pretrained_model = use_pretrained_model
        self.mobilenet_model = None
        self.decode_predictions = None
        
        if use_pretrained_model:
            self._load_pretrained_model()
    
    def _load_pretrained_model(self):
        """Load MobileNetV2 for image classification"""
        try:
            tf, keras, mobilenet = _load_tensorflow()
            
            logger.info("Loading MobileNetV2 for image validation...")
            self.mobilenet_model = mobilenet.MobileNetV2(
                weights='imagenet',
                include_top=True,
                input_shape=(224, 224, 3)
            )
            self.preprocess_input = mobilenet.preprocess_input
            self.decode_predictions = keras.applications.mobilenet_v2.decode_predictions
            logger.info("✓ MobileNetV2 loaded for validation")
            
        except Exception as e:
            logger.warning(f"Could not load MobileNetV2: {e}")
            logger.warning("Falling back to characteristics-only validation")
            self.use_pretrained_model = False
    
    def _analyze_image_characteristics(self, image: Image.Image) -> Dict:
        """
        Analyze image characteristics to detect X-ray-like properties.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with analysis results
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Get image dimensions
        height, width = img_array.shape[:2]
        aspect_ratio = width / height
        
        # Check if image is grayscale or near-grayscale
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Calculate color variance
            r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]
            
            # Measure how similar RGB channels are (grayscale = very similar)
            rgb_diff = np.abs(r.astype(float) - g.astype(float)) + \
                      np.abs(g.astype(float) - b.astype(float)) + \
                      np.abs(r.astype(float) - b.astype(float))
            avg_rgb_diff = np.mean(rgb_diff)
            max_possible_diff = 255 * 3
            grayscale_score = 1 - (avg_rgb_diff / max_possible_diff)
            
            # Convert to grayscale for histogram analysis
            gray = np.mean(img_array[:,:,:3], axis=2)
        else:
            grayscale_score = 1.0
            gray = img_array if len(img_array.shape) == 2 else img_array[:,:,0]
        
        # Analyze histogram distribution
        hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
        hist_normalized = hist / hist.sum()
        
        # Calculate contrast (standard deviation of pixel values)
        contrast = np.std(gray)
        
        # Calculate histogram entropy (X-rays tend to have lower entropy)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        
        # Check for bimodal distribution (common in X-rays: dark background, lighter anatomy)
        dark_pixels = np.sum(gray < 50) / gray.size
        bright_pixels = np.sum(gray > 200) / gray.size
        mid_pixels = np.sum((gray >= 50) & (gray <= 200)) / gray.size
        
        return {
            "width": width,
            "height": height,
            "aspect_ratio": round(aspect_ratio, 3),
            "grayscale_score": round(grayscale_score, 3),
            "contrast": round(contrast, 2),
            "histogram_entropy": round(entropy, 2),
            "dark_pixel_ratio": round(dark_pixels, 3),
            "bright_pixel_ratio": round(bright_pixels, 3),
            "mid_tone_ratio": round(mid_pixels, 3)
        }
    
    def _validate_characteristics(self, characteristics: Dict) -> Tuple[bool, float, str, str]:
        """
        Validate image based on characteristics analysis.
        
        Returns:
            Tuple of (is_valid, confidence, message_en, message_ar)
        """
        issues = []
        confidence = 1.0
        
        # Check aspect ratio
        ar = characteristics["aspect_ratio"]
        if ar < self.ASPECT_RATIO_MIN or ar > self.ASPECT_RATIO_MAX:
            issues.append("unusual_aspect_ratio")
            confidence -= 0.2
        
        # Check grayscale score
        gs = characteristics["grayscale_score"]
        if gs < self.GRAYSCALE_THRESHOLD:
            issues.append("too_colorful")
            confidence -= 0.3
        
        # Check contrast
        contrast = characteristics["contrast"]
        if contrast < self.MIN_CONTRAST_THRESHOLD:
            issues.append("low_contrast")
            confidence -= 0.2
        elif contrast > self.MAX_CONTRAST_THRESHOLD:
            issues.append("high_contrast")
            confidence -= 0.1
        
        # Check pixel distribution (X-rays typically have significant dark regions)
        if characteristics["dark_pixel_ratio"] < 0.1:
            issues.append("no_dark_regions")
            confidence -= 0.2
        
        # Determine result
        confidence = max(0, confidence)
        is_valid = len(issues) <= 1 and confidence >= 0.5
        
        if is_valid:
            message_en = "Image characteristics are consistent with chest X-ray"
            message_ar = "خصائص الصورة متوافقة مع أشعة الصدر"
        else:
            if "too_colorful" in issues:
                message_en = "Image appears to be a color photograph, not an X-ray. Please upload a chest X-ray image."
                message_ar = "الصورة تبدو صورة ملونة وليست أشعة. من فضلك ارفع صورة أشعة صدر."
            elif "low_contrast" in issues:
                message_en = "Image has insufficient contrast for an X-ray. Please upload a clear chest X-ray."
                message_ar = "الصورة ذات تباين منخفض. من فضلك ارفع صورة أشعة صدر واضحة."
            else:
                message_en = "Image does not appear to be a valid chest X-ray. This system only accepts chest X-ray images."
                message_ar = "الصورة لا تبدو أشعة صدر صالحة. هذا النظام يقبل صور أشعة الصدر فقط."
        
        return is_valid, confidence, message_en, message_ar
    
    def _validate_with_pretrained_model(self, image: Image.Image) -> Tuple[bool, float, str, str, Dict]:
        """
        Validate image using MobileNetV2 classification.
        
        Returns:
            Tuple of (is_valid, confidence, message_en, message_ar, predictions)
        """
        if not self.use_pretrained_model or self.mobilenet_model is None:
            return True, 1.0, "", "", {}
        
        tf, keras, _ = _load_tensorflow()
        
        try:
            # Resize and preprocess for MobileNetV2
            img_resized = image.resize((224, 224))
            img_array = np.array(img_resized, dtype=np.float32)
            
            # Ensure 3 channels
            if len(img_array.shape) == 2:
                img_array = np.stack([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 1:
                img_array = np.concatenate([img_array] * 3, axis=-1)
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            
            # Add batch dimension and preprocess
            img_array = np.expand_dims(img_array, axis=0)
            img_array = self.preprocess_input(img_array)
            
            # Get predictions
            predictions = self.mobilenet_model.predict(img_array, verbose=0)
            decoded = self.decode_predictions(predictions, top=5)[0]
            
            # Check if any top predictions indicate non-X-ray content
            top_predictions = {}
            is_rejected = False
            rejected_class = None
            
            for class_id, class_name, prob in decoded:
                top_predictions[class_name] = float(prob)
                
                # Check if this is a clearly non-X-ray class
                class_lower = class_name.lower()
                for reject_keyword in self.REJECT_KEYWORDS:
                    if reject_keyword in class_lower:
                        if prob > 0.1:  # If confidence > 10% for rejected class
                            is_rejected = True
                            rejected_class = class_name
                            break
                
                if is_rejected:
                    break
            
            if is_rejected:
                confidence = 0.0
                message_en = f"This appears to be a '{rejected_class}' image, not a chest X-ray. Please upload only chest X-ray images."
                message_ar = f"هذه الصورة تبدو '{rejected_class}' وليست أشعة صدر. من فضلك ارفع صور أشعة الصدر فقط."
                return False, confidence, message_en, message_ar, top_predictions
            
            # If not clearly rejected, consider it potentially valid
            # X-rays often get classified as abstract patterns
            return True, 0.7, "", "", top_predictions
            
        except Exception as e:
            logger.warning(f"Pre-trained model validation failed: {e}")
            return True, 0.5, "", "", {}
    
    def validate(self, image_bytes: bytes) -> ValidationResult:
        """
        Perform full validation on an image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            ValidationResult with detailed results
        """
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB for consistent processing
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Layer 1: Analyze image characteristics
            characteristics = self._analyze_image_characteristics(image)
            char_valid, char_confidence, char_msg_en, char_msg_ar = self._validate_characteristics(characteristics)
            
            # Layer 2: Pre-trained model validation
            model_valid, model_confidence, model_msg_en, model_msg_ar, model_predictions = \
                self._validate_with_pretrained_model(image)
            
            # Combine results
            validation_details = {
                "characteristics": characteristics,
                "characteristics_valid": char_valid,
                "model_valid": model_valid,
                "model_predictions": model_predictions
            }
            
            # Determine final result
            if not char_valid and char_confidence < 0.3:
                # Clearly not an X-ray based on characteristics
                return ValidationResult(
                    is_valid=False,
                    confidence=char_confidence,
                    message_en=char_msg_en,
                    message_ar=char_msg_ar,
                    validation_details=validation_details
                )
            
            if not model_valid:
                # Pre-trained model identified non-X-ray content
                return ValidationResult(
                    is_valid=False,
                    confidence=model_confidence,
                    message_en=model_msg_en,
                    message_ar=model_msg_ar,
                    validation_details=validation_details
                )
            
            # Image passed validation
            combined_confidence = (char_confidence + model_confidence) / 2
            
            return ValidationResult(
                is_valid=True,
                confidence=combined_confidence,
                message_en="Image validated as potential chest X-ray",
                message_ar="تم التحقق من الصورة كأشعة صدر محتملة",
                validation_details=validation_details
            )
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                message_en=f"Could not process image: {str(e)}",
                message_ar=f"لم نتمكن من معالجة الصورة: {str(e)}",
                validation_details={"error": str(e)}
            )
    
    def validate_prediction_confidence(self, prediction_confidence: float, threshold: float = 0.6) -> Tuple[bool, str, str]:
        """
        Layer 3: Validate based on prediction model confidence.
        
        If the pneumonia detection model is very uncertain, the image
        might not be a proper chest X-ray.
        
        Args:
            prediction_confidence: Confidence from the prediction model
            threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (is_confident, message_en, message_ar)
        """
        if prediction_confidence < threshold:
            return (
                False,
                f"The model has low confidence ({prediction_confidence:.1%}) in this prediction. "
                "The image may not be a clear chest X-ray.",
                f"الموديل لديه ثقة منخفضة ({prediction_confidence:.1%}) في هذا التنبؤ. "
                "قد لا تكون الصورة أشعة صدر واضحة."
            )
        return (True, "", "")


# Singleton instance for reuse
_validator_instance: Optional[XrayValidator] = None

def get_validator() -> XrayValidator:
    """Get or create the global validator instance"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = XrayValidator()
    return _validator_instance
