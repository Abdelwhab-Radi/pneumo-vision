"""
Simple Prediction Example
This script shows how to make a prediction on a single image using the trained model
WITHOUT needing the API (direct model usage)
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import json
from pathlib import Path


def predict_single_image(model_path: str, image_path: str, img_size: int = 224):
    """
    Make a prediction on a single image
    
    Args:
        model_path: Path to the trained model (.keras file)
        image_path: Path to the X-ray image
        img_size: Input size for the model (default: 224)
        
    Returns:
        Dictionary with prediction results
    """
    print(f"\n{'='*60}")
    print("PNEUMONIA DETECTION - SINGLE IMAGE PREDICTION")
    print(f"{'='*60}")
    
    # Load model
    print(f"\n1. Loading model...")
    print(f"   Model: {model_path}")
    try:
        model = keras.models.load_model(model_path)
        print(f"   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        return None
    
    # Load and preprocess image
    print(f"\n2. Processing image...")
    print(f"   Image: {image_path}")
    
    try:
        # Load image
        img = Image.open(image_path)
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize((img_size, img_size))
        
        # Convert to array
        img_array = np.array(img, dtype=np.float32)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"   ✅ Image processed: {img_array.shape}")
        
    except Exception as e:
        print(f"   ❌ Error processing image: {e}")
        return None
    
    # Make prediction
    print(f"\n3. Making prediction...")
    
    try:
        prediction = model.predict(img_array, verbose=0)[0][0]
        
        # Convert to class
        class_names = ['NORMAL', 'PNEUMONIA']
        predicted_class_idx = int(prediction >= 0.5)
        predicted_class = class_names[predicted_class_idx]
        confidence = float(prediction if predicted_class_idx == 1 else 1 - prediction)
        
        # Calculate probabilities
        prob_normal = float(1 - prediction)
        prob_pneumonia = float(prediction)
        
        print(f"   ✅ Prediction complete")
        
        # Display results
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"\n🔍 Prediction: {predicted_class}")
        print(f"📊 Confidence: {confidence:.2%}")
        print(f"\n📈 Probabilities:")
        print(f"   NORMAL:    {prob_normal:.4f} ({prob_normal*100:.2f}%)")
        print(f"   PNEUMONIA: {prob_pneumonia:.4f} ({prob_pneumonia*100:.2f}%)")
        
        # Interpretation
        print(f"\n💡 Interpretation:")
        if predicted_class == 'PNEUMONIA':
            if confidence >= 0.9:
                print(f"   ⚠️  HIGH CONFIDENCE: Pneumonia detected")
            elif confidence >= 0.7:
                print(f"   ⚠️  MODERATE CONFIDENCE: Pneumonia likely")
            else:
                print(f"   ⚠️  LOW CONFIDENCE: Pneumonia possible")
        else:
            if confidence >= 0.9:
                print(f"   ✅ HIGH CONFIDENCE: Normal chest X-ray")
            elif confidence >= 0.7:
                print(f"   ✅ MODERATE CONFIDENCE: Likely normal")
            else:
                print(f"   ⚠️  LOW CONFIDENCE: Uncertain, review needed")
        
        print(f"\n{'='*60}")
        
        # Return results
        result = {
            'prediction': predicted_class,
            'confidence': round(confidence, 4),
            'probabilities': {
                'NORMAL': round(prob_normal, 4),
                'PNEUMONIA': round(prob_pneumonia, 4)
            },
            'raw_score': float(prediction)
        }
        
        return result
        
    except Exception as e:
        print(f"   ❌ Error during prediction: {e}")
        return None


def predict_batch_images(model_path: str, image_paths: list, img_size: int = 224):
    """
    Make predictions on multiple images
    
    Args:
        model_path: Path to the trained model
        image_paths: List of paths to X-ray images
        img_size: Input size for the model
        
    Returns:
        List of prediction results
    """
    print(f"\n{'='*60}")
    print(f"BATCH PREDICTION - {len(image_paths)} images")
    print(f"{'='*60}")
    
    # Load model once
    print(f"\nLoading model...")
    try:
        model = keras.models.load_model(model_path)
        print(f"✅ Model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Process each image
    results = []
    for i, image_path in enumerate(image_paths, 1):
        print(f"\n--- Image {i}/{len(image_paths)} ---")
        print(f"File: {Path(image_path).name}")
        
        try:
            # Load and preprocess
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((img_size, img_size))
            img_array = np.array(img, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_array, verbose=0)[0][0]
            
            # Process result
            class_names = ['NORMAL', 'PNEUMONIA']
            predicted_class_idx = int(prediction >= 0.5)
            predicted_class = class_names[predicted_class_idx]
            confidence = float(prediction if predicted_class_idx == 1 else 1 - prediction)
            
            result = {
                'image': str(image_path),
                'prediction': predicted_class,
                'confidence': round(confidence, 4),
                'probabilities': {
                    'NORMAL': round(float(1 - prediction), 4),
                    'PNEUMONIA': round(float(prediction), 4)
                }
            }
            
            results.append(result)
            
            # Display
            print(f"Result: {predicted_class} ({confidence:.2%} confident)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                'image': str(image_path),
                'error': str(e)
            })
    
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE - {len(results)} results")
    print(f"{'='*60}")
    
    return results


def main():
    """Example usage"""
    
    # Configuration
    model_path = "a:/project/results/models/model_stage1_frozen.keras"
    
    print("\n" + "="*60)
    print("SIMPLE PREDICTION SCRIPT")
    print("="*60)
    print("\nThis script allows you to make predictions WITHOUT starting the API.")
    print("Just provide the path to an X-ray image.")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"\n❌ Model not found at: {model_path}")
        print("Please train your model first or update the model_path variable.")
        return
    
    print(f"\n✅ Model found: {model_path}")
    
    # Example 1: Single image prediction
    print("\n" + "="*60)
    print("EXAMPLE 1: SINGLE IMAGE PREDICTION")
    print("="*60)
    print("\nTo predict on a single image:")
    print("""
    result = predict_single_image(
        model_path="a:/project/results/models/model_stage1_frozen.keras",
        image_path="path/to/your/chest_xray.jpg"
    )
    """)
    
    # If you have a test image, uncomment this:
    """
    test_image = "a:/project/chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
    if Path(test_image).exists():
        result = predict_single_image(model_path, test_image)
        
        if result:
            print(f"\nResult saved:")
            print(json.dumps(result, indent=2))
    """
    
    # Example 2: Batch prediction
    print("\n" + "="*60)
    print("EXAMPLE 2: BATCH PREDICTION")
    print("="*60)
    print("\nTo predict on multiple images:")
    print("""
    results = predict_batch_images(
        model_path="a:/project/results/models/model_stage1_frozen.keras",
        image_paths=[
            "path/to/xray1.jpg",
            "path/to/xray2.jpg",
            "path/to/xray3.jpg"
        ]
    )
    
    for result in results:
        print(f"{result['image']}: {result['prediction']} ({result['confidence']:.2%})")
    """)
    
    print("\n" + "="*60)
    print("USAGE INSTRUCTIONS")
    print("="*60)
    print("""
To use this script with your own images:

1. Edit this file (simple_predict.py)
2. Update the test_image path in the main() function
3. Uncomment the test code section
4. Run: python simple_predict.py

Or use it in your own code:
    
    from simple_predict import predict_single_image
    
    result = predict_single_image(
        model_path="path/to/model.keras",
        image_path="path/to/xray.jpg"
    )
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    """)
    
    print("\n" + "="*60)
    print("READY!")
    print("="*60)
    print("\n✅ Functions available:")
    print("   - predict_single_image(model_path, image_path)")
    print("   - predict_batch_images(model_path, image_paths)")
    print("\n✅ Import this module to use in your code:")
    print("   from simple_predict import predict_single_image")


if __name__ == "__main__":
    main()
