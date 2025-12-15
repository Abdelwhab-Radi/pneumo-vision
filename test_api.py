"""
API Client Test Script
This script demonstrates how to use the Pneumonia Detection API
"""

import requests
from pathlib import Path
import json


class PneumoniaAPIClient:
    """Client for interacting with the Pneumonia Detection API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url
    
    def health_check(self):
        """Check if the API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Health check failed: {e}")
            return None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        try:
            response = requests.get(f"{self.base_url}/model/info")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to get model info: {e}")
            return None
    
    def predict_single(self, image_path: str):
        """
        Predict pneumonia for a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Prediction result dictionary
        """
        try:
            with open(image_path, 'rb') as f:
                files = {'file': (Path(image_path).name, f, 'image/jpeg')}
                response = requests.post(f"{self.base_url}/predict", files=files)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Prediction failed: {e}")
            return None
    
    def predict_batch(self, image_paths: list):
        """
        Predict pneumonia for multiple images
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of prediction results
        """
        try:
            files = []
            for img_path in image_paths:
                f = open(img_path, 'rb')
                files.append(('files', (Path(img_path).name, f, 'image/jpeg')))
            
            response = requests.post(f"{self.base_url}/predict/batch", files=files)
            
            # Close file handles
            for _, (_, f, _) in files:
                f.close()
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Batch prediction failed: {e}")
            return None


def main():
    """Main test function"""
    
    print("="*60)
    print("PNEUMONIA DETECTION API - CLIENT TEST")
    print("="*60)
    
    # Initialize client
    client = PneumoniaAPIClient()
    
    # 1. Health check
    print("\n1. Checking API health...")
    health = client.health_check()
    if health:
        print(f"   ✓ API Status: {health.get('status')}")
        print(f"   ✓ Model Loaded: {health.get('model_loaded')}")
    else:
        print("   ✗ API is not responding!")
        print("   Make sure the API server is running:")
        print("   python api.py")
        return
    
    # 2. Get model info
    print("\n2. Getting model information...")
    info = client.get_model_info()
    if info:
        print(f"   Model: {info.get('model_path')}")
        print(f"   Input Size: {info.get('input_size')}")
        print(f"   Classes: {info.get('classes')}")
    
    # 3. Test single prediction
    print("\n3. Testing single image prediction...")
    print("   To test predictions, you need an X-ray image.")
    print("   Example usage:")
    print("   ")
    print("   # Predict on a single image")
    print("   result = client.predict_single('path/to/chest_xray.jpg')")
    print("   print(f\"Prediction: {result['prediction']}\")")
    print("   print(f\"Confidence: {result['confidence']}\")")
    print("   print(f\"Probabilities: {result['probabilities']}\")")
    
    # If you have test images, uncomment and modify this:
    """
    test_image = "a:/project/chest_xray/chest_xray/test/NORMAL/IM-0001-0001.jpeg"
    if Path(test_image).exists():
        result = client.predict_single(test_image)
        if result:
            print(f"\n   ✓ Prediction Result:")
            print(f"      Class: {result.get('prediction')}")
            print(f"      Confidence: {result.get('confidence'):.4f}")
            print(f"      Probabilities: {json.dumps(result.get('probabilities'), indent=8)}")
    """
    
    # 4. Test batch prediction
    print("\n4. Batch prediction example:")
    print("   ")
    print("   # Predict on multiple images")
    print("   results = client.predict_batch([")
    print("       'path/to/image1.jpg',")
    print("       'path/to/image2.jpg',")
    print("   ])")
    print("   for i, result in enumerate(results['predictions']):")
    print("       print(f\"Image {i+1}: {result['prediction']} ({result['confidence']:.2%})\")")
    
    print("\n" + "="*60)
    print("API Client Test Complete!")
    print("="*60)
    
    print("\n📝 Next Steps:")
    print("1. Make sure the API is running: python api.py")
    print("2. Test with your own images using the client methods above")
    print("3. Access interactive API docs at: http://localhost:8000/docs")
    print("4. View API info at: http://localhost:8000")


def example_usage():
    """Example of how to use the API client with real images"""
    
    # Initialize client
    client = PneumoniaAPIClient()
    
    # Example 1: Single prediction
    image_path = "path/to/your/chest_xray.jpg"
    
    result = client.predict_single(image_path)
    if result:
        print(f"\nPrediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if result['prediction'] == 'PNEUMONIA':
            print("⚠️  Pneumonia detected!")
        else:
            print("✅ Normal chest X-ray")
    
    # Example 2: Batch prediction
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg"
    ]
    
    results = client.predict_batch(image_paths)
    if results:
        for i, pred in enumerate(results['predictions']):
            print(f"\nImage {i+1}:")
            print(f"  Prediction: {pred['prediction']}")
            print(f"  Confidence: {pred['confidence']:.2%}")


if __name__ == "__main__":
    main()
