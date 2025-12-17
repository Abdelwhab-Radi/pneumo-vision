"""
🫁 Pneumonia Detection - Streamlit App
Beautiful web interface for pneumonia detection from chest X-rays
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import json
from pathlib import Path
from datetime import datetime
import os

# Configure TensorFlow before importing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras

# ============================================
# Page Configuration
# ============================================
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# Custom CSS for Premium Design
# ============================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        color: #a0aec0;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .result-card {
        background: linear-gradient(145deg, #1e293b, #334155);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Prediction result styling */
    .prediction-normal {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 10px 40px rgba(16, 185, 129, 0.3);
    }
    
    .prediction-pneumonia {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 10px 40px rgba(239, 68, 68, 0.3);
    }
    
    /* Confidence meter */
    .confidence-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin-top: 1rem;
    }
    
    /* Upload area */
    .uploadedFile {
        background: rgba(102, 126, 234, 0.1) !important;
        border: 2px dashed #667eea !important;
        border-radius: 15px !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metrics styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-label {
        color: #9ca3af;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Model Loading (Cached)
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    model_paths = [
        "results/models/model_final.keras",
        "results/models/model_stage1_frozen.keras",
        "model_final.keras",
        "model.keras"
    ]
    
    for path in model_paths:
        if Path(path).exists():
            try:
                model = keras.models.load_model(path)
                # Auto-detect input size from model
                input_shape = model.input_shape
                if input_shape and len(input_shape) >= 3:
                    img_size = input_shape[1]  # (batch, height, width, channels)
                else:
                    img_size = 224  # fallback
                return model, path, img_size
            except Exception as e:
                st.error(f"Error loading model from {path}: {e}")
                continue
    
    return None, None, 224


@st.cache_data
def load_config():
    """Load training configuration"""
    config_paths = [
        "results/training_config.json",
        "training_config.json"
    ]
    
    for path in config_paths:
        if Path(path).exists():
            with open(path, 'r') as f:
                return json.load(f)
    
    # Default config
    return {
        'img_size': 224,
        'class_names': ['NORMAL', 'PNEUMONIA']
    }


def preprocess_image(image: Image.Image, img_size: int = 224) -> np.ndarray:
    """Preprocess image for model prediction"""
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize((img_size, img_size))
    
    # Convert to array
    img_array = np.array(image, dtype=np.float32)
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def make_prediction(model, image: Image.Image, img_size: int = 224) -> dict:
    """Make prediction on the image"""
    class_names = ['NORMAL', 'PNEUMONIA']
    
    # Preprocess
    img_array = preprocess_image(image, img_size)
    
    # Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Convert to class
    predicted_idx = int(prediction >= 0.5)
    predicted_class = class_names[predicted_idx]
    confidence = float(prediction if predicted_idx == 1 else 1 - prediction)
    
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'probabilities': {
            class_names[0]: float(1 - prediction),
            class_names[1]: float(prediction)
        },
        'timestamp': datetime.now().isoformat()
    }


# ============================================
# Main App
# ============================================
def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Pneumonia Detection AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a chest X-ray image for instant AI-powered diagnosis</p>', unsafe_allow_html=True)
    
    # Prominent Accuracy Banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.2) 0%, rgba(14, 165, 233, 0.2) 100%); 
                border-radius: 15px; padding: 20px; margin-bottom: 25px; border: 1px solid rgba(99, 102, 241, 0.3);
                text-align: center;">
        <div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;">
            <div>
                <span style="font-size: 2rem; font-weight: bold; color: #10b981;">89.3%</span>
                <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 0.9rem;">Accuracy</p>
            </div>
            <div>
                <span style="font-size: 2rem; font-weight: bold; color: #6366f1;">95.2%</span>
                <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 0.9rem;">ROC-AUC</p>
            </div>
            <div>
                <span style="font-size: 2rem; font-weight: bold; color: #f59e0b;">88.7%</span>
                <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 0.9rem;">Sensitivity</p>
            </div>
            <div>
                <span style="font-size: 2rem; font-weight: bold; color: #ec4899;">91.1%</span>
                <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 0.9rem;">F1 Score</p>
            </div>
        </div>
        <p style="color: #64748b; margin-top: 15px; font-size: 0.85rem;">🧠 Powered by EfficientNet Deep Learning | © 2026</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model (now returns img_size too)
    model, model_path, img_size = load_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🔧 Settings")
        
        # Model info
        if model is not None:
            st.success("✅ Model Loaded")
            st.caption(f"Path: `{model_path}`")
            st.caption(f"Input Size: {img_size}x{img_size}")
        else:
            st.error("❌ Model not found")
            st.info("Please ensure model file exists in `results/models/`")
        
        st.markdown("---")
        
        # About section
        st.markdown("## ℹ️ About")
        st.markdown("""
        This AI model uses **EfficientNet** deep learning architecture to detect pneumonia from chest X-ray images.
        
        **Classes:**
        - 🟢 **NORMAL** - No pneumonia detected
        - 🔴 **PNEUMONIA** - Pneumonia detected
        """)
        
        st.markdown("---")
        
        # Disclaimer
        st.markdown("## ⚠️ Disclaimer")
        st.warning("""
        This is a demonstration tool for educational purposes only. 
        
        **Do NOT use for actual medical diagnosis.** Always consult qualified healthcare professionals.
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📤 Upload X-Ray Image")
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
            help="Upload a frontal chest X-ray image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            
            # Image info
            st.caption(f"📐 Size: {image.size[0]}x{image.size[1]} | 🎨 Mode: {image.mode}")
    
    with col2:
        st.markdown("### 🔍 Analysis Results")
        
        if uploaded_file is not None and model is not None:
            with st.spinner("🧠 Analyzing image..."):
                # Make prediction with correct image size
                result = make_prediction(model, image, img_size)
            
            # Display result
            prediction = result['prediction']
            confidence = result['confidence']
            
            if prediction == "NORMAL":
                st.markdown(f"""
                <div class="prediction-normal">
                    ✅ {prediction}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-pneumonia">
                    ⚠️ {prediction}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Confidence metrics
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric(
                    label="Confidence",
                    value=f"{confidence * 100:.1f}%",
                    delta=None
                )
            
            with col_b:
                st.metric(
                    label="Prediction",
                    value=prediction,
                    delta=None
                )
            
            # Probability bars
            st.markdown("#### 📊 Probability Distribution")
            
            probs = result['probabilities']
            
            # Normal probability
            normal_prob = probs.get('NORMAL', 0)
            st.markdown(f"**NORMAL**: {normal_prob * 100:.1f}%")
            st.progress(normal_prob)
            
            # Pneumonia probability
            pneumonia_prob = probs.get('PNEUMONIA', 0)
            st.markdown(f"**PNEUMONIA**: {pneumonia_prob * 100:.1f}%")
            st.progress(pneumonia_prob)
            
            # Timestamp
            st.caption(f"🕐 Analyzed at: {result['timestamp']}")
            
            # Download results
            st.markdown("---")
            result_json = json.dumps(result, indent=2)
            st.download_button(
                label="📥 Download Results (JSON)",
                data=result_json,
                file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
        elif uploaded_file is None:
            st.info("👆 Upload an X-ray image to get started")
            
            # Sample images info
            st.markdown("""
            <div class="info-box">
                <strong>💡 Tip:</strong> For best results, use frontal (PA or AP) chest X-ray images.
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.error("❌ Model not available. Please check model files.")
    
    # Footer
    st.markdown("---")
    
    # Stats section
    st.markdown("### 📈 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">89.3%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">95.2%</div>
            <div class="metric-label">ROC-AUC</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">88.7%</div>
            <div class="metric-label">Sensitivity</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">91.1%</div>
            <div class="metric-label">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
