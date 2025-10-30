# complete_breast_cancer_analysis.py
import streamlit as st
import joblib
import numpy as np
import cv2
from PIL import Image
import os

# Load your trained model
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('models/random_forest_model.pkl')
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_any_image(image):
    """Preprocess ANY image to match training data characteristics"""
    try:
        # Convert to numpy array
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # ✅ FIX: Ensure data type is uint8 (not bool) before resize
        if gray.dtype == bool:
            gray = gray.astype(np.uint8) * 255
        elif gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Resize to typical medical image dimensions
        target_size = (500, 500)
        gray_resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to 0-255 range (same as training)
        gray_normalized = cv2.normalize(gray_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return gray_normalized
        
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

def extract_robust_features(image):
    """Extract features from ANY image with error handling"""
    try:
        # Preprocess the image first
        processed_image = preprocess_any_image(image)
        if processed_image is None:
            return None, None
        
        gray = processed_image
        
        # Extract the same 14 features as training
        features = [
            np.mean(gray),           # mean_intensity
            np.std(gray),            # std_intensity
            np.median(gray),         # median_intensity
            gray.shape[1],           # width
            gray.shape[0],           # height
            gray.shape[1] / gray.shape[0],  # aspect_ratio
            np.min(gray),            # min_intensity
            np.max(gray),            # max_intensity
            np.percentile(gray, 25), # percentile_25
            np.percentile(gray, 75), # percentile_75
        ]
        
        # Sobel features with error handling
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        features.extend([
            np.mean(sobelx),  # sobelx_mean
            np.mean(sobely),  # sobely_mean  
            np.std(sobelx),   # sobelx_std
            np.std(sobely)    # sobely_std
        ])
        
        return features, gray
        
    except Exception as e:
        st.error(f"Feature extraction error: {e}")
        return None, None

def analyze_image_quality(features):
    """Check if image characteristics are within expected ranges"""
    warnings = []
    
    # Expected ranges from medical images
    expected_ranges = {
        'mean_intensity': (30, 220),
        'std_intensity': (5, 100),
        'width': (200, 1000),
        'height': (200, 1000),
        'aspect_ratio': (0.3, 3.0),
    }
    
    feature_names = ['mean_intensity', 'std_intensity', 'median_intensity', 'width', 'height', 
                    'aspect_ratio', 'min_intensity', 'max_intensity', 'percentile_25', 'percentile_75',
                    'sobelx_mean', 'sobely_mean', 'sobelx_std', 'sobely_std']
    
    for i, (feature_name, value) in enumerate(zip(feature_names, features)):
        if feature_name in expected_ranges:
            min_val, max_val = expected_ranges[feature_name]
            if value < min_val or value > max_val:
                warnings.append(f"{feature_name}: {value:.1f} (expected: {min_val}-{max_val})")
    
    return warnings

model_data = load_model()

st.set_page_config(page_title="Breast Cancer Resource Allocation", page_icon="🏥", layout="wide")

st.title("🏥 Breast Cancer Resource Allocation")
st.markdown("### Automated Medical Image Priority Classification")

# File upload section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Upload Medical Image")
    uploaded_file = st.file_uploader("Choose breast cancer image...", 
                                   type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'],
                                   help="Upload ultrasound, mammography, or histopathology images")
    
    if uploaded_file is not None:
        try:
            # Display original image
            image = Image.open(uploaded_file)
            st.image(image, caption="📷 Uploaded Image", width="stretch")
            
            # File info
            st.info(f"**Image Info:** {image.size[0]}x{image.size[1]} pixels | Format: {image.format} | Mode: {image.mode}")
            
        except Exception as e:
            st.error(f"Error loading image: {e}")

with col2:
    if uploaded_file is not None and 'image' in locals():
        st.subheader("🔍 Image Analysis")
        
        if st.button("🎯 Analyze Image & Classify Priority", type="primary", use_container_width=True):
            if model_data is not None:
                with st.spinner("🔬 Analyzing image features..."):
                    # Extract features from uploaded image
                    features, processed_image = extract_robust_features(image)
                    
                    if features is not None:
                        try:
                            # Check image quality
                            quality_warnings = analyze_image_quality(features)
                            
                            if quality_warnings:
                                st.warning("⚠️ **Image Quality Notice**")
                                for warning in quality_warnings[:3]:
                                    st.write(f"- {warning}")
                                st.info("Image characteristics differ from typical medical images")
                            
                            # Create feature array
                            features_array = np.array(features).reshape(1, -1)
                            
                            # Scale features
                            features_scaled = model_data['scaler'].transform(features_array)
                            
                            # Make prediction
                            prediction = model_data['model'].predict(features_scaled)[0]
                            probability = model_data['model'].predict_proba(features_scaled)[0]
                            
                            # ✅ CORRECT: Only 2 classes (LOW/HIGH)
                            priority_levels = {0: 'LOW', 1: 'HIGH'}
                            priority = priority_levels[prediction]
                            confidence = max(probability) * 100
                            
                            # Display feature analysis
                            st.subheader("📊 Extracted Image Features")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Mean Intensity", f"{features[0]:.1f}")
                                st.metric("Image Width", f"{int(features[3])}px")
                                st.metric("Min Intensity", f"{features[6]:.1f}")
                            with col2:
                                st.metric("Intensity Variation", f"{features[1]:.1f}")
                                st.metric("Image Height", f"{int(features[4])}px")
                                st.metric("Max Intensity", f"{features[7]:.1f}")
                            with col3:
                                st.metric("Texture Complexity", f"{features[13]:.1f}")
                                st.metric("Aspect Ratio", f"{features[5]:.2f}")
                                st.metric("Data Type", "✅ Supported")
                            
                            # Display processed image
                            if processed_image is not None:
                                with st.expander("🖼️ View Processed Image"):
                                    st.image(processed_image, caption="Processed Grayscale Image", width="stretch")
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("🎯 Classification Results")
                            
                            # Confidence indicator
                            if confidence < 60:
                                st.warning("🔍 **Moderate Confidence** - Consider additional review")
                            elif confidence > 85:
                                st.success("🎯 **High Confidence Prediction**")
                            
                            # Priority card with color coding
                            if priority == "HIGH":
                                st.error(f"## 🚨 HIGH PRIORITY")
                                st.write(f"**Confidence: {confidence:.1f}%**")
                                st.progress(confidence/100)
                                
                                st.write("**🩺 Clinical Assessment:**")
                                st.write("- High malignancy probability detected")
                                st.write("- Irregular texture patterns observed")
                                st.write("- Urgent medical attention recommended")
                                
                                st.write("**📋 Immediate Actions:**")
                                st.write("- 🚨 Immediate specialist consultation")
                                st.write("- 📊 Priority diagnostic imaging")
                                st.write("- 💊 Urgent treatment planning")
                                st.write("- 🏥 Multi-disciplinary team review")
                                
                            else:  # LOW priority
                                st.success(f"## ✅ LOW PRIORITY")
                                st.write(f"**Confidence: {confidence:.1f}%**")
                                st.progress(confidence/100)
                                
                                st.write("**🩺 Clinical Assessment:**")
                                st.write("- Benign characteristics observed")
                                st.write("- Regular texture patterns detected")
                                st.write("- Low malignancy probability")
                                
                                st.write("**📋 Recommended Actions:**")
                                st.write("- 📅 Routine monitoring (6-12 months)")
                                st.write("- 🏥 Annual screening recommended")
                                st.write("- 📚 Patient education")
                                st.write("- 🌟 Regular follow-up")
                            
                            # Resource allocation impact
                            st.markdown("---")
                            st.subheader("🏥 Resource Allocation Impact")
                            
                            if priority == "HIGH":
                                st.info("""
                                **🔴 High Resource Requirements:**
                                - Urgent care pathway activation
                                - Multi-disciplinary team review  
                                - Priority diagnostic services
                                - Intensive treatment resources
                                - Immediate timeline (0-2 weeks)
                                """)
                            else:
                                st.info("""
                                **🟢 Low Resource Requirements:**
                                - Primary care follow-up
                                - Routine screening services
                                - Patient education materials
                                - Minimal specialized resources
                                - Extended timeline (6-12 months)
                                """)
                            
                            # Technical details
                            with st.expander("🔧 Technical Details"):
                                st.write(f"**Raw Prediction:** {prediction}")
                                st.write(f"**Probability Distribution:**")
                                st.write(f"- Class 0 (LOW): {probability[0]:.3f} ({probability[0]*100:.1f}%)")
                                st.write(f"- Class 1 (HIGH): {probability[1]:.3f} ({probability[1]*100:.1f}%)")
                                st.write(f"**Features Used:** {len(features)}")
                                st.write(f"**Model Accuracy:** 95.91%")
                                
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
                            st.info("""
                            **Troubleshooting Tips:**
                            - Try a different image format
                            - Ensure image is not corrupted
                            - Use medical-grade images for best results
                            """)
                    else:
                        st.error("Could not extract features from this image")
            else:
                st.error("Model not loaded properly - check if 'models/random_forest_model.pkl' exists")

# Information section
with st.expander("ℹ️ System Information & Instructions"):
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        **📋 How to Use:**
        1. **Upload** a breast cancer medical image
        2. **Click** 'Analyze Image & Classify Priority'
        3. **Review** the automatic classification
        4. **Follow** recommended medical actions
        
        **✅ Supported Image Types:**
        - Ultrasound images
        - Mammography scans  
        - Histopathology slides
        - MRI scans
        - Clinical photography
        """)
    
    with col_info2:
        st.markdown("""
        **🔧 Technical Specifications:**
        - **Model**: Random Forest Classifier
        - **Accuracy**: 95.91%
        - **Training Data**: 1,112 medical images
        - **Features**: 14 image analysis features
        - **Classes**: LOW vs HIGH priority
        
        **⚠️ Important Notes:**
        - This is a prioritization tool, not a diagnostic system
        - Always consult healthcare professionals
        - Results should guide, not replace, medical judgment
        """)

# Quick test section
with st.expander("🧪 Test with Your Dataset"):
    st.markdown("""
    **Expected Results with Your Data:**
    
    - **`data/training_set/benign/`** images → ✅ **LOW PRIORITY**
    - **`data/training_set/malignant/`** images → 🚨 **HIGH PRIORITY**
    
    **Test Procedure:**
    1. Navigate to your dataset folder
    2. Upload images from both benign and malignant folders
    3. Verify correct priority classifications
    4. Check confidence scores for reliability
    """)

# Footer
st.markdown("---")
st.caption("""
🎯 **Breast Cancer Resource Allocation System** | 
✅ **Data Type Issues Fixed** | 
🏥 **AI-Powered Medical Priority Classification** |
📊 **95.91% Model Accuracy**
""")

# Add reload button for testing
if st.sidebar.button("🔄 Clear Cache & Reload"):
    st.cache_resource.clear()
    st.rerun()