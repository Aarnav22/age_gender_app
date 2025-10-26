import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import os

# ---------------------------
# App Configuration
# ---------------------------
st.set_page_config(
    page_title="Age & Gender Detector",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ---------------------------
# Dark / Techie CSS
# ---------------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #0a0f1b;
    color: #ffffff;
    font-family: 'Cambria', serif;
}
h1, h2, h3, h4, h5, h6 {
    color: #00fff5;
    font-weight: 700;
}
.css-1kyxreq.edgvbvh3 {
    background-color: #0a0f1b !important;
    border: 1px solid #444;
    border-radius: 12px;
    padding: 10px;
}
.uploaded-image-container {
    border: 2px solid #444;
    padding: 10px;
    border-radius: 12px;
    background-color:#1b1b2f;
    text-align: center;
    margin-bottom: 10px;
}
.prediction-card {
    background: linear-gradient(135deg, #1b1b2f, #0a0f1b);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    border: 2px solid #00fff5;
    box-shadow: 0 0 20px #00fff5;
    font-weight: 600;
}
h2 { font-size: 22px; }
p, span, b {
    font-family: 'Cambria', serif;
    font-weight: 600;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_age_gender_model():
    model_path = os.path.join(os.path.dirname(__file__), "models", "age_gender_model.h5")
    model = load_model(model_path, compile=False)
    return model

model = load_age_gender_model()

# ---------------------------
# Initialize Face Detector
# ---------------------------
detector = MTCNN()

# ---------------------------
# Main Header
# ---------------------------
st.markdown("<h1 style='color:#00fff5; font-size:64px; text-align:center; font-weight:800;'>🧠 Age & Gender Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:20px; color:#cccccc;'>Upload a person's face image to predict age and gender</p>", unsafe_allow_html=True)

# ---------------------------
# File Uploader
# ---------------------------
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_pil = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown(f"<b style='color:#ffffff; font-family:Cambria; font-weight:600;'>File Name:</b> {uploaded_file.name}", unsafe_allow_html=True)
    with col2:
        st.image(image_pil, caption="Uploaded Image", use_column_width=False, width=200)

    # ---------------------------
    # Convert to numpy array (RGB)
    # ---------------------------
    img_rgb = np.array(image_pil)

    # ---------------------------
    # Face Detection
    # ---------------------------
    faces = detector.detect_faces(img_rgb)
    if len(faces) == 0:
        st.error("⚠️ No face detected. Please upload a clear face image.")
    else:
        # Use the first detected face
        x, y, w, h = faces[0]['box']
        x, y = max(0, x), max(0, y)
        face_crop = img_rgb[y:y+h, x:x+w]

        # ---------------------------
        # Preprocess: Resize, Normalize, Batch
        # ---------------------------
        face_resized = cv2.resize(face_crop, (64, 64))  # match model input
        face_normalized = face_resized.astype('float32') / 255.0
        processed_img = np.expand_dims(face_normalized, axis=0)

        # ---------------------------
        # Prediction
        # ---------------------------
        with st.spinner("🔍 Detecting age and gender..."):
            preds = model.predict(processed_img)

        # ---------------------------
        # Handle different output structures
        # ---------------------------
        if isinstance(preds, list):
            age = int(preds[0][0][0])
            gender_prob = preds[1][0][0]
        else:
            if preds.shape[1] == 2:
                age = int(preds[0][0])
                gender_prob = preds[0][1]
            else:
                age = int(preds[0][0])
                gender_prob = 0.5

        gender = "Male" if gender_prob > 0.5 else "Female"
        confidence = gender_prob if gender == "Male" else 1 - gender_prob

        # ---------------------------
        # Display Predictions
        # ---------------------------
        st.subheader("🔍 Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style="color:#00fff5;">👶 Age</h2>
                <h1 style="color:#00fff5;">{age} years</h1>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="prediction-card">
                <h2 style="color:#ff6f00;">🚹 Gender</h2>
                <h1 style="color:#ff6f00;">{gender}</h1>
            </div>
            """, unsafe_allow_html=True)
