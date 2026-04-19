# STEP 10: Streamlit Demo App

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ==============================
# Config
# ==============================
IMG_SIZE = (224, 224)

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

# ==============================
# Load Model
# ==============================
@st.cache_resource
def load_trained_model():
    return load_model(r"C:\Sham\Project\TrashNet\waste_classifier_model_final.keras")

model = load_trained_model()

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Waste Classification Demo", layout="centered")

st.title("♻️ Waste Classification System")
st.write("Upload an image of waste to classify it using a CNN + Transfer Learning model.")

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# Prediction
# ==============================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = preds[0][class_idx]

    st.subheader("Prediction Result")
    st.success(f"**{class_names[class_idx].upper()}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
