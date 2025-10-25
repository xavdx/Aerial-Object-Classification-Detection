#app.py — Streamlit Deployment
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

#Paths
PROJECT_ROOT='/content/drive/MyDrive/aerial_project'
SAVED_MODELS=os.path.join(PROJECT_ROOT, 'saved_models')

#Load models lazily
@st.cache_resource
def load_model(model_name):
    model_path=os.path.join(SAVED_MODELS, model_name)
    model=tf.keras.models.load_model(model_path)
    return model

#Prediction function
def predict_image(model, image, class_names=['bird', 'drone']):
    img=image.resize((224, 224))
    img_array=tf.keras.utils.img_to_array(img)
    img_array=np.expand_dims(img_array, axis=0) / 255.0
    preds=model.predict(img_array)
    pred_class=np.argmax(preds, axis=1)[0]
    confidence=float(np.max(preds))
    return class_names[pred_class].capitalize(), confidence

#Streamlit UI
st.set_page_config(page_title="Aerial Object Classifier", layout="centered")

st.title("Aerial Object Classification (Bird vs Drone)")
st.markdown("This app uses a trained deep learning model to classify aerial images as **Bird** or **Drone**.")

st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Select Model:",
    ("Custom CNN (Best Performing)", "EfficientNet (Transfer Learning)")
)

if model_choice == "Custom CNN (Best Performing)":
    model_file = "best_custom_cnn.h5"
else:
    model_file = "best_efficientnet_full.h5"

# Load selected model
with st.spinner(f"Loading {model_choice}..."):
    model = load_model(model_file)
st.success(f"{model_choice} loaded successfully!")

# Image upload section
uploaded_file = st.file_uploader("📤 Upload an aerial image (Bird or Drone):", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Classify Image"):
        with st.spinner("Analyzing..."):
            pred_class, confidence = predict_image(model, image)
        st.success(f"**Prediction:** {pred_class}")
        st.info(f"**Confidence:** {confidence*100:.2f}%")

        st.markdown("---")
        st.caption("YOLOv8 object detection can be integrated here in the future for bounding box visualization.")
else:
    st.warning("Please upload an image to classify.")

st.markdown("---")
st.markdown("**Project:** Aerial Object Classification & Detection  \n**Domain:** Aerial Surveillance, Wildlife Monitoring, Security & Defense")
