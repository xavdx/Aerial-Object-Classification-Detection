import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os, json
#paths & class names loading
PROJECT_ROOT='/content/drive/MyDrive/aerial_project'
SAVED_MODELS=os.path.join(PROJECT_ROOT, 'saved_models')
CLASS_NAMES_FILE=os.path.join(SAVED_MODELS, 'class_names.json')

#loading class names saved at training time (fallback to default)
if os.path.exists(CLASS_NAMES_FILE):
    with open(CLASS_NAMES_FILE, 'r') as f:
        CLASS_NAMES=json.load(f)
else:
    CLASS_NAMES=['bird', 'drone']  #fallback, keeping the same order as training

#Model loading (cached)
@st.cache_resource
def load_models():
    """Load both models once."""
    custom_path=os.path.join(SAVED_MODELS,"best_custom_cnn.h5")
    eff_path=os.path.join(SAVED_MODELS,"efficientnet_finetuned_full.h5")

    #Safe load= will raise if missing; let the user know in UI
    models={}
    if os.path.exists(custom_path):
        models['Custom CNN']=tf.keras.models.load_model(custom_path)
    if os.path.exists(eff_path):
        models['EfficientNetB0']=tf.keras.models.load_model(eff_path)
    return models

MODELS=load_models()
#utility= if model includes rescaling
def model_has_rescaling(model):
    """Return True if model contains a Rescaling layer (so we should NOT rescale again)."""
    from tensorflow.keras.layers import Rescaling
    for layer in model.layers:
        if isinstance(layer, Rescaling):
            return True
        #if model is a Functional model and contains nested models, we'll check their layers too
        if hasattr(layer, 'layers'):
            for sub in layer.layers:
                if isinstance(sub, Rescaling):
                    return True
    return False

#Prediction function (that's robust)
def predict_image_with_model(pil_image: Image.Image, model):
    """
    Accepts a PIL image and a loaded Keras model.
    Detects whether model contains Rescaling layer and scales accordingly.
    Returns (label_str, confidence_float).
    """
    #ensuring the RGB
    img=pil_image.convert('RGB').resize((224, 224))
    arr=tf.keras.utils.img_to_array(img)  #shape (224,224,3), dtype=float32
    arr=np.expand_dims(arr, axis=0)       #shape (1,224,224,3)

    #If model doesn't include a rescaling lay. we shall scale here
    if not model_has_rescaling(model):
        arr=arr/255.0

    preds=model.predict(arr, verbose=0)   #shape (1,2) or (1,1)
    #interpreting predictions
    if preds.ndim == 2 and preds.shape[1] == 2:
        idx=int(np.argmax(preds[0]))
        conf=float(np.max(preds[0]))
    else:
        prob=float(preds.ravel()[0])
        idx=int(prob > 0.5)
        conf=float(max(prob, 1 - prob))
    label=CLASS_NAMES[idx].capitalize()
    return label, conf

#streamlit ui
st.set_page_config(page_title="Aerial Object Classifier", layout="centered")
st.title("Aerial Object Classification (Bird vs Drone)")
st.markdown("Upload an aerial image and the app will classify it as **Bird** or **Drone** and show a confidence score.")

#model selector (list available models)
available_models=list(MODELS.keys())
if not available_models:
    st.error(f"No models found in {SAVED_MODELS}. Place your trained .h5 files there (best_custom_cnn.h5 / best_efficientnet_full.h5).")
    st.stop()

model_choice=st.sidebar.selectbox("Select model", available_models)
uploaded_file=st.file_uploader("Upload an image (jpg/png)", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image=Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        model=MODELS[model_choice]
        with st.spinner("Running model..."):
            label, conf = predict_image_with_model(image, model)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: **{conf*100:.2f}%**")

st.markdown("---")
st.caption("Project: Aerial Object Classification & Detection- Streamlit demo")
