import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import os

from util import classify, set_background


set_background(os.path.join(os.path.dirname(__file__), "../images/streamlit.png"))

# set title
st.markdown("<h1 style='text-align: center;'> 🦴 Bone Fracture Detection</h1>", unsafe_allow_html=True)

# set description
st.markdown("""
Fractures are a common clinical issue, and X-rays remain the most widely used diagnostic tool to identify them. However, interpreting these images manually can be time-consuming and prone to error—especially when fractures are subtle or image quality is low.

This app uses a **deep learning model** to automate the binary classification of bone X-rays into:

- **Fractured ⚠️**
- **Not Fractured ✅**

Leveraging transfer learning with pre-trained convolutional neural networks, the model aims to assist in achieving high-accuracy fracture detection directly from raw X-ray images.
""")

# upload file
file = st.file_uploader('Please upload a  X-ray image to predict if the bone is fractured.', type=['jpeg', 'jpg', 'png'])

# load classifier
model = load_model(os.path.join(os.path.dirname(__file__), '../models/best_model_resnet.h5'))

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_container_width=True, width=100)

    # classify image
    class_name, conf_score = classify(image, model)

    # write classification
    st.write("## {}".format(class_name))
    st.write("### Confidence score: {}%".format(int(conf_score * 10000) / 100))