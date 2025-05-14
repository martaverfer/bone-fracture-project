import base64
import tensorflow as tf
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input


def set_background(image_file):
    """
    This function sets the background of a Streamlit app to an image specified by the given image file.

    Parameters:
        image_file (str): The path to the image file to be used as the background.

    Returns:
        None
    """
    try:
        with open(image_file, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()
        style = f"""
            <style>
            .stApp {{
                background-image: url(data:image/png;base64,{b64_encoded});
                background-size: cover;
            }}
            </style>
        """
        st.markdown(style, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ Background image not found. Using default background.")


def classify(image, model):
    """
    This function takes an image, a model, and a list of class names and returns the predicted class and confidence
    score of the image.

    Parameters:
        image (PIL.Image.Image): An image to be classified.
        model (tensorflow.keras.Model): A trained machine learning model for image classification.
        class_names (list): A list of class names corresponding to the classes that the model can predict.

    Returns:
        A tuple of the predicted class name and the confidence score for that prediction.
    """

    # Preprocess the image to match the input size the model expects
    img = image.resize((224, 224))  # Resize to (224, 224) for ResNet50 input size
    img_array = np.array(img)  # Convert to numpy array
    img_array_add_dimension = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array_preprocessed = preprocess_input(img_array_add_dimension)  # Preprocess input for ResNet50

    # make prediction
    raw_pred = model.predict(img_array_preprocessed)

    threshold = 0.1
    prediction = "Not Fractured ✅" if raw_pred[0] >= threshold else "Fractured ⚠️"
    confidence = raw_pred if raw_pred >= threshold else 1 - raw_pred

    return prediction, confidence