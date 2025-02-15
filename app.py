import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Class names
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Load the model
@st.cache_resource
def load_model():
    model_path = r"E:\jupyter_notebook\ml_files\potato_disease\saved_model\1"
    return tf.keras.models.load_model(model_path)

MODEL = load_model()

# Image preprocessing function (matching your model testing code)
def preprocess_image(image):
    # Convert uploaded image to array
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    # Expand dimensions to add batch size
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

# Prediction function (consistent with your testing method)
def predict(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Streamlit UI
st.title("Potato Disease Detector")
st.write("Upload an image of a potato leaf to detect the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = image.resize((256, 256))  # Ensure the image is resized to 256x256
    img_array = preprocess_image(image)

    # Predict using the model
    predicted_class, confidence = predict(MODEL, img_array)

    # Display results
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}%")
