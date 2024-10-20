import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained U-Net model
model = load_model('liver_cancer_unet.h5')

# Function to preprocess the uploaded image
def preprocess_image(image, target_size=(128, 128)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to determine tumor presence based on prediction
def detect_tumor(prediction):
    # Check if any pixel in the predicted mask indicates a tumor
    mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8)  # Threshold the prediction
    tumor_detected = np.any(mask)  # Check if there's any positive detection
    return tumor_detected

# Streamlit app
st.title("Liver Cancer Detection Using U-Net")
st.write("Upload a liver scan image to check for tumor detection.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Processing...")

    # Preprocess the image
    image_array = preprocess_image(image)

    # Make prediction
    prediction = model.predict(image_array)

    # Detect tumor presence
    tumor_detected = detect_tumor(prediction)

    # Display the prediction result
    if tumor_detected:
        st.write("Tumor Detected!")
    else:
        st.write("No Tumor Detected.")
