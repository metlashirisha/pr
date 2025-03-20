import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Debug: Print model input shape
st.write(f"Model Input Shape: {model.input_shape}")

# Handle models that return a list of shapes
if isinstance(model.input_shape, list):
    input_shape = model.input_shape[0]
else:
    input_shape = model.input_shape

# Ensure the input shape is valid
if input_shape is not None and len(input_shape) == 4:
    _, img_height, img_width, img_channels = input_shape  # Ignore batch dimension
else:
    st.error("Invalid model input shape. Please check the model architecture.")
    st.stop()

# Define image preproces
