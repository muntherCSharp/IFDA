import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2

def preprocess_image(image_path):
    """
    Preprocess the input image.
    """
    image = cv2.imread('image.jpg')
    image = cv2.resize(image, (128, 128))  # Resize image to match model input shape
    image = image / 255.0  # Normalize pixel values
    return image.reshape(1, 128, 128, 3)  # Reshape image for model input

def detect_fingerprint_forgery(image_path, model_path):
    """
    Detect fingerprint forgery using the trained model.
    """
    # Load the trained model
    model = load_model(model_path)

    # Preprocess the input image
    preprocessed_image = preprocess_image(image_path)

    # Perform prediction
    prediction = model.predict(preprocessed_image)

    # Classify the image as authentic or manipulated based on the prediction
    if prediction[0][0] >= 0.5:
        return "Authentic"
    else:
        return "Manipulated"
