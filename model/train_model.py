import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from data.load_data import load_fvc2004_dataset  # Import load_fvc2004_dataset function

def train_fingerprint_detection_model(dataset_path, model_path):
    """
    Train the fingerprint detection model.
    """
    # Load dataset and labels using load_fvc2004_dataset function
    data, labels = load_fvc2004_dataset(dataset_path)

    # Split dataset into training and validation sets
    #X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)
    #X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.1, train_size=0.9, random_state=42)

    # Normalize pixel values
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    # Define the model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification (authentic or manipulated)
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    # Save the trained model
    model.save(model_path)
    print("Model saved successfully.")
