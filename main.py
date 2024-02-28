from data.load_data import load_fvc2004_dataset
from model.train_model import train_fingerprint_detection_model
from detection.detect_forgery import detect_fingerprint_forgery

if __name__ == "__main__":
    # Path to the preprocessed FVC2004 dataset
    dataset_path = "datasets/MICC-F220"

    # Path to save/load the trained model
    model_path = "models/fingerprint_detection_model.h5"

    # Train the fingerprint detection model
    train_fingerprint_detection_model(dataset_path, model_path)

    # Path to the input fingerprint image
    image_path = "input_images/forgery_image.jpg"

    # Detect fingerprint forgery
    result = detect_fingerprint_forgery(image_path, model_path)
    print("Result:", result)
