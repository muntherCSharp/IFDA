import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_fvc2004_dataset(dataset_path):
    """
    Load preprocessed FVC2004 dataset and labels.
    """
    data = []
    labels = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = tf.keras.preprocessing.image.load_img(image_path, target_size=(128, 128))
                image_array = tf.keras.preprocessing.image.img_to_array(image)
                data.append(image_array)
                # labels.append(int(root.split('/')[-1][-1]))  # Extract label from directory name
                # Extract label from directory name (assuming last character is the label)
                label = root.split('/')[-1][-1]
                if label.isdigit():  # Check if the extracted label is a digit
                    labels.append(int(label))
                else:
                    print(f"Invalid label found: {label}. Skipping...")
    return np.array(data), np.array(labels)
