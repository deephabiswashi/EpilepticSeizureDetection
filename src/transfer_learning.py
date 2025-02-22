#!/usr/bin/env python3
"""
Transfer Learning Pipeline for Epileptic Seizure Detection using VGG16.
Loads EEG data from CSV, generates spectrogram images, trains a VGG16-based model,
and saves the trained model along with its training history.
"""

import ssl
# Disable SSL certificate verification (for development/testing only)
ssl._create_default_https_context = ssl._create_unverified_context

import os
import sys

# Ensure the project root is in sys.path for relative imports
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    __package__ = "src"

from .utils.data_utils import load_dataset, generate_spectrogram
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle

def generate_spectrogram_image(signal, target_size=(224, 224)):
    """
    Generate a spectrogram image from a 1D EEG signal and convert it to an array.
    """
    fig = generate_spectrogram(signal)
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    img = img.resize(target_size)
    img_array = img_to_array(img)
    buf.close()
    return img_array

def load_data_and_generate_images(csv_file):
    """
    Load the dataset and generate spectrogram images for each EEG signal sample.
    
    :param csv_file: Path to the CSV file.
    :return: Tuple (images, y) where images is a NumPy array of processed images.
    """
    X, y = load_dataset(csv_file, binary=True)
    if X.shape[0] == 0:
        raise ValueError("No valid data loaded from CSV. Check your CSV file and conversion steps.")
    images = []
    for i in range(X.shape[0]):
        # Each row in X should be a 1D numeric array representing an EEG signal.
        img_array = generate_spectrogram_image(X[i])
        images.append(img_array)
    images = np.array(images)
    print(f"Generated {images.shape[0]} spectrogram images with shape {images.shape[1:]} each.")
    # Preprocess images for VGG16
    images = preprocess_input(images)
    return images, y

def build_transfer_learning_model(input_shape=(224, 224, 3)):
    """
    Build a transfer learning model using VGG16 as the base.
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze the base model

    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Specify the path to your CSV file
    csv_file = os.path.join('data', 'processed', 'Epileptic Seizure Recognition.csv')
    print("Loading data and generating spectrogram images...")
    images, y = load_data_and_generate_images(csv_file)
    
    X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=42)
    
    print("Building transfer learning model using VGG16...")
    model = build_transfer_learning_model()
    model.summary()
    
    print("Training the model...")
    history_obj = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Specify the directory where you want to save the model and history.
    # Change 'saved_models' to your desired path.
    save_dir = os.path.join("/Users/deephabiswashi/Desktop/Deep Learning/Epilectic Seizure Detection/models")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model_save_path = os.path.join(save_dir, "saved_model.h5")
    history_save_path = os.path.join(save_dir, "training_history.pkl")
    
    # Save the model
    model.save(model_save_path)
    print(f"Model saved to '{model_save_path}'.")
    
    # Save the training history using pickle
    with open(history_save_path, "wb") as f:
        pickle.dump(history_obj.history, f)
    print(f"Training history saved to '{history_save_path}'.")

if __name__ == '__main__':
    main()
