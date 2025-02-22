import numpy as np
import pandas as pd
import pywt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def apply_tqdwt(signal, wavelet='db4', level=3):
    """
    Apply multi-level wavelet decomposition as a placeholder for TQDWT.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate(coeffs)

def load_dataset(csv_file):
    """
    Load the dataset from CSV.
    Converts labels to binary (1 for seizure, 0 for non-seizure).
    Handles any non-numeric data issues.
    """
    df = pd.read_csv(csv_file)

    # Drop any non-numeric columns (e.g., extra headers, IDs)
    df = df.select_dtypes(include=[np.number])

    # Ensure all values are numerical
    X = df.iloc[:, :-1].astype(float).values  # Feature columns
    y = df.iloc[:, -1].values  # Target column

    # Convert to binary: seizure (class 1) vs. non-seizure (classes 2-5)
    y_binary = (y == 1).astype(int)

    return X, y_binary

def preprocess_data(X):
    """
    Preprocess each sample using the TQDWT placeholder.
    """
    X_transformed = np.array([apply_tqdwt(x) for x in X])
    return X_transformed

def build_model(input_shape):
    """
    Build a simple 1D CNN for binary classification.
    """
    model = Sequential()
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))  # Single neuron for binary classification
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # Update the file path to point to the processed dataset folder
    X, y = load_dataset('data/processed/Epileptic Seizure Recognition.csv')
    
    # Preprocess data using the TQDWT placeholder
    X_processed = preprocess_data(X)
    
    # Reshape for Conv1D: (samples, timesteps, channels)
    X_processed = X_processed[..., np.newaxis]
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    input_shape = X_train.shape[1:]
    model = build_model(input_shape)
    model.summary()
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    main()
