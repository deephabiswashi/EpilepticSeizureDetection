import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy import signal as sig

def load_dataset(csv_file, binary=True):
    """
    Load the dataset from a CSV file.
    Assumes that the CSV file has an extra index column that should be skipped.
    Converts feature columns (all except the last) to numeric and drops rows with NaN.
    
    :param csv_file: Path to the CSV file.
    :param binary: If True, converts labels to binary (1 for seizure, 0 for non-seizure).
    :return: tuple (X, y)
    """
    # Read CSV while skipping the first column (the index)
    df = pd.read_csv(csv_file, index_col=0)
    
    # Convert all columns to numeric; non-numeric values become NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Drop rows with any NaN values in the features or label
    df.dropna(inplace=True)
    
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns from {csv_file}.")
    
    # Assume that the last column is the label
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)
    
    if binary:
        y = (y == 1).astype(int)
    
    return X, y

def apply_tqdwt(signal, wavelet='db4', level=3):
    """
    Apply multi-level wavelet decomposition as a placeholder for TQDWT.
    
    :param signal: 1D numeric array.
    :return: Concatenated wavelet coefficients.
    """
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    return np.concatenate(coeffs)

def preprocess_data(X, transform=True):
    """
    Preprocess the raw EEG data.
    
    :param X: 2D array of raw signals.
    :param transform: If True, apply wavelet transform.
    :return: Processed data array.
    """
    if transform:
        X_processed = np.array([apply_tqdwt(x) for x in X])
    else:
        X_processed = X
    return X_processed

def generate_spectrogram(signal, fs=178, nperseg=64):
    """
    Generate a spectrogram from a 1D numeric signal.
    
    :param signal: 1D numeric array.
    :param fs: Sampling frequency.
    :param nperseg: Length of each segment for the spectrogram.
    :return: A matplotlib figure.
    """
    f, t, Sxx = sig.spectrogram(signal, fs=fs, nperseg=nperseg)
    fig, ax = plt.subplots(figsize=(4, 4))
    cax = ax.pcolormesh(t, f, Sxx, shading='gouraud')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    fig.colorbar(cax, ax=ax)
    plt.close(fig)
    return fig
