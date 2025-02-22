# Epileptic Seizure Detection

## A Deep Learning Pipeline for EEG-Based Seizure Classification

---

## Overview
This repository provides a complete solution for detecting epileptic seizures from EEG data using:

- **Wavelet Transforms** (as a placeholder for TQDWT)
- **1D Convolutional Neural Networks**
- **Transfer Learning** (VGG16-based approach)
- **Keras/TensorFlow** for deep learning

The primary objective is to classify EEG signals into seizure vs. non-seizure classes.

---

## Directory Structure

```
Epileptic Seizure Detection/
├── data/
│   └── processed/
│       └── Epileptic Seizure Recognition.csv
├── env1/                      # Local Python virtual environment
├── models/
│   ├── saved_model.h5
│   └── training_history.pkl
├── notebooks/
│   └── exploratory_analysis.ipynb
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn_model.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── data_utils.py
│   ├── main.py                # Baseline 1D CNN
│   └── transfer_learning.py   # Transfer learning (VGG16)
├── .gitignore
├── requirements.txt
└── README.md                  
```

### Description of Key Files

- **`data/processed/`** contains the CSV file with EEG data.
- **`env1/`** is your local Python virtual environment (excluded via `.gitignore`).
- **`models/`** stores the trained model (`saved_model.h5`) and training history (`training_history.pkl`).
- **`notebooks/`** holds a Jupyter notebook for exploratory analysis and model evaluation.
- **`src/`** contains all source code, including utility functions, CNN code, and transfer learning scripts.

---

## Installation & Setup

### 1. Clone the Repository:
```bash
git clone https://github.com/deephabiswashi/EpilepticSeizureDetection.git
cd EpilepticSeizureDetection
```

### 2. Create & Activate a Virtual Environment (Recommended):
```bash
python -m venv env1
source env1/bin/activate   # On Windows: env1\Scripts\activate
```

### 3. Install Dependencies:
```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Baseline 1D CNN (`main.py`):
Trains a 1D CNN on wavelet-transformed EEG signals.
```bash
python src/main.py
```
- Loads data from `data/processed/Epileptic Seizure Recognition.csv`
- Applies wavelet decomposition (placeholder for TQDWT)
- Trains and evaluates a simple CNN model

### 2. Transfer Learning (`transfer_learning.py`):
Uses a frozen VGG16 base and trains a classifier on spectrogram images generated from EEG signals.
```bash
python src/transfer_learning.py
```
- Converts EEG signals to spectrogram images
- Trains a transfer learning model
- Saves `saved_model.h5` and `training_history.pkl` in the `models/` folder

### 3. Exploratory Analysis & Evaluation:
- Open `notebooks/exploratory_analysis.ipynb` in Jupyter or VS Code
- Run the cells to visualize data distribution, generate spectrograms, load the saved model, and evaluate performance

---

## Results

- **Accuracy:** ~97–98% on test data (depending on hyperparameters and dataset splits)
- **Confusion Matrix & Classification Report:** Shown in the `exploratory_analysis.ipynb` notebook

---

## Acknowledgments

- **EEG Dataset:** [Epileptic Seizure Recognition dataset](https://www.kaggle.com/datasets/harunshimanto/epileptic-seizure-recognition)
- **Wavelet Library:** [PyWavelets](https://pywavelets.readthedocs.io/en/latest/)
- **Deep Learning Framework:** [Keras/TensorFlow](https://www.tensorflow.org/)

---

**Made by Deep Habiswashi** 🚀
