# Pneumonia Detection App

## Description
This repository hosts a web application that utilizes a TensorFlow Lite model to detect pneumonia from chest X-ray images. The app is built using Streamlit and provides an intuitive interface for users to upload images and view predictions.

### Key Features:
- Lightweight TensorFlow Lite model for efficient inference.
- Streamlit-based user interface for ease of use.
- Preprocessing and augmentation pipeline integrated for optimal performance.

## Dataset
The model is trained on the **Chest X-Ray Images (Pneumonia)** dataset, which can be downloaded from Kaggle. The dataset consists of X-ray images categorized into:
- Normal
- Pneumonia

### Data Structure:
- **Training set**: `chest_xray/train`
- **Test set**: `chest_xray/test`

## Model
The model leverages the VGG16 architecture, fine-tuned with additional dense layers for binary classification (Normal vs. Pneumonia). It has been converted to TensorFlow Lite for deployment.

### Model Architecture:
- **Base Model**: VGG16 pretrained on ImageNet.
- **Custom Layers**: Augmentation, Flatten, Dense (ReLU activations), and Output (Sigmoid activation).
- **Optimization**: Binary cross-entropy loss, Adam optimizer, and class-weighting to handle data imbalance.


### Training the Model:
- Follow the notebook (`notebooks/PneumoniaDetection.ipynb`) to:
  - Download and preprocess the dataset.
  - Train and validate the model.
  - Convert the trained model to TensorFlow Lite.

## Results
The TensorFlow Lite model efficiently detects pneumonia with high accuracy. The app provides:
- Confidence scores for predictions.
- User-friendly image upload and display functionality.
