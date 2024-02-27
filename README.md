# AI-Winter2024







# MNIST Digit Classification

This repository contains a Python script for classifying handwritten digits from the MNIST dataset using Support Vector Machine (SVM) and Logistic Regression models. It demonstrates data loading, preprocessing, model training, evaluation, and visualization of results.

## Prerequisites

Before running this script, ensure you have the following packages installed:

- numpy
- matplotlib
- seaborn
- scikit-learn
- requests

You can install these packages using pip:

```bash
pip install numpy matplotlib seaborn scikit-learn requests

Dataset
The MNIST dataset is used in this project, which is a collection of 28x28 pixel grayscale images of handwritten digits (0 through 9). The script automatically downloads the dataset using provided URLs.

Features
Data Loading: Functions to download and load MNIST dataset images and labels.
Preprocessing: Flattening of image data and normalization.
Model Training: Training of SVM and Logistic Regression models on the preprocessed data.
Evaluation: Calculation and display of accuracy, precision, recall, and F1-score for both models. Visualization of confusion matrix and sample predictions.
Results
After running the script, it will print the accuracy, precision, recall, and F1-score of both SVM and Logistic Regression models. Additionally, it will display a confusion matrix and sample predictions for visual evaluation of the model performance.
