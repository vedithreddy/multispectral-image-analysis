# Multispectral Image Analysis for Oral Disease Detection

This project focuses on detecting oral diseases using simulated multispectral image inputs. Starting with a standard RGB dataset, we generate synthetic versions that represent different wavelengths (VIBGYOR), and then train an EfficientNet-based model to classify the diseases based on these enhanced inputs.

## Dataset

We use the "Oral Diseases" dataset available on Kaggle:

Kaggle link: https://www.kaggle.com/datasets/salmansajid05/oral-diseases

The dataset includes images labeled by oral disease types such as:
- Gingivitis
- Calculus
- Tooth Discoloration
- Mouth Ulcer
- Dental Caries
- Hypodontia

## Synthetic Multispectral Data

Since actual multispectral imaging data isn't publicly available, we simulate it by applying transformations to RGB images to mimic seven wavelength bands: Violet, Indigo, Blue, Green, Yellow, Orange, and Red.

The notebook `generate_synthetic_vibgyor.ipynb` handles this step and outputs one synthetic image per wavelength for each original image.

## Model and Training

The notebook `effiecientnetv2.ipynb` contains the full pipeline:
- Preprocessing the 7-band synthetic image inputs
- Building and training an EfficientNet model using those inputs
- Evaluating the model using standard metrics like accuracy and confusion matrix
- Plotting training and validation performance over time

The model takes all seven synthetic wavelength bands as input, treating them as a 7-channel tensor.

## Features

- Converts RGB oral disease images into synthetic VIBGYOR-based multispectral data using a custom preprocessing pipeline.
- Uses an EfficientNet model adapted to handle 7-channel input representing different simulated wavelength bands.
- End-to-end training and evaluation pipeline, including accuracy tracking, confusion matrix generation, and performance visualization.
- Notebooks are optimized for easy execution on Google Colab with GPU support—no additional configuration needed.

## Requirements(for jupyter or any local env)

To install the required packages:

```bash
pip install -r requirements.txt
