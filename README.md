# Food Spoilage Detection Using Deep Learning

## Author

ANIL KUMAR REDDY VANTERU

## Project Overview

This project detects early spoilage in packaged food using deep learning. It analyzes subtle texture patterns from images and predicts whether food is spoiled or not. The system compares multiple models including a CNN baseline, ResNet, ResNet with attention, and a Vision Transformer. The goal is to identify spoilage before visible signs appear and evaluate which model performs best.

## Project Structure

food_spoilage_detection/

README.md
requirements.txt
config.yaml
train_all_models.py
evaluate.py
inference.py

models/
cnn.py
resnet.py
resnet_attention.py
transformer.py

data/
dataset.py
transforms.py
images/
fresh/
spoiled/

utils/
metrics.py
plots.py
logger.py

experiments/
run_experiments.py

outputs/
models/
plots/
results.csv

## Setup Instructions

Make sure Python 3.8 or above is installed. It is better to create a virtual environment before installing dependencies.

Create virtual environment
python -m venv venv

Activate environment
For Windows
venv\Scripts\activate

For Mac/Linux
source venv/bin/activate

Install required libraries
pip install -r requirements.txt

## Dependencies

The project uses the following main libraries

PyTorch for deep learning models
Torchvision for pretrained models and transforms
NumPy and Pandas for data handling
Matplotlib and Seaborn for visualizations
Scikit-learn for evaluation metrics
Pillow for image processing
PyYAML for configuration handling

## Dataset Setup

Inside the data folder, create an images folder.
Inside images, create two folders

fresh → contains fresh food images
spoiled → contains spoiled food images

Example

data/images/fresh/
data/images/spoiled/

## How to Run

Train all models (CNN, ResNet, Attention, Transformer)
python train_all_models.py

Run experiments and generate comparison graphs
python experiments/run_experiments.py

View evaluation results in table format
python evaluate.py

Run inference on a single image
python inference.py

## Outputs

Trained models are saved in
outputs/models/

Visual plots are saved in
outputs/plots/

Generated outputs include

confusion matrix for each model
model comparison accuracy chart
failure case images
performance trends

All model performance metrics are saved in
outputs/results.csv

## Key Features

This project includes multiple deep learning models for comparison
It performs ablation study using ResNet with and without attention
It includes a Transformer model as required
It generates real evaluation metrics instead of dummy values
It produces multiple visual outputs for analysis
It supports reproducibility using config file

## Notes

Images should be clear and consistent for better results
Training time depends on system performance
GPU is recommended but CPU also works
Make sure dataset is properly structured before running

## Summary

This project provides a complete pipeline from data loading to model comparison and evaluation. It helps in understanding how different deep learning models perform for early spoilage detection and gives visual insights into results.
