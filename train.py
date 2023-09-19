"""
train.py

This module was used to train a YOLOv8 model on the license plate-vehicle
dataset (https://universe.roboflow.com/samrat-sahoo/license-plates-f8vsn)
for 100 epochs using the Ultralytics library.

Author: Jacob Pitsenberger
Date: 9/19/23

Usage:
    - Ensure the 'yolov8n.pt' pre-trained model is available in the working directory.
    - Use the 'model.train' method to train the YOLOv5 model on the dataset specified in the 'config.yaml' file.
"""

from ultralytics import YOLO

# Load pretrained model.
model = YOLO('yolov8n.pt')

# Train over dataset for 100 epochs.
model.train(data="config.yaml", epochs=100)

