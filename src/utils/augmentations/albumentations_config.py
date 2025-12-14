"""
Offline augmentation pipeline using Albumentations (useful if you want CLAHE and other transforms
applied before training). You can run this script to create an augmented copy of the train/images and train/labels.
"""
import albumentations as A
import cv2
import os
from pathlib import Path
import numpy as np


# Note: This script shows transforms. Applying transforms to bounding boxes and saving
# normalized YOLO labels requires careful handling (shown in example below if needed).


train_transform = A.Compose([
A.RandomResizedCrop(1280, 1280, scale=(0.8, 1.2), p=1.0),
A.HorizontalFlip(p=0.5),
A.OneOf([
A.RandomBrightnessContrast(p=0.6),
A.CLAHE(p=0.6),
], p=0.7),
A.GaussianBlur(blur_limit=(3,7), p=0.2),
A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT, p=0.4),
A.ColorJitter(p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


# Example usage (not auto-run): apply to one image + boxes
# augmented = train_transform(image=image, bboxes=bboxes, class_labels=labels)