"""
Helper wrappers for common YOLOv8 tasks: small functions to load model, run validation, and collect metrics.
"""
from ultralytics import YOLO
import json
from pathlib import Path




def load_model(weights: str):
    return YOLO(weights)




def run_val(weights: str, data_yaml: str, imgsz: int = 1280, batch: int = 8):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz, batch=batch)
    # metrics is returned as a dict-like object in many ultralytics versions
    # Save to file
    out = Path('runs') / 'val_metrics.json'
    with open(out, 'w') as f:
        json.dump(metrics, f, default=str)
    return metrics




# Weighted Box Fusion or ensembling could be added here.