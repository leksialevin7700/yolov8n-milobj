"""
Optimized YOLOv8 training script for fast retraining (≤35 epochs).
Focuses on maximizing mAP with minimal training time.
"""
import argparse
import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml


def main():
    parser = argparse.ArgumentParser(description='Fast YOLOv8 training (optimized for speed + accuracy)')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset.yaml')
    parser.add_argument('--weights', type=str, default='yolov8m.pt', 
                       help='Pretrained weights (yolov8n/s/m/l/x)')
    parser.add_argument('--epochs', type=int, default=35, help='Max epochs (default: 35)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size (default: 640)')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda:0)')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project directory')
    parser.add_argument('--name', type=str, default='milobj_fast', help='Run name')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load model
    model = YOLO(args.weights)
    
    print("=" * 80)
    print("FAST YOLOV8 TRAINING - OPTIMIZED CONFIGURATION")
    print("=" * 80)
    print(f"Data: {args.data}")
    print(f"Model: {args.weights}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Device: {args.device}")
    print(f"Early stopping patience: {args.patience}")
    print("=" * 80)
    
    # Optimized training parameters
    # Key optimizations:
    # - lr0: Higher initial LR for faster convergence
    # - lrf: Lower final LR (0.01) for cosine annealing
    # - momentum: Standard 0.937
    # - weight_decay: 0.0005 for regularization
    # - warmup_epochs: 3 for stable start
    # - box: 7.5 (balanced box loss)
    # - cls: 0.5 (lower cls weight to handle validation issues)
    # - dfl: 1.5 (distribution focal loss)
    # - hsv_h/s/v: Moderate augmentation
    # - degrees: Rotation augmentation
    # - translate: Translation augmentation
    # - scale: Scale augmentation
    # - fliplr: Horizontal flip
    # - mosaic: 1.0 (full mosaic)
    # - mixup: 0.1 (light mixup)
    # - copy_paste: 0.1 (light copy-paste)
    
    try:
        results = model.train(
            data=str(Path(args.data).resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            patience=args.patience,
            resume=args.resume,
            
            # Learning rate (optimized for fast convergence)
            lr0=0.01,           # Initial learning rate (higher for faster start)
            lrf=0.01,            # Final learning rate (cosine annealing to 1% of lr0)
            
            # Optimizer
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            
            # Loss weights (adjusted for validation issues)
            box=7.5,             # Box loss gain
            cls=0.5,             # Class loss gain (lowered due to validation issues)
            dfl=1.5,             # DFL loss gain
            
            # Augmentation (balanced for speed + generalization)
            hsv_h=0.015,         # Image HSV-Hue augmentation
            hsv_s=0.7,           # Image HSV-Saturation augmentation
            hsv_v=0.4,           # Image HSV-Value augmentation
            degrees=10.0,        # Image rotation (+/- deg)
            translate=0.1,       # Image translation (+/- fraction)
            scale=0.5,           # Image scale (+/- gain)
            shear=2.0,           # Image shear (+/- deg)
            perspective=0.0,     # Image perspective (+/- fraction)
            flipud=0.0,          # Image flip up-down (probability)
            fliplr=0.5,          # Image flip left-right (probability)
            mosaic=1.0,          # Image mosaic (probability)
            mixup=0.1,           # Image mixup (probability)
            copy_paste=0.1,      # Segment copy-paste (probability)
            
            # Training settings
            optimizer='AdamW',   # AdamW optimizer (faster convergence than SGD)
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=False,
            rect=False,
            cos_lr=True,         # Cosine LR scheduler (critical for fast training)
            close_mosaic=10,     # Disable mosaic in last 10 epochs
            
            # Validation
            val=True,
            plots=True,
            save=True,
            save_period=10,      # Save checkpoint every 10 epochs
            
            # Model settings
            amp=True,            # Automatic Mixed Precision (faster training)
            fraction=1.0,        # Dataset fraction to use
            profile=False,
            freeze=None,
            multi_scale=False,   # Disable multi-scale for speed
        )
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Results saved to: {results.save_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()

