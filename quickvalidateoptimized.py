"""
Quick validation with optimized thresholds - no retraining needed.
"""
import argparse
from ultralytics import YOLO
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Quick validation with optimized thresholds')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset.yaml')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image size')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='IOU threshold')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("QUICK VALIDATION WITH OPTIMIZED THRESHOLDS")
    print("=" * 80)
    print(f"Model: {args.weights}")
    print(f"Data: {args.data}")
    print(f"Image size: {args.imgsz}")
    print(f"Confidence: {args.conf}")
    print(f"IOU: {args.iou}")
    print("=" * 80)
    
    model = YOLO(args.weights)
    
    # Run validation
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=300,
        verbose=True
    )
    
    # Print results
    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    
    if hasattr(metrics, 'box'):
        print(f"mAP@50: {metrics.box.map50:.4f}")
        print(f"mAP@50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")
    else:
        # Fallback for different ultralytics versions
        results_dict = metrics.results_dict if hasattr(metrics, 'results_dict') else {}
        print(f"mAP@50: {results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"mAP@50-95: {results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"Precision: {results_dict.get('metrics/precision(B)', 'N/A')}")
        print(f"Recall: {results_dict.get('metrics/recall(B)', 'N/A')}")
    
    print("=" * 80)


if __name__ == '__main__':
    main()

