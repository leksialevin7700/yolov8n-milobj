"""
Find optimal confidence and IOU thresholds for your trained model.
This can improve mAP by 2-5% without retraining.
"""
import argparse
from ultralytics import YOLO
from pathlib import Path
import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description='Find optimal inference thresholds')
    p.add_argument('--weights', type=str, required=True, help='Path to trained model')
    p.add_argument('--data', type=str, required=True, help='Path to dataset.yaml')
    p.add_argument('--imgsz', type=int, default=1280, help='Image size')
    return p.parse_args()


def grid_search_thresholds(model, data_yaml, imgsz=1280):
    """
    Grid search for optimal conf and iou thresholds.
    Tests multiple combinations and returns best.
    """
    print("=" * 80)
    print("FINDING OPTIMAL INFERENCE THRESHOLDS")
    print("=" * 80)
    
    # Test ranges
    conf_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    iou_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    
    best_map50 = 0.0
    best_map50_95 = 0.0
    best_conf = 0.25
    best_iou = 0.45
    best_results = {}
    
    total_combinations = len(conf_thresholds) * len(iou_thresholds)
    print(f"Testing {total_combinations} threshold combinations...\n")
    
    results = []
    
    for conf in tqdm(conf_thresholds, desc="Confidence thresholds"):
        for iou in iou_thresholds:
            try:
                # Run validation with these thresholds
                metrics = model.val(
                    data=data_yaml,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    max_det=300,
                    verbose=False
                )
                
                map50 = metrics.box.map50 if hasattr(metrics.box, 'map50') else metrics.results_dict.get('metrics/mAP50(B)', 0)
                map50_95 = metrics.box.map if hasattr(metrics.box, 'map') else metrics.results_dict.get('metrics/mAP50-95(B)', 0)
                
                results.append({
                    'conf': conf,
                    'iou': iou,
                    'map50': map50,
                    'map50_95': map50_95
                })
                
                # Track best (prioritize mAP50-95, then mAP50)
                if map50_95 > best_map50_95 or (map50_95 == best_map50_95 and map50 > best_map50):
                    best_map50 = map50
                    best_map50_95 = map50_95
                    best_conf = conf
                    best_iou = iou
                    best_results = {
                        'map50': map50,
                        'map50_95': map50_95,
                        'precision': metrics.results_dict.get('metrics/precision(B)', 0),
                        'recall': metrics.results_dict.get('metrics/recall(B)', 0)
                    }
                
            except Exception as e:
                print(f"Error with conf={conf}, iou={iou}: {e}")
                continue
    
    # Print results
    print("\n" + "=" * 80)
    print("TOP 10 THRESHOLD COMBINATIONS")
    print("=" * 80)
    
    # Sort by mAP50-95, then mAP50
    results.sort(key=lambda x: (x['map50_95'], x['map50']), reverse=True)
    
    for i, r in enumerate(results[:10], 1):
        marker = " ⭐ BEST" if r['conf'] == best_conf and r['iou'] == best_iou else ""
        print(f"{i:2d}. conf={r['conf']:.2f}, iou={r['iou']:.2f} -> "
              f"mAP50={r['map50']:.4f}, mAP50-95={r['map50_95']:.4f}{marker}")
    
    print("\n" + "=" * 80)
    print("RECOMMENDED SETTINGS")
    print("=" * 80)
    print(f"Confidence threshold: {best_conf}")
    print(f"IOU threshold: {best_iou}")
    print(f"mAP@50: {best_map50:.4f}")
    print(f"mAP@50-95: {best_map50_95:.4f}")
    print(f"Precision: {best_results.get('precision', 0):.4f}")
    print(f"Recall: {best_results.get('recall', 0):.4f}")
    print("=" * 80)
    
    print(f"\nUse these in inference:")
    print(f"  --conf {best_conf} --iou {best_iou}")
    
    return best_conf, best_iou, best_results


if __name__ == '__main__':
    args = parse_args()
    model = YOLO(args.weights)
    best_conf, best_iou, results = grid_search_thresholds(model, args.data, args.imgsz)

