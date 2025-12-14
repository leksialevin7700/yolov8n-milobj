"""
Fast threshold optimization for YOLOv8 - finds optimal conf/iou in minimal time.
Provides exact commands for validation re-runs.
"""
import argparse
from ultralytics import YOLO
import json
from pathlib import Path
from tqdm import tqdm


def optimize_thresholds(weights_path, data_yaml, imgsz=1280, fast_mode=True):
    """
    Find optimal confidence and IOU thresholds.
    
    Args:
        fast_mode: If True, tests fewer combinations (faster)
    """
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print(f"Model: {weights_path}")
    print(f"Data: {data_yaml}")
    print(f"Image size: {imgsz}")
    print(f"Mode: {'Fast' if fast_mode else 'Comprehensive'}")
    print("=" * 80)
    
    model = YOLO(weights_path)
    
    if fast_mode:
        # Fast mode: test key combinations only
        conf_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35]
        iou_thresholds = [0.40, 0.45, 0.50, 0.55]
    else:
        # Comprehensive mode
        conf_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
        iou_thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    
    best_map50 = 0.0
    best_map50_95 = 0.0
    best_conf = 0.25
    best_iou = 0.45
    best_results = {}
    all_results = []
    
    total = len(conf_thresholds) * len(iou_thresholds)
    print(f"\nTesting {total} threshold combinations...\n")
    
    for conf in tqdm(conf_thresholds, desc="Confidence"):
        for iou in iou_thresholds:
            try:
                metrics = model.val(
                    data=data_yaml,
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    max_det=300,
                    verbose=False
                )
                
                if hasattr(metrics, 'box'):
                    map50 = metrics.box.map50
                    map50_95 = metrics.box.map
                    precision = metrics.box.mp
                    recall = metrics.box.mr
                else:
                    results_dict = metrics.results_dict if hasattr(metrics, 'results_dict') else {}
                    map50 = results_dict.get('metrics/mAP50(B)', 0)
                    map50_95 = results_dict.get('metrics/mAP50-95(B)', 0)
                    precision = results_dict.get('metrics/precision(B)', 0)
                    recall = results_dict.get('metrics/recall(B)', 0)
                
                all_results.append({
                    'conf': conf,
                    'iou': iou,
                    'map50': float(map50),
                    'map50_95': float(map50_95),
                    'precision': float(precision),
                    'recall': float(recall)
                })
                
                # Track best (prioritize mAP50-95)
                if map50_95 > best_map50_95 or (map50_95 == best_map50_95 and map50 > best_map50):
                    best_map50 = map50
                    best_map50_95 = map50_95
                    best_conf = conf
                    best_iou = iou
                    best_results = {
                        'map50': float(map50),
                        'map50_95': float(map50_95),
                        'precision': float(precision),
                        'recall': float(recall)
                    }
            
            except Exception as e:
                print(f"\nError with conf={conf}, iou={iou}: {e}")
                continue
    
    # Print results
    print("\n" + "=" * 80)
    print("TOP 10 THRESHOLD COMBINATIONS")
    print("=" * 80)
    
    all_results.sort(key=lambda x: (x['map50_95'], x['map50']), reverse=True)
    
    for i, r in enumerate(all_results[:10], 1):
        marker = " ⭐ BEST" if r['conf'] == best_conf and r['iou'] == best_iou else ""
        print(f"{i:2d}. conf={r['conf']:.2f}, iou={r['iou']:.2f} | "
              f"mAP50={r['map50']:.4f}, mAP50-95={r['map50_95']:.4f}{marker}")
    
    print("\n" + "=" * 80)
    print("OPTIMAL THRESHOLDS")
    print("=" * 80)
    print(f"Confidence: {best_conf}")
    print(f"IOU: {best_iou}")
    print(f"mAP@50: {best_map50:.4f}")
    print(f"mAP@50-95: {best_map50_95:.4f}")
    print(f"Precision: {best_results['precision']:.4f}")
    print(f"Recall: {best_results['recall']:.4f}")
    print("=" * 80)
    
    # Generate exact commands
    print("\n" + "=" * 80)
    print("EXACT YOLOV8 COMMANDS")
    print("=" * 80)
    
    print("\n1. Validation with optimal thresholds:")
    print(f"   yolo detect val model={weights_path} data={data_yaml} imgsz={imgsz} conf={best_conf} iou={best_iou} max_det=300")
    
    print("\n2. Python validation:")
    print(f"   python quick_validate_optimized.py --weights {weights_path} --data {data_yaml} --conf {best_conf} --iou {best_iou} --imgsz {imgsz}")
    
    print("\n3. Inference with optimal thresholds:")
    print(f"   yolo detect predict model={weights_path} source=<image_dir> conf={best_conf} iou={best_iou} imgsz={imgsz}")
    
    # Save results
    output_file = Path('threshold_optimization_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'optimal': {
                'conf': best_conf,
                'iou': best_iou,
                'metrics': best_results
            },
            'all_results': all_results
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return best_conf, best_iou, best_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize inference thresholds')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset.yaml')
    parser.add_argument('--imgsz', type=int, default=1280, help='Image size')
    parser.add_argument('--comprehensive', action='store_true', help='Use comprehensive mode (slower)')
    args = parser.parse_args()
    
    optimize_thresholds(args.weights, args.data, args.imgsz, fast_mode=not args.comprehensive)

