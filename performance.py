"""
Comprehensive class-wise performance analysis for YOLOv8.
Generates detailed metrics and identifies problematic classes.
"""
import argparse
import json
import yaml
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from collections import Counter


def analyze_class_performance(weights_path, data_yaml, output_dir='analysis'):
    """
    Analyze per-class performance and generate detailed report.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load dataset info
    with open(data_yaml, 'r') as f:
        cfg = yaml.safe_load(f)
    class_names = cfg.get('names', {})
    
    print("=" * 80)
    print("CLASS-WISE PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    # Load model and run validation
    model = YOLO(weights_path)
    
    print("\nRunning validation...")
    metrics = model.val(
        data=data_yaml,
        imgsz=1280,
        conf=0.25,
        iou=0.45,
        max_det=300,
        verbose=True
    )
    
    # Extract metrics
    if hasattr(metrics, 'box'):
        overall_map50 = metrics.box.map50
        overall_map50_95 = metrics.box.map
        overall_precision = metrics.box.mp
        overall_recall = metrics.box.mr
        per_class_ap50 = metrics.box.maps50.tolist() if hasattr(metrics.box, 'maps50') else []
        per_class_ap50_95 = metrics.box.maps.tolist() if hasattr(metrics.box, 'maps') else []
    else:
        results_dict = metrics.results_dict if hasattr(metrics, 'results_dict') else {}
        overall_map50 = results_dict.get('metrics/mAP50(B)', 0)
        overall_map50_95 = results_dict.get('metrics/mAP50-95(B)', 0)
        overall_precision = results_dict.get('metrics/precision(B)', 0)
        overall_recall = results_dict.get('metrics/recall(B)', 0)
        per_class_ap50 = results_dict.get('metrics/mAP50(B)', [])
        per_class_ap50_95 = results_dict.get('metrics/mAP50-95(B)', [])
    
    # Count samples per class
    base_path = Path(data_yaml).parent
    train_labels = list((base_path / 'train' / 'labels').glob('*.txt'))
    val_labels = list((base_path / 'val' / 'labels').glob('*.txt'))
    
    train_counts = Counter()
    val_counts = Counter()
    
    for lbl_file in train_labels:
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    train_counts[cls_id] += 1
    
    for lbl_file in val_labels:
        with open(lbl_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    val_counts[cls_id] += 1
    
    # Create detailed report
    report_data = []
    for cls_id in range(len(class_names)):
        cls_name = class_names.get(cls_id, f"class_{cls_id}")
        ap50 = per_class_ap50[cls_id] if cls_id < len(per_class_ap50) else 0.0
        ap50_95 = per_class_ap50_95[cls_id] if cls_id < len(per_class_ap50_95) else 0.0
        train_count = train_counts.get(cls_id, 0)
        val_count = val_counts.get(cls_id, 0)
        total_count = train_count + val_count
        
        # Categorize performance
        if ap50_95 >= 0.7:
            performance = "Excellent"
        elif ap50_95 >= 0.5:
            performance = "Good"
        elif ap50_95 >= 0.3:
            performance = "Fair"
        elif ap50_95 >= 0.1:
            performance = "Poor"
        else:
            performance = "Very Poor"
        
        # Categorize sample count
        if total_count < 10:
            sample_status = "Critically Low"
        elif total_count < 50:
            sample_status = "Low"
        elif total_count < 200:
            sample_status = "Moderate"
        else:
            sample_status = "Adequate"
        
        report_data.append({
            'Class ID': cls_id,
            'Class Name': cls_name,
            'AP@50': round(ap50, 4),
            'AP@50-95': round(ap50_95, 4),
            'Train Samples': train_count,
            'Val Samples': val_count,
            'Total Samples': total_count,
            'Performance': performance,
            'Sample Status': sample_status
        })
    
    # Create DataFrame
    df = pd.DataFrame(report_data)
    df = df.sort_values('AP@50-95', ascending=False)
    
    # Save to CSV
    csv_path = output_dir / 'class_performance.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved class performance to {csv_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    print(f"mAP@50: {overall_map50:.4f}")
    print(f"mAP@50-95: {overall_map50_95:.4f}")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    
    print("\n" + "=" * 80)
    print("TOP 5 PERFORMING CLASSES")
    print("=" * 80)
    top5 = df.head(5)
    for _, row in top5.iterrows():
        print(f"{row['Class Name']:25s} | AP@50: {row['AP@50']:.4f} | AP@50-95: {row['AP@50-95']:.4f} | Samples: {row['Total Samples']}")
    
    print("\n" + "=" * 80)
    print("BOTTOM 5 PERFORMING CLASSES")
    print("=" * 80)
    bottom5 = df.tail(5)
    for _, row in bottom5.iterrows():
        print(f"{row['Class Name']:25s} | AP@50: {row['AP@50']:.4f} | AP@50-95: {row['AP@50-95']:.4f} | Samples: {row['Total Samples']}")
    
    print("\n" + "=" * 80)
    print("CLASSES WITH CRITICALLY LOW SAMPLES (<10)")
    print("=" * 80)
    low_samples = df[df['Sample Status'] == 'Critically Low']
    if len(low_samples) > 0:
        for _, row in low_samples.iterrows():
            print(f"{row['Class Name']:25s} | Samples: {row['Total Samples']} | AP@50-95: {row['AP@50-95']:.4f}")
    else:
        print("None")
    
    # Save JSON summary
    summary = {
        'overall_metrics': {
            'mAP50': float(overall_map50),
            'mAP50_95': float(overall_map50_95),
            'precision': float(overall_precision),
            'recall': float(overall_recall)
        },
        'per_class_metrics': report_data,
        'recommendations': {
            'remove_classes': [row['Class Name'] for _, row in df.iterrows() if row['Sample Status'] == 'Critically Low' and row['AP@50-95'] < 0.1],
            'strong_classes': [row['Class Name'] for _, row in df.iterrows() if row['AP@50-95'] >= 0.7]
        }
    }
    
    json_path = output_dir / 'class_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved detailed analysis to {json_path}")
    
    return summary, df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze class-wise performance')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset.yaml')
    parser.add_argument('--output', type=str, default='analysis', help='Output directory')
    args = parser.parse_args()
    
    analyze_class_performance(args.weights, args.data, args.output)

