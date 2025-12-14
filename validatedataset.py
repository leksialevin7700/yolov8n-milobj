"""
Fast dataset validation script for YOLOv8 - detects high-impact issues.
Focuses on validation set issues that cause high cls_loss.
"""
import os
from pathlib import Path
from collections import Counter
import yaml
import cv2

def validate_dataset(data_yaml_path: str, fix_issues: bool = False):
    """
    Validate YOLO dataset and report critical issues.
    
    Args:
        data_yaml_path: Path to dataset.yaml
        fix_issues: If True, auto-fix issues where possible (remove empty labels, etc.)
    """
    # Load dataset config
    with open(data_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    base_path = Path(data_yaml_path).parent
    num_classes = cfg['nc']
    class_names = cfg.get('names', {})
    
    print("=" * 80)
    print("YOLO DATASET VALIDATION REPORT")
    print("=" * 80)
    print(f"Dataset: {data_yaml_path}")
    print(f"Classes: {num_classes}")
    print(f"Class names: {list(class_names.values())}")
    print("=" * 80)
    
    issues = {
        'empty_labels': [],
        'invalid_class_ids': [],
        'invalid_bbox_coords': [],
        'missing_labels': [],
        'bbox_out_of_bounds': [],
        'invalid_format': []
    }
    
    stats = {
        'train': {'images': 0, 'labels': 0, 'objects': 0, 'class_dist': Counter()},
        'val': {'images': 0, 'labels': 0, 'objects': 0, 'class_dist': Counter()}
    }
    
    # Validate each split
    for split in ['train', 'val']:
        print(f"\n[{split.upper()}]")
        print("-" * 80)
        
        img_dir = base_path / split / 'images'
        lbl_dir = base_path / split / 'labels'
        
        if not img_dir.exists():
            print(f"ERROR: {img_dir} does not exist!")
            continue
        if not lbl_dir.exists():
            print(f"ERROR: {lbl_dir} does not exist!")
            continue
        
        # Get all images and labels
        image_files = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))
        label_files = sorted(lbl_dir.glob('*.txt'))
        
        stats[split]['images'] = len(image_files)
        stats[split]['labels'] = len(label_files)
        
        print(f"Images: {len(image_files)}")
        print(f"Label files: {len(label_files)}")
        
        # Check for missing labels
        image_stems = {f.stem for f in image_files}
        label_stems = {f.stem for f in label_files}
        missing_labels = image_stems - label_stems
        
        if missing_labels:
            print(f"⚠️  WARNING: {len(missing_labels)} images missing label files!")
            issues['missing_labels'].extend([(split, str(s)) for s in list(missing_labels)[:10]])
            if len(missing_labels) > 10:
                print(f"   (showing first 10)")
        
        # Validate each label file
        empty_count = 0
        invalid_class_count = 0
        invalid_bbox_count = 0
        out_of_bounds_count = 0
        format_error_count = 0
        
        for lbl_file in label_files:
            try:
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()
                
                # Check for empty file
                if not lines or all(not line.strip() for line in lines):
                    empty_count += 1
                    issues['empty_labels'].append((split, str(lbl_file.name)))
                    if fix_issues:
                        # Optionally create placeholder or remove
                        continue
                
                # Validate each annotation line
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        format_error_count += 1
                        issues['invalid_format'].append((split, str(lbl_file.name), line_num, line))
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                    except ValueError:
                        format_error_count += 1
                        issues['invalid_format'].append((split, str(lbl_file.name), line_num, line))
                        continue
                    
                    # Check class ID
                    if class_id < 0 or class_id >= num_classes:
                        invalid_class_count += 1
                        issues['invalid_class_ids'].append((split, str(lbl_file.name), line_num, class_id))
                        continue
                    
                    stats[split]['objects'] += 1
                    stats[split]['class_dist'][class_id] += 1
                    
                    # Check bbox normalization (should be 0-1)
                    if x_center < 0 or x_center > 1 or y_center < 0 or y_center > 1:
                        out_of_bounds_count += 1
                        issues['bbox_out_of_bounds'].append((split, str(lbl_file.name), line_num, 
                                                             f"center=({x_center:.4f}, {y_center:.4f})"))
                    
                    if width < 0 or width > 1 or height < 0 or height > 1:
                        out_of_bounds_count += 1
                        issues['bbox_out_of_bounds'].append((split, str(lbl_file.name), line_num,
                                                             f"size=({width:.4f}, {height:.4f})"))
                    
                    # Check if bbox extends beyond image (x_center ± width/2, y_center ± height/2)
                    if (x_center - width/2) < 0 or (x_center + width/2) > 1:
                        invalid_bbox_count += 1
                        issues['invalid_bbox_coords'].append((split, str(lbl_file.name), line_num,
                                                              f"x_center={x_center:.4f}, width={width:.4f}"))
                    
                    if (y_center - height/2) < 0 or (y_center + height/2) > 1:
                        invalid_bbox_count += 1
                        issues['invalid_bbox_coords'].append((split, str(lbl_file.name), line_num,
                                                              f"y_center={y_center:.4f}, height={height:.4f}"))
                    
                    # Check for zero or negative dimensions
                    if width <= 0 or height <= 0:
                        invalid_bbox_count += 1
                        issues['invalid_bbox_coords'].append((split, str(lbl_file.name), line_num,
                                                              f"zero/neg size: w={width:.4f}, h={height:.4f}"))
            
            except Exception as e:
                format_error_count += 1
                issues['invalid_format'].append((split, str(lbl_file.name), f"ERROR: {str(e)}"))
        
        # Print summary for this split
        print(f"\nValidation Results:")
        print(f"  ✓ Valid objects: {stats[split]['objects']}")
        print(f"  ⚠️  Empty label files: {empty_count}")
        print(f"  ⚠️  Invalid class IDs: {invalid_class_count}")
        print(f"  ⚠️  Invalid bbox coordinates: {invalid_bbox_count}")
        print(f"  ⚠️  Bbox out of bounds (not 0-1): {out_of_bounds_count}")
        print(f"  ⚠️  Format errors: {format_error_count}")
        
        # Class distribution
        if stats[split]['class_dist']:
            print(f"\n  Class distribution:")
            for cls_id in sorted(stats[split]['class_dist'].keys()):
                cls_name = class_names.get(cls_id, f"class_{cls_id}")
                count = stats[split]['class_dist'][cls_id]
                print(f"    {cls_id:2d} ({cls_name:25s}): {count:5d}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("CRITICAL ISSUES SUMMARY")
    print("=" * 80)
    
    total_empty = len([x for x in issues['empty_labels'] if x[0] == 'val'])
    total_invalid_class = len([x for x in issues['invalid_class_ids'] if x[0] == 'val'])
    total_invalid_bbox = len([x for x in issues['invalid_bbox_coords'] if x[0] == 'val'])
    
    print(f"\nVALIDATION SET (High Priority - causes high cls_loss):")
    print(f"  🔴 Empty label files: {total_empty}")
    print(f"  🔴 Invalid class IDs: {total_invalid_class}")
    print(f"  🔴 Invalid bbox coordinates: {total_invalid_bbox}")
    
    if total_empty > 0:
        print(f"\n  Sample empty validation labels (first 10):")
        for split, fname in issues['empty_labels'][:10]:
            if split == 'val':
                print(f"    - {fname}")
    
    if total_invalid_class > 0:
        print(f"\n  Sample invalid class IDs (first 10):")
        for item in issues['invalid_class_ids'][:10]:
            if item[0] == 'val':
                print(f"    - {item[1]}: line {item[2]}, class_id={item[3]} (valid: 0-{num_classes-1})")
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if total_empty > 0:
        print(f"1. FIX: Remove or label {total_empty} empty validation label files")
        print("   Command: python fix_empty_labels.py --data data/military_object_dataset/military_dataset.yaml")
    
    if total_invalid_class > 0:
        print(f"2. FIX: Correct {total_invalid_class} invalid class IDs in validation set")
    
    if total_invalid_bbox > 0:
        print(f"3. FIX: Correct {total_invalid_bbox} invalid bbox coordinates")
    
    if total_empty == 0 and total_invalid_class == 0 and total_invalid_bbox == 0:
        print("✓ No critical issues found in validation set!")
    
    return issues, stats


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Validate YOLO dataset')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to dataset.yaml file')
    parser.add_argument('--fix', action='store_true',
                       help='Auto-fix issues where possible')
    args = parser.parse_args()
    
    validate_dataset(args.data, fix_issues=args.fix)

