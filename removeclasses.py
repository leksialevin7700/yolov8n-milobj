"""
Remove severely underrepresented classes from dataset to improve overall mAP.
Creates a new dataset YAML with only well-represented classes.
"""
import argparse
import yaml
from pathlib import Path
from collections import Counter
import shutil


def remove_classes(data_yaml_path, classes_to_remove, min_samples=10, output_yaml=None):
    """
    Remove specified classes from dataset and create new YAML.
    
    Args:
        data_yaml_path: Path to original dataset.yaml
        classes_to_remove: List of class names or IDs to remove
        min_samples: Minimum samples threshold (classes below this will be auto-removed)
        output_yaml: Output YAML path (default: adds '_filtered' suffix)
    """
    # Load original config
    with open(data_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    base_path = Path(data_yaml_path).parent
    class_names = cfg.get('names', {})
    
    # Count samples per class
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
    
    # Identify classes to remove
    classes_to_remove_ids = set()
    
    # Add explicitly specified classes
    for cls_spec in classes_to_remove:
        if isinstance(cls_spec, str):
            # Find by name
            for cls_id, name in class_names.items():
                if name == cls_spec:
                    classes_to_remove_ids.add(cls_id)
                    break
        else:
            # Assume it's an ID
            classes_to_remove_ids.add(int(cls_spec))
    
    # Auto-remove classes below min_samples threshold
    for cls_id in range(len(class_names)):
        total_samples = train_counts.get(cls_id, 0) + val_counts.get(cls_id, 0)
        if total_samples < min_samples:
            classes_to_remove_ids.add(cls_id)
            print(f"Auto-removing {class_names.get(cls_id, f'class_{cls_id}')} (only {total_samples} samples)")
    
    if not classes_to_remove_ids:
        print("No classes to remove. All classes meet the minimum sample threshold.")
        return
    
    # Create new class mapping
    old_to_new = {}
    new_class_names = {}
    new_id = 0
    
    for old_id in range(len(class_names)):
        if old_id not in classes_to_remove_ids:
            old_to_new[old_id] = new_id
            new_class_names[new_id] = class_names[old_id]
            new_id += 1
    
    print(f"\nRemoving {len(classes_to_remove_ids)} classes:")
    for cls_id in sorted(classes_to_remove_ids):
        print(f"  - {class_names.get(cls_id, f'class_{cls_id}')} (ID: {cls_id})")
    
    print(f"\nKeeping {len(new_class_names)} classes:")
    for new_id, name in new_class_names.items():
        print(f"  - {name} (new ID: {new_id})")
    
    # Create filtered dataset directories
    filtered_base = base_path / 'filtered'
    for split in ['train', 'val']:
        (filtered_base / split / 'images').mkdir(parents=True, exist_ok=True)
        (filtered_base / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process labels and copy images
    total_removed = 0
    total_kept = 0
    
    for split in ['train', 'val']:
        label_dir = base_path / split / 'labels'
        image_dir = base_path / split / 'images'
        filtered_label_dir = filtered_base / split / 'labels'
        filtered_image_dir = filtered_base / split / 'images'
        
        for lbl_file in label_dir.glob('*.txt'):
            new_lines = []
            with open(lbl_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_cls_id = int(parts[0])
                    if old_cls_id not in classes_to_remove_ids:
                        new_cls_id = old_to_new[old_cls_id]
                        new_line = f"{new_cls_id} {' '.join(parts[1:])}\n"
                        new_lines.append(new_line)
            
            # Only keep files with at least one annotation
            if new_lines:
                # Copy label file
                new_lbl_file = filtered_label_dir / lbl_file.name
                with open(new_lbl_file, 'w') as f:
                    f.writelines(new_lines)
                
                # Copy corresponding image
                img_extensions = ['.jpg', '.jpeg', '.png']
                img_file = None
                for ext in img_extensions:
                    img_path = image_dir / (lbl_file.stem + ext)
                    if img_path.exists():
                        img_file = img_path
                        break
                
                if img_file and img_file.exists():
                    shutil.copy2(img_file, filtered_image_dir / img_file.name)
                    total_kept += 1
            else:
                total_removed += 1
    
    print(f"\n✓ Filtered dataset created:")
    print(f"  - Kept: {total_kept} images with annotations")
    print(f"  - Removed: {total_removed} images (no annotations after filtering)")
    
    # Create new YAML
    if output_yaml is None:
        output_yaml = base_path / 'military_dataset_filtered.yaml'
    else:
        output_yaml = Path(output_yaml)
    
    new_cfg = {
        'path': str(filtered_base.relative_to(filtered_base.parent)),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(new_class_names),
        'names': new_class_names
    }
    
    with open(output_yaml, 'w') as f:
        yaml.dump(new_cfg, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Created filtered dataset YAML: {output_yaml}")
    print(f"\nTo use the filtered dataset:")
    print(f"  python train_yolov8.py --data {output_yaml}")
    
    return output_yaml


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove underrepresented classes from dataset')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset.yaml')
    parser.add_argument('--remove', type=str, nargs='+', default=[], 
                       help='Class names or IDs to remove (e.g., trench civilian)')
    parser.add_argument('--min-samples', type=int, default=10,
                       help='Auto-remove classes with fewer than this many samples')
    parser.add_argument('--output', type=str, default=None,
                       help='Output YAML path (default: adds _filtered suffix)')
    args = parser.parse_args()
    
    remove_classes(args.data, args.remove, args.min_samples, args.output)

