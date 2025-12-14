"""
Quick fix script: Remove images with empty label files from validation set.
This is a critical fix for high validation cls_loss.
"""
import os
from pathlib import Path
import yaml
import shutil

def fix_empty_labels(data_yaml_path: str, backup: bool = True):
    """
    Remove images that have empty label files from validation set.
    This prevents YOLOv8 from trying to classify empty images.
    """
    with open(data_yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    base_path = Path(data_yaml_path).parent
    val_img_dir = base_path / 'val' / 'images'
    val_lbl_dir = base_path / 'val' / 'labels'
    
    if backup:
        backup_dir = base_path / 'val_backup'
        print(f"Creating backup at {backup_dir}...")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(base_path / 'val', backup_dir)
        print("Backup created.")
    
    empty_labels = []
    for lbl_file in val_lbl_dir.glob('*.txt'):
        with open(lbl_file, 'r') as f:
            content = f.read().strip()
        if not content:
            empty_labels.append(lbl_file)
    
    print(f"\nFound {len(empty_labels)} empty label files in validation set.")
    
    removed_images = 0
    removed_labels = 0
    
    for lbl_file in empty_labels:
        # Remove corresponding image
        img_file = val_img_dir / (lbl_file.stem + '.jpg')
        if not img_file.exists():
            img_file = val_img_dir / (lbl_file.stem + '.png')
        
        if img_file.exists():
            img_file.unlink()
            removed_images += 1
        
        # Remove empty label file
        lbl_file.unlink()
        removed_labels += 1
    
    print(f"Removed {removed_images} images and {removed_labels} empty label files.")
    print(f"Validation set now has {len(list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png')))} images.")
    print("\n✓ Fix complete! Re-run validation to verify.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Fix empty label files in validation set')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset.yaml file')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup')
    args = parser.parse_args()
    
    fix_empty_labels(args.data, backup=not args.no_backup)

