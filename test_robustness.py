"""
Comprehensive robustness testing for YOLOv8 model.
Tests performance under varied lighting, occlusion, and scale conditions.
Addresses 25% of judging criteria.
"""
import argparse
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import albumentations as A


def apply_lighting_variations(image):
    """Apply different lighting conditions to test robustness."""
    variations = {}
    
    # Brightness variations
    bright = cv2.convertScaleAbs(image, alpha=1.5, beta=30)  # +50% brighter
    dark = cv2.convertScaleAbs(image, alpha=0.6, beta=-30)  # -40% darker
    
    # Contrast variations
    high_contrast = cv2.convertScaleAbs(image, alpha=1.8, beta=0)  # High contrast
    low_contrast = cv2.convertScaleAbs(image, alpha=0.7, beta=0)  # Low contrast
    
    # Color temperature variations (warm/cool)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_warm = hsv.copy()
    hsv_warm[:, :, 0] = np.clip(hsv_warm[:, :, 0] + 10, 0, 179)  # Warmer (more yellow)
    warm = cv2.cvtColor(hsv_warm, cv2.COLOR_HSV2BGR)
    
    hsv_cool = hsv.copy()
    hsv_cool[:, :, 0] = np.clip(hsv_cool[:, :, 0] - 10, 0, 179)  # Cooler (more blue)
    cool = cv2.cvtColor(hsv_cool, cv2.COLOR_HSV2BGR)
    
    variations = {
        'original': image,
        'bright': bright,
        'dark': dark,
        'high_contrast': high_contrast,
        'low_contrast': low_contrast,
        'warm': warm,
        'cool': cool
    }
    
    return variations


def apply_occlusion(image, occlusion_level=0.3):
    """Apply synthetic occlusion to test robustness."""
    h, w = image.shape[:2]
    variations = {}
    
    # Random patches occlusion
    occluded = image.copy()
    num_patches = int(occlusion_level * 10)
    for _ in range(num_patches):
        patch_size = int(min(h, w) * 0.1)
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)
        occluded[y:y+patch_size, x:x+patch_size] = 0  # Black patch
    
    # Center occlusion (simulates partial blocking)
    center_occluded = image.copy()
    center_h, center_w = h // 2, w // 2
    block_size = int(min(h, w) * occlusion_level)
    y1 = center_h - block_size // 2
    y2 = center_h + block_size // 2
    x1 = center_w - block_size // 2
    x2 = center_w + block_size // 2
    center_occluded[y1:y2, x1:x2] = 128  # Gray block
    
    variations = {
        'original': image,
        'random_occlusion_30pct': occluded,
        'center_occlusion_30pct': center_occluded
    }
    
    return variations


def apply_scale_variations(image, base_size=1280):
    """Apply different scales to test robustness."""
    h, w = image.shape[:2]
    variations = {}
    
    scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    
    for scale in scales:
        new_w = int(w * scale)
        new_h = int(h * scale)
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Resize back to original size for consistent evaluation
        resized_back = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
        variations[f'scale_{scale:.2f}'] = resized_back
    
    variations['original'] = image
    
    return variations


def test_robustness(weights_path, test_images_dir, output_dir='robustness_analysis', num_samples=50):
    """
    Comprehensive robustness testing under varied conditions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("ROBUSTNESS TESTING")
    print("=" * 80)
    print(f"Model: {weights_path}")
    print(f"Test images: {test_images_dir}")
    print(f"Testing {num_samples} samples")
    print("=" * 80)
    
    model = YOLO(weights_path)
    
    # Get test images
    test_dir = Path(test_images_dir)
    test_images = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')))[:num_samples]
    
    if not test_images:
        print(f"Error: No images found in {test_images_dir}")
        return
    
    results = {
        'lighting': {},
        'occlusion': {},
        'scale': {}
    }
    
    # Test lighting variations
    print("\n[1/3] Testing lighting variations...")
    lighting_results = []
    
    for img_path in tqdm(test_images[:20], desc="Lighting"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        variations = apply_lighting_variations(image)
        
        for var_name, var_image in variations.items():
            try:
                pred = model.predict(
                    source=var_image,
                    imgsz=1280,
                    conf=0.25,
                    iou=0.45,
                    verbose=False
                )[0]
                
                num_detections = len(pred.boxes)
                avg_confidence = float(pred.boxes.conf.mean()) if len(pred.boxes) > 0 else 0.0
                
                lighting_results.append({
                    'condition': var_name,
                    'num_detections': num_detections,
                    'avg_confidence': avg_confidence
                })
            except Exception as e:
                print(f"Error processing {var_name}: {e}")
                continue
    
    # Aggregate lighting results
    lighting_df = pd.DataFrame(lighting_results) if lighting_results else pd.DataFrame()
    if len(lighting_df) > 0:
        lighting_summary = lighting_df.groupby('condition').agg({
            'num_detections': ['mean', 'std'],
            'avg_confidence': ['mean', 'std']
        }).round(4)
        # Convert MultiIndex columns to string keys for JSON serialization
        results['lighting'] = {str(k): v for k, v in lighting_summary.to_dict().items()}
    else:
        lighting_df = pd.DataFrame()
    
    # Test occlusion variations
    print("\n[2/3] Testing occlusion robustness...")
    occlusion_results = []
    
    for img_path in tqdm(test_images[:20], desc="Occlusion"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        variations = apply_occlusion(image, occlusion_level=0.3)
        
        for var_name, var_image in variations.items():
            try:
                pred = model.predict(
                    source=var_image,
                    imgsz=1280,
                    conf=0.25,
                    iou=0.45,
                    verbose=False
                )[0]
                
                num_detections = len(pred.boxes)
                avg_confidence = float(pred.boxes.conf.mean()) if len(pred.boxes) > 0 else 0.0
                
                occlusion_results.append({
                    'condition': var_name,
                    'num_detections': num_detections,
                    'avg_confidence': avg_confidence
                })
            except Exception as e:
                print(f"Error processing {var_name}: {e}")
                continue
    
    # Aggregate occlusion results
    occlusion_df = pd.DataFrame(occlusion_results) if occlusion_results else pd.DataFrame()
    if len(occlusion_df) > 0:
        occlusion_summary = occlusion_df.groupby('condition').agg({
            'num_detections': ['mean', 'std'],
            'avg_confidence': ['mean', 'std']
        }).round(4)
        # Convert MultiIndex columns to string keys for JSON serialization
        results['occlusion'] = {str(k): v for k, v in occlusion_summary.to_dict().items()}
    else:
        occlusion_df = pd.DataFrame()
    
    # Test scale variations
    print("\n[3/3] Testing scale robustness...")
    scale_results = []
    
    for img_path in tqdm(test_images[:20], desc="Scale"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        variations = apply_scale_variations(image)
        
        for var_name, var_image in variations.items():
            try:
                pred = model.predict(
                    source=var_image,
                    imgsz=1280,
                    conf=0.25,
                    iou=0.45,
                    verbose=False
                )[0]
                
                num_detections = len(pred.boxes)
                avg_confidence = float(pred.boxes.conf.mean()) if len(pred.boxes) > 0 else 0.0
                
                scale_results.append({
                    'condition': var_name,
                    'num_detections': num_detections,
                    'avg_confidence': avg_confidence
                })
            except Exception as e:
                print(f"Error processing {var_name}: {e}")
                continue
    
    # Aggregate scale results
    scale_df = pd.DataFrame(scale_results) if scale_results else pd.DataFrame()
    if len(scale_df) > 0:
        scale_summary = scale_df.groupby('condition').agg({
            'num_detections': ['mean', 'std'],
            'avg_confidence': ['mean', 'std']
        }).round(4)
        # Convert MultiIndex columns to string keys for JSON serialization
        results['scale'] = {str(k): v for k, v in scale_summary.to_dict().items()}
    else:
        scale_df = pd.DataFrame()
    
    # Save results
    results_file = output_dir / 'robustness_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("ROBUSTNESS TEST RESULTS SUMMARY")
    print("=" * 80)
    
    if lighting_df is not None and len(lighting_df) > 0:
        print("\nLighting Variations:")
        baseline = lighting_df[lighting_df['condition'] == 'original']
        for condition in lighting_df['condition'].unique():
            if condition == 'original':
                continue
            cond_data = lighting_df[lighting_df['condition'] == condition]
            det_change = (cond_data['num_detections'].mean() - baseline['num_detections'].mean()) / baseline['num_detections'].mean() * 100
            conf_change = (cond_data['avg_confidence'].mean() - baseline['avg_confidence'].mean()) / baseline['avg_confidence'].mean() * 100
            print(f"  {condition:20s}: Detections {det_change:+.1f}%, Confidence {conf_change:+.1f}%")
    
    if occlusion_df is not None and len(occlusion_df) > 0:
        print("\nOcclusion Variations:")
        baseline = occlusion_df[occlusion_df['condition'] == 'original']
        for condition in occlusion_df['condition'].unique():
            if condition == 'original':
                continue
            cond_data = occlusion_df[occlusion_df['condition'] == condition]
            det_change = (cond_data['num_detections'].mean() - baseline['num_detections'].mean()) / baseline['num_detections'].mean() * 100
            conf_change = (cond_data['avg_confidence'].mean() - baseline['avg_confidence'].mean()) / baseline['avg_confidence'].mean() * 100
            print(f"  {condition:20s}: Detections {det_change:+.1f}%, Confidence {conf_change:+.1f}%")
    
    if scale_df is not None and len(scale_df) > 0:
        print("\nScale Variations:")
        baseline = scale_df[scale_df['condition'] == 'original']
        for condition in scale_df['condition'].unique():
            if condition == 'original':
                continue
            cond_data = scale_df[scale_df['condition'] == condition]
            det_change = (cond_data['num_detections'].mean() - baseline['num_detections'].mean()) / baseline['num_detections'].mean() * 100
            conf_change = (cond_data['avg_confidence'].mean() - baseline['avg_confidence'].mean()) / baseline['avg_confidence'].mean() * 100
            print(f"  {condition:20s}: Detections {det_change:+.1f}%, Confidence {conf_change:+.1f}%")
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test model robustness under varied conditions')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Directory with test images')
    parser.add_argument('--output', type=str, default='robustness_analysis', help='Output directory')
    parser.add_argument('--samples', type=int, default=50, help='Number of samples to test')
    args = parser.parse_args()
    
    test_robustness(args.weights, args.source, args.output, args.samples)

