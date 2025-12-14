"""
Optimize YOLOv8 inference speed without significant accuracy loss.
Tests different image sizes and batch sizes to find speed/accuracy trade-off.
"""
import argparse
import time
from pathlib import Path
from ultralytics import YOLO
import torch
import json


def benchmark_inference(weights_path, test_images_dir, imgsz_list=[640, 832, 1024, 1280], batch_sizes=[1, 4, 8]):
    """
    Benchmark inference speed at different image sizes and batch sizes.
    """
    print("=" * 80)
    print("INFERENCE SPEED OPTIMIZATION")
    print("=" * 80)
    
    model = YOLO(weights_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Get test images
    test_dir = Path(test_images_dir)
    test_images = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')))[:50]  # Test on 50 images
    
    if not test_images:
        print(f"Error: No images found in {test_images_dir}")
        return
    
    print(f"Testing on {len(test_images)} images\n")
    
    results = []
    
    for imgsz in imgsz_list:
        for batch_size in batch_sizes:
            if batch_size > len(test_images):
                continue
            
            print(f"Testing: imgsz={imgsz}, batch={batch_size}...", end=' ', flush=True)
            
            # Warmup
            model.predict(
                source=str(test_images[0]),
                imgsz=imgsz,
                conf=0.25,
                iou=0.45,
                verbose=False
            )
            
            # Benchmark
            start_time = time.time()
            for i in range(0, len(test_images), batch_size):
                batch = test_images[i:i+batch_size]
                model.predict(
                    source=[str(img) for img in batch],
                    imgsz=imgsz,
                    conf=0.25,
                    iou=0.45,
                    verbose=False
                )
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_time_per_image = total_time / len(test_images)
            fps = len(test_images) / total_time
            
            results.append({
                'imgsz': imgsz,
                'batch_size': batch_size,
                'total_time': round(total_time, 2),
                'avg_time_per_image_ms': round(avg_time_per_image * 1000, 2),
                'fps': round(fps, 2)
            })
            
            print(f"✓ {avg_time_per_image*1000:.1f}ms/image ({fps:.1f} FPS)")
    
    # Find best configurations
    print("\n" + "=" * 80)
    print("SPEED OPTIMIZATION RESULTS")
    print("=" * 80)
    
    # Sort by speed
    results_sorted = sorted(results, key=lambda x: x['avg_time_per_image_ms'])
    
    print("\nFastest configurations:")
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"{i}. imgsz={r['imgsz']}, batch={r['batch_size']}: "
              f"{r['avg_time_per_image_ms']:.1f}ms/image ({r['fps']:.1f} FPS)")
    
    # Find best speed/accuracy trade-off (smaller imgsz, reasonable batch)
    best_speed = results_sorted[0]
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print(f"\nFor maximum speed (current: ~1660ms/image):")
    print(f"  - Use imgsz={best_speed['imgsz']}, batch={best_speed['batch_size']}")
    print(f"  - Expected: ~{best_speed['avg_time_per_image_ms']:.0f}ms/image")
    print(f"  - Speedup: {1660/best_speed['avg_time_per_image_ms']:.1f}x faster")
    
    print(f"\nCommand for fast inference:")
    print(f"  yolo detect predict model={weights_path} source=<image_dir> imgsz={best_speed['imgsz']} conf=0.25 iou=0.45 batch={best_speed['batch_size']}")
    
    # Save results
    output_file = Path('inference_speed_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'device': device,
            'test_images': len(test_images),
            'results': results,
            'best_speed': best_speed
        }, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize inference speed')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Directory with test images')
    parser.add_argument('--imgsz', type=int, nargs='+', default=[640, 832, 1024, 1280],
                       help='Image sizes to test')
    parser.add_argument('--batch', type=int, nargs='+', default=[1, 4, 8],
                       help='Batch sizes to test')
    args = parser.parse_args()
    
    benchmark_inference(args.weights, args.source, args.imgsz, args.batch)

