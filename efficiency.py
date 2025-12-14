"""
Comprehensive efficiency analysis: model size, inference speed, CPU feasibility.
Addresses 15% of judging criteria.
"""
import argparse
import json
import time
import torch
from pathlib import Path
from ultralytics import YOLO
import os
import psutil
import platform


def get_model_size(weights_path):
    """Get model size information."""
    file_size_mb = os.path.getsize(weights_path) / (1024 * 1024)
    
    # Load model to get parameter count
    model = YOLO(weights_path)
    
    # Get model info
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    # Estimate model size in memory (FP32)
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    return {
        'file_size_mb': round(file_size_mb, 2),
        'model_size_mb': round(model_size_mb, 2),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_name': Path(weights_path).stem
    }


def benchmark_cpu_inference(weights_path, test_images_dir, imgsz_list=[640, 832, 1024, 1280], num_samples=20):
    """Benchmark inference speed on CPU."""
    print("=" * 80)
    print("CPU INFERENCE BENCHMARKING")
    print("=" * 80)
    
    model = YOLO(weights_path)
    
    # Force CPU
    device = 'cpu'
    print(f"Device: {device}")
    print(f"CPU: {platform.processor()}")
    print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count(logical=True)} logical")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB total")
    print("=" * 80)
    
    # Get test images
    test_dir = Path(test_images_dir)
    test_images = sorted(list(test_dir.glob('*.jpg')) + list(test_dir.glob('*.png')))[:num_samples]
    
    if not test_images:
        print(f"Error: No images found in {test_images_dir}")
        return {}
    
    results = []
    
    for imgsz in imgsz_list:
        print(f"\nTesting imgsz={imgsz}...")
        
        # Warmup
        model.predict(
            source=str(test_images[0]),
            imgsz=imgsz,
            conf=0.25,
            iou=0.45,
            device=device,
            verbose=False
        )
        
        # Benchmark
        times = []
        memory_usage = []
        
        for img_path in test_images:
            # Measure memory before
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 * 1024)  # MB
            
            # Time inference
            start = time.time()
            pred = model.predict(
                source=str(img_path),
                imgsz=imgsz,
                conf=0.25,
                iou=0.45,
                device=device,
                verbose=False
            )
            end = time.time()
            
            # Measure memory after
            mem_after = process.memory_info().rss / (1024 * 1024)  # MB
            
            times.append(end - start)
            memory_usage.append(mem_after - mem_before)
        
        avg_time = sum(times) / len(times)
        avg_time_ms = avg_time * 1000
        fps = 1.0 / avg_time
        std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
        max_memory = max(memory_usage)
        
        results.append({
            'imgsz': imgsz,
            'avg_time_ms': round(avg_time_ms, 2),
            'std_time_ms': round(std_time * 1000, 2),
            'fps': round(fps, 2),
            'min_time_ms': round(min(times) * 1000, 2),
            'max_time_ms': round(max(times) * 1000, 2),
            'peak_memory_mb': round(max_memory, 2)
        })
        
        print(f"  Average: {avg_time_ms:.1f}ms/image ({fps:.2f} FPS)")
        print(f"  Range: {min(times)*1000:.1f} - {max(times)*1000:.1f}ms")
        print(f"  Peak Memory: {max_memory:.1f} MB")
    
    return results


def analyze_efficiency(weights_path, test_images_dir, output_dir='efficiency_analysis'):
    """
    Comprehensive efficiency analysis.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("MODEL EFFICIENCY ANALYSIS")
    print("=" * 80)
    print(f"Model: {weights_path}")
    print(f"Test images: {test_images_dir}")
    print("=" * 80)
    
    # 1. Model size analysis
    print("\n[1/3] Analyzing model size...")
    size_info = get_model_size(weights_path)
    
    print(f"\nModel Size Information:")
    print(f"  File size: {size_info['file_size_mb']} MB")
    print(f"  Model size (FP32): {size_info['model_size_mb']} MB")
    print(f"  Total parameters: {size_info['total_parameters']:,}")
    print(f"  Trainable parameters: {size_info['trainable_parameters']:,}")
    
    # 2. CPU inference benchmarking
    print("\n[2/3] Benchmarking CPU inference...")
    cpu_results = benchmark_cpu_inference(weights_path, test_images_dir)
    
    # 3. Feasibility assessment
    print("\n[3/3] Assessing CPU-class device feasibility...")
    
    # CPU-class device criteria (typical edge devices)
    cpu_class_criteria = {
        'max_model_size_mb': 100,  # Reasonable for edge devices
        'target_fps': 1.0,  # At least 1 FPS for real-time
        'max_memory_mb': 500,  # Reasonable memory footprint
        'target_latency_ms': 1000  # <1 second per image
    }
    
    feasibility = {}
    
    # Check model size
    feasibility['model_size_ok'] = size_info['file_size_mb'] <= cpu_class_criteria['max_model_size_mb']
    
    # Check inference speed
    if cpu_results:
        best_result = min(cpu_results, key=lambda x: x['avg_time_ms'])
        feasibility['speed_ok'] = best_result['fps'] >= cpu_class_criteria['target_fps']
        feasibility['latency_ok'] = best_result['avg_time_ms'] <= cpu_class_criteria['target_latency_ms']
        feasibility['memory_ok'] = best_result['peak_memory_mb'] <= cpu_class_criteria['max_memory_mb']
        feasibility['recommended_imgsz'] = best_result['imgsz']
        feasibility['recommended_fps'] = best_result['fps']
        feasibility['recommended_latency_ms'] = best_result['avg_time_ms']
    else:
        feasibility['speed_ok'] = False
        feasibility['latency_ok'] = False
        feasibility['memory_ok'] = False
    
    # Overall feasibility
    feasibility['cpu_class_feasible'] = all([
        feasibility.get('model_size_ok', False),
        feasibility.get('speed_ok', False),
        feasibility.get('latency_ok', False),
        feasibility.get('memory_ok', False)
    ])
    
    # Compile results
    results = {
        'model_size': size_info,
        'cpu_benchmarks': cpu_results,
        'feasibility': feasibility,
        'cpu_class_criteria': cpu_class_criteria,
        'system_info': {
            'cpu': platform.processor(),
            'cpu_cores_physical': psutil.cpu_count(logical=False),
            'cpu_cores_logical': psutil.cpu_count(logical=True),
            'ram_gb': round(psutil.virtual_memory().total / (1024**3), 1),
            'platform': platform.platform()
        }
    }
    
    # Save results
    results_file = output_dir / 'efficiency_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EFFICIENCY ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"\nModel Size:")
    print(f"  File: {size_info['file_size_mb']} MB")
    print(f"  Parameters: {size_info['total_parameters']:,}")
    print(f"  CPU-class feasible: {'✓ Yes' if feasibility['model_size_ok'] else '✗ No'}")
    
    if cpu_results:
        print(f"\nCPU Inference Performance:")
        best = min(cpu_results, key=lambda x: x['avg_time_ms'])
        print(f"  Best configuration: imgsz={best['imgsz']}")
        print(f"  Speed: {best['fps']:.2f} FPS ({best['avg_time_ms']:.1f}ms/image)")
        print(f"  Peak Memory: {best['peak_memory_mb']:.1f} MB")
        print(f"  CPU-class feasible: {'✓ Yes' if feasibility['cpu_class_feasible'] else '✗ No'}")
    
    print(f"\nOverall CPU-Class Device Feasibility:")
    if feasibility['cpu_class_feasible']:
        print("  ✓ Model is feasible for CPU-class devices")
        print(f"  Recommended: imgsz={feasibility['recommended_imgsz']}, "
              f"{feasibility['recommended_fps']:.2f} FPS")
    else:
        print("  ✗ Model may not be optimal for CPU-class devices")
        print("  Recommendations:")
        if not feasibility['model_size_ok']:
            print("    - Consider model quantization (INT8)")
        if not feasibility.get('speed_ok', False):
            print("    - Use smaller image size (imgsz=640)")
        if not feasibility.get('memory_ok', False):
            print("    - Reduce batch size or use model pruning")
    
    print(f"\n✓ Results saved to {results_file}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze model efficiency')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--source', type=str, required=True, help='Directory with test images')
    parser.add_argument('--output', type=str, default='efficiency_analysis', help='Output directory')
    args = parser.parse_args()
    
    analyze_efficiency(args.weights, args.source, args.output)

