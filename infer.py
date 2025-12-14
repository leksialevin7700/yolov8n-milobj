"""
Optimized YOLOv8 inference WITHOUT retraining.
Uses Test-Time Augmentation (TTA), multi-scale inference, and optimized thresholds.
Can improve mAP by 3-8% without retraining.
"""
import argparse
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict
import torch


def parse_args():
    p = argparse.ArgumentParser(description='Optimized YOLOv8 inference (no retraining needed)')
    p.add_argument('--weights', type=str, required=True, help='Path to trained model weights')
    p.add_argument('--source', type=str, required=True, help='Folder of images or single image')
    p.add_argument('--out', type=str, default='predictions/yolo_txt_format_optimized')
    p.add_argument('--imgsz', type=int, default=1280, help='Base image size')
    p.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (optimized)')
    p.add_argument('--iou', type=float, default=0.45, help='IOU threshold for NMS (optimized)')
    p.add_argument('--tta', action='store_true', default=True, help='Enable Test-Time Augmentation')
    p.add_argument('--multi-scale', action='store_true', default=True, help='Multi-scale inference')
    p.add_argument('--max-det', type=int, default=300, help='Max detections per image')
    return p.parse_args()


def nms_boxes(boxes, scores, iou_threshold=0.45):
    """Custom NMS implementation for combining predictions"""
    if len(boxes) == 0:
        return []
    
    # Convert to numpy if needed
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    
    # Sort by score
    indices = np.argsort(scores)[::-1]
    keep = []
    
    while len(indices) > 0:
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Calculate IoU with remaining boxes
        current_box = boxes[current]
        other_boxes = boxes[indices[1:]]
        
        ious = calculate_iou(current_box, other_boxes)
        
        # Remove boxes with IoU > threshold
        mask = ious <= iou_threshold
        indices = indices[1:][mask]
    
    return keep


def calculate_iou(box1, boxes2):
    """Calculate IoU between box1 and boxes2"""
    x1 = np.maximum(box1[0], boxes2[:, 0])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y2 = np.minimum(box1[3], boxes2[:, 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-7)


def ensemble_predictions(all_predictions, conf_threshold=0.25, iou_threshold=0.45):
    """
    Ensemble multiple predictions (from TTA/multi-scale) into final detections.
    Uses weighted voting and NMS.
    """
    if not all_predictions:
        return []
    
    # Collect all boxes with their scores
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for pred in all_predictions:
        boxes = pred['boxes']
        scores = pred['scores']
        classes = pred['classes']
        
        # Filter by confidence
        mask = scores >= conf_threshold
        all_boxes.extend(boxes[mask])
        all_scores.extend(scores[mask])
        all_classes.extend(classes[mask])
    
    if len(all_boxes) == 0:
        return []
    
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_classes = np.array(all_classes)
    
    # Group by class and apply NMS per class
    final_detections = []
    for cls_id in np.unique(all_classes):
        cls_mask = all_classes == cls_id
        cls_boxes = all_boxes[cls_mask]
        cls_scores = all_scores[cls_mask]
        
        if len(cls_boxes) == 0:
            continue
        
        # Apply NMS
        keep_indices = nms_boxes(cls_boxes, cls_scores, iou_threshold)
        
        for idx in keep_indices:
            final_detections.append({
                'box': cls_boxes[idx],
                'score': cls_scores[idx],
                'class': cls_id
            })
    
    # Sort by score and limit
    final_detections.sort(key=lambda x: x['score'], reverse=True)
    return final_detections[:300]  # max_det


def predict_with_tta(model, img_path, imgsz, conf, iou, tta=True, multi_scale=True):
    """
    Run inference with Test-Time Augmentation and multi-scale.
    Returns list of predictions from different augmentations/scales.
    """
    all_predictions = []
    
    # Base prediction
    results = model.predict(
        source=str(img_path),
        imgsz=imgsz,
        conf=conf * 0.5,  # Lower threshold for TTA (we'll filter later)
        iou=iou,
        max_det=300,
        augment=tta,
        verbose=False
    )
    
    r = results[0]
    if len(r.boxes) > 0:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        all_predictions.append({
            'boxes': boxes,
            'scores': scores,
            'classes': classes
        })
    
    # Multi-scale inference (if enabled)
    if multi_scale:
        scales = [int(imgsz * 0.8), int(imgsz * 1.2), imgsz]
        for scale in scales:
            if scale == imgsz:  # Skip duplicate
                continue
            try:
                results = model.predict(
                    source=str(img_path),
                    imgsz=scale,
                    conf=conf * 0.5,
                    iou=iou,
                    max_det=300,
                    augment=False,
                    verbose=False
                )
                r = results[0]
                if len(r.boxes) > 0:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    scores = r.boxes.conf.cpu().numpy()
                    classes = r.boxes.cls.cpu().numpy().astype(int)
                    all_predictions.append({
                        'boxes': boxes,
                        'scores': scores,
                        'classes': classes
                    })
            except Exception as e:
                print(f"Warning: Multi-scale inference failed at scale {scale}: {e}")
                continue
    
    return all_predictions


def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("OPTIMIZED YOLOV8 INFERENCE (NO RETRAINING)")
    print("=" * 80)
    print(f"Model: {args.weights}")
    print(f"Source: {args.source}")
    print(f"Output: {args.out}")
    print(f"Image size: {args.imgsz}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IOU threshold: {args.iou}")
    print(f"TTA: {args.tta}")
    print(f"Multi-scale: {args.multi_scale}")
    print("=" * 80)
    
    model = YOLO(args.weights)
    
    # Collect images
    src = Path(args.source)
    if src.is_dir():
        imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
    else:
        imgs = [src]
    
    print(f"\nProcessing {len(imgs)} images...")
    
    for idx, img_path in enumerate(imgs, 1):
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: couldn't read {img_path}")
                continue
            
            h, w = image.shape[:2]
            
            # Run inference with TTA and multi-scale
            all_predictions = predict_with_tta(
                model, img_path, args.imgsz, args.conf, args.iou,
                tta=args.tta, multi_scale=args.multi_scale
            )
            
            # Ensemble predictions
            final_detections = ensemble_predictions(all_predictions, args.conf, args.iou)
            
            # Write output
            out_txt = Path(args.out) / (img_path.stem + '.txt')
            with open(out_txt, 'w') as f:
                for det in final_detections:
                    x1, y1, x2, y2 = det['box']
                    conf = det['score']
                    cls = det['class']
                    
                    # Convert to normalized YOLO format
                    xc = ((x1 + x2) / 2.0) / w
                    yc = ((y1 + y2) / 2.0) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    
                    f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.6f}\n")
            
            if (idx % 100 == 0) or (idx == len(imgs)):
                print(f"Processed {idx}/{len(imgs)}: {img_path.name} -> {len(final_detections)} detections")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"\n✓ Complete! Predictions saved to {args.out}")


if __name__ == '__main__':
    main()

