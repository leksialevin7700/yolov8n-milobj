"""
Run inference on a folder of images and write YOLO-format .txt prediction files (normalized coords + confidence).
"""
import argparse
from ultralytics import YOLO
from pathlib import Path
import cv2
import os




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--source', type=str, required=True, help='folder of images or single image')
    p.add_argument('--out', type=str, default='predictions/yolo_txt_format')
    p.add_argument('--imgsz', type=int, default=1280)
    p.add_argument('--conf', type=float, default=0.001)
    p.add_argument('--iou', type=float, default=0.7)
    return p.parse_args()




def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    # collect images
    src = Path(args.source)
    if src.is_dir():
        imgs = sorted([p for p in src.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
    else:
        imgs = [src]

    for img_path in imgs:
        results = model.predict(source=str(img_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou, max_det=300)
        # results is a list, single image -> results[0]
        r = results[0]
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Warning: couldn't read {img_path}")
            continue
        h, w = image.shape[:2]

        out_txt = Path(args.out) / (img_path.stem + '.txt')
        with open(out_txt, 'w') as f:
            for box in r.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy
                xc = ((x1 + x2) / 2.0) / w
                yc = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f} {conf:.6f}\n")
        print(f"Wrote {out_txt} with {len(r.boxes)} detections")




if __name__ == '__main__':
    main()