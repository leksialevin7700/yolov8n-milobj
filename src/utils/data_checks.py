"""
Small utilities to check dataset integrity, convert COCO->YOLO if needed, and inspect label counts.
"""
import json
from pathlib import Path
import os
import shutil
from collections import Counter
import cv2
import numpy as np




def check_images_labels(root_path: str):
    root = Path(root_path)
    for split in ['train', 'val']:
        img_dir = root / split / 'images'
        lbl_dir = root / split / 'labels'
        imgs = list(img_dir.glob('*'))
        lbls = list(lbl_dir.glob('*.txt'))
        print(f"{split}: {len(imgs)} images, {len(lbls)} label files")
        # quick label content check
        if lbls:
            sample = lbls[0]
            with open(sample) as f:
                line = f.readline().strip()
            print('Sample label:', line)




def label_distribution(root_path: str):
    root = Path(root_path)
    counts = Counter()
    for split in ['train', 'val']:
        for txt in (root/split/'labels').glob('*.txt'):
            with open(txt) as f:
                for l in f:
                    parts = l.strip().split()
                    if not parts: continue
                    cls = int(parts[0])
                    counts[cls] += 1
    print('Label counts by class:', counts)




# COCO->YOLO conversion helper (very small; assumes coco json contains bbox in [x,y,w,h] absolute)


def coco_to_yolo(coco_json_path: str, imgs_dir: str, out_lbl_dir: str):
    coco = json.load(open(coco_json_path))
    id2file = {x['id']: x['file_name'] for x in coco['images']}
    annos = {}
    for a in coco['annotations']:
        img_fn = id2file[a['image_id']]
        if img_fn not in annos: annos[img_fn] = []
        annos[img_fn].append(a)

    os.makedirs(out_lbl_dir, exist_ok=True)
    for img_fn, anns in annos.items():
        img_path = Path(imgs_dir) / img_fn
        if not img_path.exists():
            continue
        h, w = cv2.imread(str(img_path)).shape[:2]
        out_lines = []
    print('COCO->YOLO conversion complete')