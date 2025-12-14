
"""
Train YOLOv8 on the military dataset with robust defaults and safer device / yaml handling.
"""
import argparse
import os
from ultralytics import YOLO
from pathlib import Path
import yaml
import random
import numpy as np
import torch



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, required=True, help='path to military_dataset.yaml')
    # default to repo-root yolov8s.pt if present, otherwise use model name
    default_w = str(Path(__file__).resolve().parents[1] / 'yolov8s.pt')
    if not Path(default_w).exists():
        default_w = 'yolov8s.pt'  # fallback to model name if file doesn't exist
    p.add_argument('--weights', type=str, default=default_w, help='path to pretrained weights or model name')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--imgsz', type=int, default=416)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--project', type=str, default='runs/detect')
    p.add_argument('--name', type=str, default='milobj_run')
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--device', type=str, default='auto', help="auto | cpu | cuda:0 | 0 (legacy)")
    p.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
    return p.parse_args()


# ...existing code...
def check_yaml(data_path):
    """
    Load dataset YAML. If provided path doesn't exist, attempt to discover a YAML in repo/data or repo.
    """
    p = Path(data_path)
    repo_root = Path(__file__).resolve().parents[1]
    if p.exists():
        with p.open() as f:
            data = yaml.safe_load(f)
            source = p
    else:
        # try repo/data first, then any yaml under repo
        data_dir = repo_root / 'data'
        candidates = []
        if data_dir.exists():
            candidates = sorted(data_dir.glob('*.y*ml'))
        if not candidates:
            candidates = sorted(repo_root.rglob('*.y*ml'))
        if not candidates:
            raise FileNotFoundError(f"Data YAML not found at '{data_path}' and none discovered under {repo_root}. "
                                    "Pass --data with the full path to your dataset YAML.")
        source = candidates[0]
        print(f"Info: using discovered dataset YAML: {source}")
        with source.open() as f:
            data = yaml.safe_load(f)

    assert isinstance(data, dict), f"Loaded YAML {source} did not parse as dict"
    assert 'train' in data and 'val' in data and 'names' in data and 'nc' in data, \
        f"data yaml missing required keys ('train','val','names','nc') in {source}"
    print(f"Data YAML OK: train={data['train']}, val={data['val']}, nc={data.get('nc')}, classes={len(data.get('names',[]))}")
    return data


def main():
    args = parse_args()
    cfg = check_yaml(args.data)

    # seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # cuDNN settings: benchmark for speed on fixed-size imgs
    torch.backends.cudnn.benchmark = True

    # device normalization
    if args.device == 'auto':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # allow passing numeric GPU id like '0' or '1'
    if isinstance(args.device, str) and args.device.isdigit():
        args.device = f'cuda:{args.device}'
    if args.device == '0':  # legacy default handling
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # ensure project folder exists
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # warn if specified weights missing
    if args.weights and not Path(args.weights).exists():
        print(f"Warning: weights file '{args.weights}' not found. Ultralytics may download a model if given a model name.")

    model = YOLO(args.weights)  # load model (pretrained weight or ultralytics model name)

    print(f"Starting training: epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, device={args.device}")
    try:
        model.train(
            data=str(Path(args.data).resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=args.project,
            name=args.name,
            patience=args.patience,
            resume=args.resume,
            device=args.device,
            augment=True
        )
    except Exception as e:
        print(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
