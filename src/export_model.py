"""
Export a trained YOLOv8 model to desired formats (onnx, torchscript, openvino, tflite etc.)
"""
import argparse
from ultralytics import YOLO




def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, required=True)
    p.add_argument('--formats', type=str, nargs='+', default=['onnx', 'torchscript'])
    p.add_argument('--imgsz', type=int, default=1280)
    return p.parse_args()




def main():
    args = parse_args()
    model = YOLO(args.weights)
    for fmt in args.formats:
        print(f"Exporting to {fmt}...")
        model.export(format=fmt, imgsz=args.imgsz)
    print('Export complete')




if __name__ == '__main__':
    main()