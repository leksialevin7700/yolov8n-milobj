"""
Visualization helpers: draw boxes on images and save.
"""
import cv2
from pathlib import Path




def draw_boxes(image, boxes, labels, scores=None, save_path=None, class_names=None):
    # boxes: list of [x1,y1,x2,y2]
    im = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cls = labels[i]
        score = scores[i] if scores is not None else None
        label = f"{class_names[cls] if class_names else cls} {score:.2f}" if score is not None else str(cls)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(im, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_path, im)
    return im