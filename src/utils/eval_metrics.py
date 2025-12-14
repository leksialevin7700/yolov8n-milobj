import json
from pathlib import Path
from ultralytics import YOLO


def evaluate_model(model_path, data_yaml, save_json=True):
    model = YOLO(model_path)

    # Run validation
    results = model.val(data=data_yaml, save_json=save_json)

    metrics = {
        "precision": results.box.mp,
        "recall": results.box.mr,
        "mAP50": results.box.map50,
        "mAP50-95": results.box.map,
        "per_class_AP": results.box.maps.tolist()
    }

    # Save JSON
    if save_json:
        output_path = Path("predictions/json_results/metrics.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    model_path = "runs/detect/train/weights/best.pt"
    data_yaml = "data/military_object_dataset/military_dataset.yaml"

    m = evaluate_model(model_path, data_yaml)
    print("\nEvaluation Results:")
    print(json.dumps(m, indent=4))
