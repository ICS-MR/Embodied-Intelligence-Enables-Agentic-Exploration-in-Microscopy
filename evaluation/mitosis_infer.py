"""Run MMDetection inference on the local mitosis COCO test subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

try:
    from mmdet.apis import inference_detector, init_detector
except Exception:
    inference_detector = None
    init_detector = None

try:
    import torch
except Exception:
    torch = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mitosis model inference on a COCO-style test subset.")
    parser.add_argument("--config", default="configs/mitosis_rtmdet.py", help="Path to the MMDetection config file.")
    parser.add_argument("--checkpoint", default="weights/mitosis_best.pth", help="Path to the trained checkpoint.")
    parser.add_argument(
        "--images-dir",
        default="evaluation/mitosis_testset/images",
        help="Directory containing evaluation images.",
    )
    parser.add_argument(
        "--annotations",
        default="evaluation/mitosis_testset/annotations.json",
        help="Path to the COCO annotations file.",
    )
    parser.add_argument(
        "--output",
        default="evaluation/mitosis_predictions.json",
        help="Path to the JSON file that will store predictions.",
    )
    parser.add_argument(
        "--vis-dir",
        default="evaluation/mitosis_visualizations",
        help="Directory where red-box visualization images will be saved.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.5,
        help="Confidence threshold applied before exporting detections.",
    )
    parser.add_argument(
        "--target-class",
        default="mitosis",
        help="Class name to export from the detector outputs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device, for example cpu, cuda:0, or auto.",
    )
    return parser.parse_args()


def _to_numpy_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _resolve_model_classes(model: Any) -> list[str]:
    dataset_meta = getattr(model, "dataset_meta", None)
    if isinstance(dataset_meta, dict):
        classes = dataset_meta.get("classes")
        if classes:
            return list(classes)

    classes = getattr(model, "CLASSES", None)
    if classes:
        return list(classes)

    raise RuntimeError("MMDetection model does not expose class metadata")


def _resolve_target_class_index(model: Any, target_class: str) -> int:
    classes = _resolve_model_classes(model)
    lowered = target_class.lower()

    for index, class_name in enumerate(classes):
        if str(class_name).lower() == lowered:
            return index

    if len(classes) == 1:
        return 0

    raise RuntimeError(f"Target class '{target_class}' was not found in model classes: {classes}")


def _extract_class_detections(result: Any, class_idx: int) -> np.ndarray:
    if hasattr(result, "pred_instances"):
        pred_instances = result.pred_instances
        bboxes = _to_numpy_array(getattr(pred_instances, "bboxes", None))
        scores = _to_numpy_array(getattr(pred_instances, "scores", None))
        labels = _to_numpy_array(getattr(pred_instances, "labels", None)).astype(int, copy=False)

        if bboxes.size == 0 or scores.size == 0 or labels.size == 0:
            return np.empty((0, 5), dtype=np.float32)

        keep_mask = labels == class_idx
        if not np.any(keep_mask):
            return np.empty((0, 5), dtype=np.float32)

        filtered_boxes = np.asarray(bboxes[keep_mask], dtype=np.float32)
        filtered_scores = np.asarray(scores[keep_mask], dtype=np.float32).reshape(-1, 1)
        return np.concatenate([filtered_boxes, filtered_scores], axis=1)

    if isinstance(result, (list, tuple)):
        if class_idx >= len(result):
            return np.empty((0, 5), dtype=np.float32)
        class_dets = np.asarray(result[class_idx], dtype=np.float32)
        if class_dets.size == 0:
            return np.empty((0, 5), dtype=np.float32)
        return class_dets.reshape(-1, 5)

    raise RuntimeError(f"Unsupported MMDetection result type: {type(result).__name__}")


def _resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _resolve_category_id(annotation_data: dict[str, Any], target_class: str) -> int:
    categories = annotation_data.get("categories", [])
    lowered = target_class.lower()

    for category in categories:
        if str(category.get("name", "")).lower() == lowered:
            return int(category["id"])

    if len(categories) == 1:
        return int(categories[0]["id"])

    raise RuntimeError(f"Target class '{target_class}' was not found in COCO categories: {categories}")


def _draw_prediction_boxes(
    image_path: Path,
    detections: np.ndarray,
    output_path: Path,
) -> None:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to read image for visualization: {image_path}") from exc

    draw = ImageDraw.Draw(image)

    for x1, y1, x2, y2, score in detections:
        pt1 = (int(round(float(x1))), int(round(float(y1))))
        pt2 = (int(round(float(x2))), int(round(float(y2))))
        draw.rectangle([pt1, pt2], outline=(255, 0, 0), width=3)
        label = f"mitosis {float(score):.2f}"
        text_origin = (pt1[0], max(20, pt1[1] - 8))
        draw.text(text_origin, label, fill=(255, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> int:
    args = parse_args()

    if init_detector is None or inference_detector is None:
        raise RuntimeError("MMDetection is unavailable. Please install a compatible mmdet/mmcv/torch stack.")

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    images_dir = Path(args.images_dir)
    annotations_path = Path(args.annotations)
    output_path = Path(args.output)
    vis_dir = Path(args.vis_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    annotation_data = json.loads(annotations_path.read_text(encoding="utf-8"))
    image_entries = annotation_data.get("images", [])
    if not image_entries:
        raise RuntimeError(f"No images were found in annotations file: {annotations_path}")

    device = _resolve_device(args.device)
    model = init_detector(str(config_path), str(checkpoint_path), device=device)
    class_idx = _resolve_target_class_index(model, args.target_class)
    category_id = _resolve_category_id(annotation_data, args.target_class)

    predictions: list[dict[str, Any]] = []
    total_detections = 0
    missing_images: list[str] = []
    visualized_images = 0

    for image_info in image_entries:
        image_id = int(image_info["id"])
        file_name = str(image_info["file_name"])
        image_path = images_dir / file_name

        if not image_path.exists():
            missing_images.append(file_name)
            continue

        result = inference_detector(model, str(image_path))
        detections = _extract_class_detections(result, class_idx)
        if detections.size == 0:
            valid_detections = np.empty((0, 5), dtype=np.float32)
        else:
            valid_detections = detections[detections[:, 4] >= args.score_thr]

        vis_output_path = vis_dir / file_name
        _draw_prediction_boxes(image_path, valid_detections, vis_output_path)
        visualized_images += 1

        for x1, y1, x2, y2, score in valid_detections:
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [
                        round(float(x1), 4),
                        round(float(y1), 4),
                        round(float(x2 - x1), 4),
                        round(float(y2 - y1), 4),
                    ],
                    "score": round(float(score), 6),
                }
            )
            total_detections += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    print(f"device={device}")
    print(f"images_scanned={len(image_entries)}")
    print(f"predictions_written={total_detections}")
    print(f"output={output_path}")
    print(f"visualizations_saved={visualized_images}")
    print(f"vis_dir={vis_dir}")

    if missing_images:
        print(f"missing_images={missing_images}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
