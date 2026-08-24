"""Run qualitative inference for the currently registered preset detector models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bootstrap.config import DEFAULT_DETECTION_TARGETS

try:
    from mmdet.apis import inference_detector, init_detector
except Exception:
    inference_detector = None
    init_detector = None

try:
    import torch
except Exception:
    torch = None


DEFAULT_MANIFEST = "docs_public/detector_model_examples/demo_manifest.json"
DEFAULT_OUTPUT_DIR = "docs_public/detector_model_examples/outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run reviewer-facing qualitative examples for currently registered detector presets. "
            "The manifest provides images only; model config/checkpoint paths are loaded from bootstrap.config."
        )
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Path to the qualitative example manifest.")
    parser.add_argument(
        "--target",
        action="append",
        help="Preset target to run. Repeat for multiple targets. Defaults to mitosis unless --all or --validate is used.",
    )
    parser.add_argument("--all", action="store_true", help="Run all targets listed in the manifest.")
    parser.add_argument("--list-targets", action="store_true", help="List available manifest targets and exit.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate manifest image paths and registered detector assets without running inference.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory where outputs will be written.")
    parser.add_argument("--score-thr", type=float, default=None, help="Override the registered detector score threshold.")
    parser.add_argument("--device", default="auto", help="Inference device, for example cpu, cuda:0, or auto.")
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON file: {path}") from exc


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")
    manifest = _read_json(path)
    targets = manifest.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise RuntimeError(f"Manifest does not define any targets: {path}")
    return manifest


def _select_targets(args: argparse.Namespace, manifest: dict[str, Any]) -> list[str]:
    manifest_targets = list(manifest["targets"].keys())
    if args.all:
        return manifest_targets
    if args.target:
        return args.target
    if args.validate:
        return manifest_targets
    return ["mitosis"]


def _resolve_category_id(annotation_data: dict[str, Any], target_class: str) -> int:
    categories = annotation_data.get("categories", [])
    lowered = target_class.lower()

    for category in categories:
        if str(category.get("name", "")).lower() == lowered:
            return int(category["id"])

    if len(categories) == 1:
        return int(categories[0]["id"])

    raise RuntimeError(f"Target class '{target_class}' was not found in COCO categories: {categories}")


def _draw_prediction_boxes(image_path: Path, detections: np.ndarray, output_path: Path, label_name: str) -> None:
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to read image for visualization: {image_path}") from exc

    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, score in detections:
        pt1 = (int(round(float(x1))), int(round(float(y1))))
        pt2 = (int(round(float(x2))), int(round(float(y2))))
        draw.rectangle([pt1, pt2], outline=(255, 0, 0), width=3)
        text_origin = (pt1[0], max(20, pt1[1] - 8))
        draw.text(text_origin, f"{label_name} {float(score):.2f}", fill=(255, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def _validate_target(target: str, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest_targets = manifest["targets"]
    if target not in manifest_targets:
        available = ", ".join(manifest_targets)
        raise KeyError(f"Target '{target}' is not in manifest. Available targets: {available}")
    if target not in DEFAULT_DETECTION_TARGETS:
        available = ", ".join(DEFAULT_DETECTION_TARGETS)
        raise KeyError(f"Target '{target}' is not registered in DEFAULT_DETECTION_TARGETS. Available targets: {available}")

    target_entry = DEFAULT_DETECTION_TARGETS[target]
    manifest_entry = manifest_targets[target]
    config_path = _resolve_path(target_entry["model_config"])
    checkpoint_path = _resolve_path(target_entry["model_checkpoint"])
    images_dir = _resolve_path(manifest_entry["images_dir"])
    annotations_path = _resolve_path(manifest_entry["annotations"])

    for required_path, label in [
        (config_path, "model config"),
        (checkpoint_path, "model checkpoint"),
        (images_dir, "images directory"),
        (annotations_path, "annotations file"),
    ]:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing {label} for target '{target}': {required_path}")

    annotation_data = _read_json(annotations_path)
    image_entries = annotation_data.get("images", [])
    if not image_entries:
        raise RuntimeError(f"No images were found in annotations file for target '{target}': {annotations_path}")

    missing_images = [str(image_info.get("file_name")) for image_info in image_entries if not (images_dir / str(image_info.get("file_name"))).exists()]
    if missing_images:
        raise FileNotFoundError(f"Missing image files for target '{target}' in {images_dir}: {missing_images}")

    return {
        "target": target,
        "display_name": manifest_entry.get("display_name", target),
        "target_class_name": target_entry["target_class_name"],
        "score_thr": float(target_entry["score_thr"]),
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "images_dir": images_dir,
        "annotations_path": annotations_path,
        "annotation_data": annotation_data,
        "image_count": len(image_entries),
    }


def _run_target(target_info: dict[str, Any], output_root: Path, device: str, score_thr_override: float | None) -> dict[str, Any]:
    if init_detector is None or inference_detector is None:
        raise RuntimeError("MMDetection is unavailable. Please install a compatible mmdet/mmcv/torch stack.")

    target = target_info["target"]
    target_class = target_info["target_class_name"]
    score_thr = float(score_thr_override if score_thr_override is not None else target_info["score_thr"])
    annotation_data = target_info["annotation_data"]
    image_entries = annotation_data["images"]

    model = init_detector(str(target_info["config_path"]), str(target_info["checkpoint_path"]), device=device)
    class_idx = _resolve_target_class_index(model, target_class)
    category_id = _resolve_category_id(annotation_data, target_class)

    target_output_dir = output_root / target
    predictions_path = target_output_dir / "predictions.json"
    summary_path = target_output_dir / "summary.json"
    vis_dir = target_output_dir / "visualizations"

    predictions: list[dict[str, Any]] = []
    total_detections = 0

    for image_info in image_entries:
        image_id = int(image_info["id"])
        file_name = str(image_info["file_name"])
        image_path = target_info["images_dir"] / file_name

        result = inference_detector(model, str(image_path))
        detections = _extract_class_detections(result, class_idx)
        if detections.size == 0:
            valid_detections = np.empty((0, 5), dtype=np.float32)
        else:
            valid_detections = detections[detections[:, 4] >= score_thr]

        _draw_prediction_boxes(image_path, valid_detections, vis_dir / file_name, target_class)

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

    target_output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")

    summary = {
        "target": target,
        "display_name": target_info["display_name"],
        "target_class_name": target_class,
        "score_thr": score_thr,
        "device": device,
        "images_scanned": len(image_entries),
        "predictions_written": total_detections,
        "model_config": str(target_info["config_path"].relative_to(PROJECT_ROOT)),
        "model_checkpoint": str(target_info["checkpoint_path"].relative_to(PROJECT_ROOT)),
        "predictions": str(predictions_path.relative_to(PROJECT_ROOT)),
        "visualizations": str(vis_dir.relative_to(PROJECT_ROOT)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    manifest_path = _resolve_path(args.manifest)
    manifest = _load_manifest(manifest_path)
    selected_targets = _select_targets(args, manifest)

    if args.list_targets:
        for target in manifest["targets"]:
            status = "configured" if target in DEFAULT_DETECTION_TARGETS else "missing_config"
            target_class = DEFAULT_DETECTION_TARGETS.get(target, {}).get("target_class_name", "")
            config_path = DEFAULT_DETECTION_TARGETS.get(target, {}).get("model_config", "")
            print(f"{target}\t{status}\tclass={target_class}\tconfig={config_path}")
        return 0

    target_infos = [_validate_target(target, manifest) for target in selected_targets]

    if args.validate:
        for target_info in target_infos:
            print(
                f"validated={target_info['target']} "
                f"images={target_info['image_count']} "
                f"class={target_info['target_class_name']} "
                f"score_thr={target_info['score_thr']}"
            )
        return 0

    device = _resolve_device(args.device)
    output_root = _resolve_path(args.output_dir)
    summaries = [_run_target(target_info, output_root, device, args.score_thr) for target_info in target_infos]

    run_summary = {
        "purpose": manifest.get("purpose", ""),
        "targets": summaries,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")

    print(f"device={device}")
    print(f"targets_run={','.join(summary['target'] for summary in summaries)}")
    print(f"summary={summary_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
