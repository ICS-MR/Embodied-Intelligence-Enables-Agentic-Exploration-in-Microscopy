"""Command line entry point for unified localization experiments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .pipeline import (
    LocalizationConfig,
    compare_localizations,
    run_model_localization,
    run_vlm_localization,
)


DEFAULT_MANIFEST_RELATIVE = Path("docs_public/detector_model_examples/demo_manifest.json")


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "bootstrap" / "config.py").is_file():
            return parent
    raise RuntimeError("Unable to locate repository root containing bootstrap/config.py")


PROJECT_ROOT = _find_project_root()


def _resolve_path(path_value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON file: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON file must contain an object: {path}")
    return payload


def _load_manifest(manifest_path: str | Path | None) -> dict[str, Any]:
    path = _resolve_path(manifest_path or DEFAULT_MANIFEST_RELATIVE)
    manifest = _read_json(path)
    targets = manifest.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise RuntimeError(f"Manifest does not define any targets: {path}")
    return manifest


def _load_detection_targets() -> dict[str, dict[str, Any]]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from bootstrap.config import DEFAULT_DETECTION_TARGETS

    return DEFAULT_DETECTION_TARGETS


def _list_preset_targets(manifest_path: str | Path | None = None) -> list[str]:
    return list(_load_manifest(manifest_path)["targets"].keys())


def _resolve_category(annotation_data: dict[str, Any], target_class: str) -> tuple[int, str]:
    categories = annotation_data.get("categories", [])
    if not isinstance(categories, list) or not categories:
        raise RuntimeError("COCO annotation file does not define categories")

    lowered = str(target_class).lower()
    for category in categories:
        if str(category.get("name", "")).lower() == lowered:
            return int(category["id"]), str(category["name"])

    if len(categories) == 1:
        category = categories[0]
        return int(category["id"]), str(category["name"])

    raise RuntimeError(f"Target class '{target_class}' was not found in COCO categories: {categories}")


def _resolve_preset(
    target: str,
    *,
    image_name: str | None,
    image_path: str | Path | None,
    manifest_path: str | Path | None,
    require_model_files: bool,
) -> dict[str, Any]:
    manifest_targets = _load_manifest(manifest_path)["targets"]
    if target not in manifest_targets:
        available = ", ".join(manifest_targets)
        raise KeyError(f"Target '{target}' is not in manifest. Available targets: {available}")

    detection_targets = _load_detection_targets()
    if target not in detection_targets:
        available = ", ".join(detection_targets)
        raise KeyError(f"Target '{target}' is not registered in DEFAULT_DETECTION_TARGETS. Available targets: {available}")

    manifest_entry = manifest_targets[target]
    target_entry = detection_targets[target]
    annotations_path = _resolve_path(manifest_entry["annotations"])
    images_dir = _resolve_path(manifest_entry["images_dir"])
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Preset images directory does not exist: {images_dir}")

    annotation_data = _read_json(annotations_path)
    category_id, category_name = _resolve_category(annotation_data, str(target_entry["target_class_name"]))
    images = annotation_data.get("images", [])
    if not isinstance(images, list) or not images:
        raise RuntimeError("COCO annotation file does not define images")
    if image_name and image_path and Path(image_path).name != image_name:
        raise ValueError(
            "image_path basename must match image_name when both are supplied: "
            f"{Path(image_path).name} != {image_name}"
        )

    requested_name = image_name or (Path(image_path).name if image_path else None)
    if requested_name:
        matching_images = [image for image in images if str(image.get("file_name")) == requested_name]
        if not matching_images:
            available = ", ".join(str(image.get("file_name")) for image in images)
            raise RuntimeError(f"Image '{requested_name}' was not found in annotations. Available images: {available}")
        image_info = matching_images[0]
    else:
        image_info = images[0]

    resolved_name = str(image_info["file_name"])
    resolved_image_path = _resolve_path(image_path) if image_path else images_dir / resolved_name
    if not resolved_image_path.is_file():
        raise FileNotFoundError(f"Preset image file does not exist: {resolved_image_path}")

    config_path = _resolve_path(target_entry["model_config"])
    checkpoint_path = _resolve_path(target_entry["model_checkpoint"])
    if require_model_files:
        for required_path, label in ((config_path, "model config"), (checkpoint_path, "model checkpoint")):
            if not required_path.is_file():
                raise FileNotFoundError(f"Missing {label} for target '{target}': {required_path}")

    return {
        "image_path": str(resolved_image_path),
        "gt_annotation_file": str(annotations_path),
        "image_id": int(image_info["id"]),
        "category_id": category_id,
        "query_texts": [category_name],
        "config_file": str(config_path),
        "checkpoint_file": str(checkpoint_path),
        "score_thr": float(target_entry["score_thr"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VLM localization, model localization, or COCO-style comparison."
    )
    parser.add_argument("--mode", choices=["vlm", "model", "compare"])
    parser.add_argument("--target", help="Preset detector target from docs_public/detector_model_examples.")
    parser.add_argument("--image-name", help="Image file name inside the preset target testset.")
    parser.add_argument("--manifest", help="Detector examples manifest path for preset resolution.")
    parser.add_argument("--list-targets", action="store_true", help="List preset targets and exit.")
    parser.add_argument("--image", default="", help="Input image path.")
    parser.add_argument("--output-dir", default="localization_output")
    parser.add_argument("--image-id", type=int)
    parser.add_argument("--category-id", type=int)

    parser.add_argument("--config", help="MMDetection config file.")
    parser.add_argument("--checkpoint", help="MMDetection checkpoint file.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-thr", type=float)
    parser.add_argument("--nms-thr", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)

    parser.add_argument("--queries", nargs="+", help="VLM target texts.")
    parser.add_argument("--vlm-thr", type=float, default=0.3)
    parser.add_argument(
        "--no-env-proxy",
        action="store_true",
        help="Bypass HTTP(S)_PROXY environment settings for VLM API requests.",
    )

    parser.add_argument("--gt", help="COCO ground-truth annotation file for comparison.")
    parser.add_argument("--model-pred", help="Model prediction JSON for compare mode.")
    parser.add_argument("--vlm-pred", help="VLM COCO prediction JSON for compare mode.")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--confidence-thr", type=float, default=0.3)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_targets:
        for target in _list_preset_targets(args.manifest):
            print(target)
        return

    if not args.mode:
        parser.error("--mode is required unless --list-targets is used")
    if args.image_name and not args.target:
        parser.error("--image-name requires --target")

    image_path = args.image
    gt_annotation_file = args.gt
    config_file = args.config
    checkpoint_file = args.checkpoint
    image_id = args.image_id if args.image_id is not None else 1
    category_id = args.category_id if args.category_id is not None else 0
    score_thr = args.score_thr if args.score_thr is not None else 0.5
    query_texts = args.queries if args.queries is not None else ["cell"]

    if args.target:
        try:
            preset = _resolve_preset(
                args.target,
                image_name=args.image_name,
                image_path=args.image or None,
                manifest_path=args.manifest,
                require_model_files=args.mode == "model",
            )
        except Exception as exc:
            parser.error(str(exc))

        image_path = args.image or preset["image_path"]
        gt_annotation_file = args.gt or preset["gt_annotation_file"]
        config_file = args.config or preset["config_file"]
        checkpoint_file = args.checkpoint or preset["checkpoint_file"]
        image_id = args.image_id if args.image_id is not None else preset["image_id"]
        category_id = args.category_id if args.category_id is not None else preset["category_id"]
        score_thr = args.score_thr if args.score_thr is not None else preset["score_thr"]
        query_texts = args.queries if args.queries is not None else preset["query_texts"]

    if args.mode in {"vlm", "model"} and not image_path:
        parser.error("--image is required for vlm and model modes")
    if args.mode == "model" and (not config_file or not checkpoint_file):
        parser.error("--config and --checkpoint are required for model mode")
    if args.mode == "compare" and not gt_annotation_file:
        parser.error("--gt is required for compare mode")
    if args.mode == "compare" and bool(args.model_pred) != bool(args.vlm_pred):
        parser.error("--model-pred and --vlm-pred must be supplied together")

    cfg = LocalizationConfig(
        image_path=image_path,
        output_dir=args.output_dir,
        image_id=image_id,
        category_id=category_id,
        config_file=config_file,
        checkpoint_file=checkpoint_file,
        device=args.device,
        score_thr=score_thr,
        nms_thr=args.nms_thr,
        tile_size=args.tile_size,
        overlap=args.overlap,
        detection_threshold=args.vlm_thr,
        query_texts=query_texts,
        use_env_proxy=not args.no_env_proxy,
        gt_annotation_file=gt_annotation_file,
        iou_threshold=args.iou_thr,
        confidence_threshold=args.confidence_thr,
    )

    if args.mode == "model":
        run_model_localization(cfg)
    elif args.mode == "vlm":
        run_vlm_localization(cfg)
    else:
        pred_files = None
        if args.model_pred and args.vlm_pred:
            pred_files = [args.model_pred, args.vlm_pred]
        compare_localizations(cfg, pred_files=pred_files)


if __name__ == "__main__":
    main()
