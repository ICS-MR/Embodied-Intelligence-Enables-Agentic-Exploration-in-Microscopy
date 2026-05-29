"""Public pipeline for VLM localization, model localization, and comparison."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence


@dataclass
class LocalizationConfig:
    """Configuration shared by the model and VLM localization pipeline."""

    image_path: str
    output_dir: str = "localization_output"
    image_id: int = 1
    category_id: int = 0

    # Local MMDetection model settings.
    config_file: str | None = None
    checkpoint_file: str | None = None
    device: str = "cuda:0"
    score_thr: float = 0.5
    nms_thr: float = 0.5
    tile_size: int = 1024
    overlap: int = 128
    pad_to_tile_size: bool = True

    # VLM settings.
    detection_threshold: float = 0.3
    query_texts: Sequence[str] = field(default_factory=lambda: ("cell",))

    # Optional evaluation settings.
    gt_annotation_file: str | None = None
    iou_threshold: float = 0.5
    confidence_threshold: float = 0.3

    def ensure_output_dir(self) -> Path:
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


def _vlm_json_to_coco(
    input_file: str,
    output_file: str,
    image_id: int,
    category_id: int,
) -> int:
    """Convert raw VLM xyxy JSON into COCO result xywh JSON."""
    with open(input_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    coco_results = []
    for item in raw_data:
        bbox = item.get("bbox", {})
        x_min = float(bbox["x_min"])
        y_min = float(bbox["y_min"])
        x_max = float(bbox["x_max"])
        y_max = float(bbox["y_max"])
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0:
            continue

        coco_results.append({
            "image_id": int(image_id),
            "category_id": int(category_id),
            "bbox": [
                round(x_min, 2),
                round(y_min, 2),
                round(width, 2),
                round(height, 2),
            ],
            "score": round(float(item.get("confidence", 1.0)), 4),
        })

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(coco_results, f, indent=2, ensure_ascii=False)

    return len(coco_results)


def run_model_localization(cfg: LocalizationConfig):
    """Run tiled MMDetection localization and save COCO-style predictions."""
    from .model_inference import detect_and_save_tiled_mmdet

    if not cfg.config_file:
        raise ValueError("config_file is required for model localization")
    if not cfg.checkpoint_file:
        raise ValueError("checkpoint_file is required for model localization")

    output_dir = cfg.ensure_output_dir()
    return detect_and_save_tiled_mmdet(
        config_file=cfg.config_file,
        checkpoint_file=cfg.checkpoint_file,
        img_path=cfg.image_path,
        output_json=str(output_dir / "model_detection_result.json"),
        output_img=str(output_dir / "model_result.jpg"),
        device=cfg.device,
        score_thr=cfg.score_thr,
        nms_thr=cfg.nms_thr,
        tile_size=cfg.tile_size,
        overlap=cfg.overlap,
        image_id=cfg.image_id,
        pad_to_tile_size=cfg.pad_to_tile_size,
    )


def run_vlm_localization(cfg: LocalizationConfig) -> int:
    """Run Qwen-VL localization and save both raw and comparable outputs."""
    from .vlm_inference import vlm_inference

    output_dir = cfg.ensure_output_dir()
    raw_json = output_dir / "vlm_detections.json"
    coco_json = output_dir / "vlm_output_coco.json"

    vlm_inference(
        cfg.image_path,
        str(output_dir / "vlm_result.jpg"),
        str(raw_json),
        cfg.detection_threshold,
        list(cfg.query_texts),
    )

    return _vlm_json_to_coco(str(raw_json), str(coco_json), cfg.image_id, cfg.category_id)


def compare_localizations(
    cfg: LocalizationConfig,
    pred_files: Iterable[str] | None = None,
    method_names: Sequence[str] = ("Model", "VLM"),
):
    """Compare localization methods against a COCO ground-truth annotation file."""
    from .evaluation import compare_coco_predictions

    if not cfg.gt_annotation_file:
        raise ValueError("gt_annotation_file is required for evaluation")

    output_dir = cfg.ensure_output_dir()
    if pred_files is None:
        pred_files = [
            str(output_dir / "model_detection_result.json"),
            str(output_dir / "vlm_output_coco.json"),
        ]

    return compare_coco_predictions(
        gt_annotation_file=cfg.gt_annotation_file,
        pred_files=list(pred_files),
        method_names=list(method_names),
        iou_threshold=cfg.iou_threshold,
        confidence_threshold=cfg.confidence_threshold,
        output_dir=str(output_dir),
    )
