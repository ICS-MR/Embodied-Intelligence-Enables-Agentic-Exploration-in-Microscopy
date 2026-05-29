"""Comparison utilities hidden behind the public compare command."""

from __future__ import annotations

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def _compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    ious = np.zeros((len(boxes1), len(boxes2)))
    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = area1 + area2 - inter
            ious[i, j] = inter / union if union > 0 else 0
    return ious


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = boxes[:, 0]
    xyxy[:, 1] = boxes[:, 1]
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
    return xyxy


def _center(box: np.ndarray) -> tuple[float, float]:
    return box[0] + box[2] / 2, box[1] + box[3] / 2


def _evaluate_one_method(
    pred_results: list[dict],
    coco_gt: COCO,
    method_name: str,
    iou_threshold: float,
    confidence_threshold: float,
) -> dict:
    pred_by_image: dict[int, list[dict]] = {}
    for pred in pred_results:
        if pred["score"] < confidence_threshold:
            continue
        pred_by_image.setdefault(pred["image_id"], []).append(pred)

    area_errors = []
    center_distances = []

    for img_id in tqdm(coco_gt.imgs.keys(), desc=f"Evaluating {method_name}", leave=False):
        gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))
        preds = pred_by_image.get(img_id, [])
        if not gt_anns or not preds:
            continue

        gt_boxes = np.array([ann["bbox"] for ann in gt_anns])
        gt_areas = np.array([ann["area"] for ann in gt_anns])
        pred_boxes = np.array([pred["bbox"] for pred in preds])

        iou_matrix = _compute_iou_matrix(_xywh_to_xyxy(pred_boxes), _xywh_to_xyxy(gt_boxes))
        pairs = [(iou_matrix[i, j], i, j) for i in range(len(pred_boxes)) for j in range(len(gt_boxes))]
        pairs.sort(key=lambda item: item[0], reverse=True)

        matched_pred = set()
        matched_gt = set()
        for iou, pred_idx, gt_idx in pairs:
            if iou < iou_threshold:
                break
            if pred_idx in matched_pred or gt_idx in matched_gt:
                continue
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)

            pred_area = pred_boxes[pred_idx][2] * pred_boxes[pred_idx][3]
            gt_area = gt_areas[gt_idx]
            area_errors.append(abs(pred_area - gt_area) / gt_area if gt_area > 0 else 0)

            pred_center = _center(pred_boxes[pred_idx])
            gt_center = _center(gt_boxes[gt_idx])
            center_distances.append(
                float(np.sqrt((pred_center[0] - gt_center[0]) ** 2 + (pred_center[1] - gt_center[1]) ** 2))
            )

    return {
        "mape": float(np.mean(area_errors)) if area_errors else 0.0,
        "mean_center_distance": float(np.mean(center_distances)) if center_distances else 0.0,
        "median_center_distance": float(np.median(center_distances)) if center_distances else 0.0,
        "matched_count": len(area_errors),
        "area_errors": area_errors,
        "center_distances": center_distances,
    }


def compare_coco_predictions(
    gt_annotation_file: str,
    pred_files: list[str],
    method_names: list[str],
    iou_threshold: float = 0.5,
    confidence_threshold: float = 0.3,
    output_dir: str = "localization_output",
) -> dict:
    """Compare COCO-style prediction files and save metrics/plots."""
    if len(pred_files) != len(method_names):
        raise ValueError("pred_files and method_names must have the same length")

    os.makedirs(output_dir, exist_ok=True)
    coco_gt = COCO(gt_annotation_file)

    results = {}
    for pred_file, name in zip(pred_files, method_names):
        with open(pred_file, "r", encoding="utf-8") as f:
            preds = json.load(f)
        results[name] = _evaluate_one_method(
            preds,
            coco_gt,
            name,
            iou_threshold=iou_threshold,
            confidence_threshold=confidence_threshold,
        )

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for name in method_names:
        if results[name]["area_errors"]:
            plt.hist(results[name]["area_errors"], bins=50, alpha=0.7, label=name, density=True)
    plt.xlabel("面积相对误差")
    plt.ylabel("密度")
    plt.title("面积误差分布")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for name in method_names:
        if results[name]["center_distances"]:
            plt.hist(results[name]["center_distances"], bins=50, alpha=0.7, label=name, density=True)
    plt.xlabel("中心点距离 (像素)")
    plt.ylabel("密度")
    plt.title("中心点误差分布")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_analysis.png"), dpi=150, bbox_inches="tight")

    with open(os.path.join(output_dir, "error_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
