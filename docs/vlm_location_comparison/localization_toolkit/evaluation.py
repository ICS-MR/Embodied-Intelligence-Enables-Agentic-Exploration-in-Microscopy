"""Comparison utilities hidden behind the public compare command."""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm

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
    pred_by_group: dict[tuple[int, int], list[dict]] = {}
    for pred in pred_results:
        if pred["score"] < confidence_threshold:
            continue
        key = (int(pred["image_id"]), int(pred["category_id"]))
        pred_by_group.setdefault(key, []).append(pred)

    gt_by_group: dict[tuple[int, int], list[dict]] = {}
    for ann in coco_gt.dataset.get("annotations", []):
        key = (int(ann["image_id"]), int(ann["category_id"]))
        gt_by_group.setdefault(key, []).append(ann)

    area_errors = []
    center_distances = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    group_keys = sorted(set(gt_by_group) | set(pred_by_group))
    for key in tqdm(group_keys, desc=f"Evaluating {method_name}", leave=False):
        gt_anns = gt_by_group.get(key, [])
        preds = pred_by_group.get(key, [])
        if not gt_anns:
            false_positives += len(preds)
            continue
        if not preds:
            false_negatives += len(gt_anns)
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
            true_positives += 1

            pred_area = pred_boxes[pred_idx][2] * pred_boxes[pred_idx][3]
            gt_area = gt_areas[gt_idx]
            area_errors.append(abs(pred_area - gt_area) / gt_area if gt_area > 0 else 0)

            pred_center = _center(pred_boxes[pred_idx])
            gt_center = _center(gt_boxes[gt_idx])
            center_distances.append(
                float(np.sqrt((pred_center[0] - gt_center[0]) ** 2 + (pred_center[1] - gt_center[1]) ** 2))
            )

        false_positives += len(preds) - len(matched_pred)
        false_negatives += len(gt_anns) - len(matched_gt)

    prediction_count = sum(len(preds) for preds in pred_by_group.values())
    gt_count = sum(len(anns) for anns in gt_by_group.values())
    precision = true_positives / prediction_count if prediction_count else 0.0
    recall = true_positives / gt_count if gt_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "mape": float(np.mean(area_errors)) if area_errors else None,
        "mean_center_distance": float(np.mean(center_distances)) if center_distances else None,
        "median_center_distance": float(np.median(center_distances)) if center_distances else None,
        "gt_count": gt_count,
        "prediction_count": prediction_count,
        "true_positive": true_positives,
        "false_positive": false_positives,
        "false_negative": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_count": true_positives,
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
    plt.xlabel("Relative area error")
    plt.ylabel("Density")
    plt.title("Area error distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for name in method_names:
        if results[name]["center_distances"]:
            plt.hist(results[name]["center_distances"], bins=50, alpha=0.7, label=name, density=True)
    plt.xlabel("Center distance (pixels)")
    plt.ylabel("Density")
    plt.title("Center error distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_analysis.png"), dpi=150, bbox_inches="tight")

    with open(os.path.join(output_dir, "error_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results
