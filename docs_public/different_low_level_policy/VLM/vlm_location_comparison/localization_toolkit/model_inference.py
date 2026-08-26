from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
import cv2
import json
import os
import numpy as np
from pathlib import Path


def detect_and_save_tiled_mmdet(
    config_file,
    checkpoint_file,
    img_path,
    output_json="detection_result.json",
    output_img="result.jpg",
    device='cuda:0',
    score_thr=0.5,
    nms_thr=0.5,
    tile_size=1024,
    overlap=128,
    image_id=0,
    pad_to_tile_size=True,  # Pad edge tiles to tile_size.
    category_id=0,
):
    """
    Run tiled MMDetection inference with padding, global NMS, and COCO output.
    """
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if not 0.0 <= float(score_thr) <= 1.0:
        raise ValueError("score_thr must be between 0 and 1")
    if not 0.0 <= float(nms_thr) <= 1.0:
        raise ValueError("nms_thr must be between 0 and 1")
    for path_value, label in (
        (config_file, "MMDetection config"),
        (checkpoint_file, "MMDetection checkpoint"),
        (img_path, "input image"),
    ):
        if not path_value or not Path(path_value).is_file():
            raise FileNotFoundError(f"{label} does not exist: {path_value}")

    print("Initializing MMDetection model...")
    model = init_detector(config_file, checkpoint_file, device=device)

    print(f"Reading image: {img_path}")
    full_img = cv2.imread(img_path)
    if full_img is None:
        raise ValueError(f"Unable to read image: {img_path}")
    orig_h, orig_w = full_img.shape[:2]
    print(f"   Image size: {orig_w} x {orig_h}")

    step = tile_size - overlap
    num_cols = (orig_w + step - 1) // step
    num_rows = (orig_h + step - 1) // step
    print(f"Tiling: {num_rows} rows x {num_cols} columns (tile={tile_size}, overlap={overlap})")

    all_detections = []

    tile_idx = 0
    for y in range(num_rows):
        for x in range(num_cols):
            tile_idx += 1

            x_min = x * step
            y_min = y * step
            x_max = min(x_min + tile_size, orig_w)
            y_max = min(y_min + tile_size, orig_h)

            tile_img = full_img[y_min:y_max, x_min:x_max]
            if tile_img.size == 0:
                continue

            # Optionally pad edge tiles to the configured size.
            pad_h = tile_size - tile_img.shape[0]
            pad_w = tile_size - tile_img.shape[1]
            if pad_to_tile_size and (pad_h > 0 or pad_w > 0):
                tile_img = cv2.copyMakeBorder(
                    tile_img, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

            # Run inference on this tile.
            try:
                result = inference_detector(model, tile_img)
            except Exception as exc:
                raise RuntimeError(
                    f"MMDetection inference failed for tile {tile_idx} "
                    f"at x={x_min}:{x_max}, y={y_min}:{y_max}"
                ) from exc

            pred = result.pred_instances.cpu()
            mask = pred.scores >= score_thr
            if mask.sum() == 0:
                continue

            bboxes = pred.bboxes[mask].numpy()  # [N, 4] in (x1, y1, x2, y2)
            scores = pred.scores[mask].numpy()
            labels = pred.labels[mask].numpy().astype(int)

            # Clip padded detections back to the unpadded tile bounds.
            if pad_to_tile_size:
                bboxes[:, [2, 3]] = np.minimum(bboxes[:, [2, 3]], [x_max - x_min, y_max - y_min])

            all_detections.append({
                'boxes': bboxes.tolist(),
                'scores': scores.tolist(),
                'labels': labels.tolist(),
                'x_off': x_min,
                'y_off': y_min
            })

            print(f"   Tile {tile_idx}: detected {len(bboxes)} objects")

    # Merge detections into the full-image coordinate system.
    merged_boxes = []
    merged_scores = []
    merged_labels = []

    for det in all_detections:
        x_off, y_off = det['x_off'], det['y_off']
        for bbox, score, label in zip(det['boxes'], det['scores'], det['labels']):
            x1, y1, x2, y2 = bbox
            # Keep boxes within the original image bounds.
            x1 = max(0, x1 + x_off)
            y1 = max(0, y1 + y_off)
            x2 = min(orig_w, x2 + x_off)
            y2 = min(orig_h, y2 + y_off)
            if x2 <= x1 or y2 <= y1:
                continue
            merged_boxes.append([x1, y1, x2, y2])
            merged_scores.append(float(score))
            merged_labels.append(int(label))

    print(f"Merged {len(merged_boxes)} boxes before NMS")

    # Apply global NMS.
    if not merged_boxes:
        kept_boxes, kept_scores, kept_labels = [], [], []
    else:
        boxes_np = np.array(merged_boxes, dtype=np.float32)
        scores_np = np.array(merged_scores, dtype=np.float32)

        # OpenCV NMS expects boxes in xywh format.
        boxes_xywh = boxes_np.copy()
        boxes_xywh[:, 2] = boxes_np[:, 2] - boxes_np[:, 0]
        boxes_xywh[:, 3] = boxes_np[:, 3] - boxes_np[:, 1]
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh.tolist(),
            scores=scores_np.tolist(),
            score_threshold=max(0.0, score_thr - 0.05),
            nms_threshold=nms_thr
        )

        # Normalize the index shape returned by different OpenCV versions.
        if len(indices) == 0:
            kept_boxes, kept_scores, kept_labels = [], [], []
        else:
            indices = np.asarray(indices).reshape(-1).astype(int).tolist()

            kept_boxes = [merged_boxes[i] for i in indices]
            kept_scores = [merged_scores[i] for i in indices]
            kept_labels = [merged_labels[i] for i in indices]

    print(f"Kept {len(kept_boxes)} boxes after NMS")

    # Save COCO-style predictions.
    json_results = []
    for box, score in zip(kept_boxes, kept_scores):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        json_results.append({
            "image_id": int(image_id),
            "category_id": int(category_id),
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "score": round(score, 4)
        })

    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved detection results to {output_json} ({len(json_results)} objects)")

    # Render a visualization.
    if kept_boxes:
        pred_instances = InstanceData()
        pred_instances.bboxes = np.array(kept_boxes, dtype=np.float32)
        pred_instances.scores = np.array(kept_scores, dtype=np.float32)
        pred_instances.labels = np.array(kept_labels, dtype=np.int64)

        final_result = DetDataSample()
        final_result.pred_instances = pred_instances

        visualizer = DetLocalVisualizer()
        # Dataset metadata is required to render class names.
        if hasattr(model, 'dataset_meta') and 'classes' in model.dataset_meta:
            visualizer.dataset_meta = model.dataset_meta
        else:
            # Fall back to generic class names when metadata is unavailable.
            visualizer.dataset_meta = {'classes': [f'class_{i}' for i in range(80)]}

        visualizer.add_datasample(
            name='result',
            image=full_img,
            data_sample=final_result,
            draw_gt=False,
            pred_score_thr=score_thr,
            show=False,
            out_file=output_img
        )
        print(f"Saved visualization to {output_img}")
    else:
        cv2.imwrite(output_img, full_img)
        print(f"No detections; saved the original image to {output_img}")

    return json_results, model
