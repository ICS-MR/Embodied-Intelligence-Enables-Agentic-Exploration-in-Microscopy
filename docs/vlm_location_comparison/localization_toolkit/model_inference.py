# mmdet_tiled_inference_clean.py
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
    pad_to_tile_size=True  # 是否将小 tile padding 到 tile_size
):
    """
    使用 MMDetection 进行分块检测，支持 padding、全局 NMS、COCO 格式输出。
    """
    print("🔧 初始化 MMDetection 模型...")
    model = init_detector(config_file, checkpoint_file, device=device)

    print(f"🖼️ 读取图像: {img_path}")
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"图像不存在: {img_path}")
    full_img = cv2.imread(img_path)
    if full_img is None:
        raise ValueError(f"无法读取图像: {img_path}")
    orig_h, orig_w = full_img.shape[:2]
    print(f"   图像尺寸: {orig_w} x {orig_h}")

    step = tile_size - overlap
    num_cols = (orig_w + step - 1) // step
    num_rows = (orig_h + step - 1) // step
    print(f"✂️ 分块: {num_rows} 行 × {num_cols} 列 (tile={tile_size}, overlap={overlap})")

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

            # 👉 可选：padding 到 tile_size（避免模型对小图报错）
            pad_h = tile_size - tile_img.shape[0]
            pad_w = tile_size - tile_img.shape[1]
            if pad_to_tile_size and (pad_h > 0 or pad_w > 0):
                tile_img = cv2.copyMakeBorder(
                    tile_img, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

            # 推理
            try:
                result = inference_detector(model, tile_img)
            except Exception as e:
                print(f"   ⚠️ Tile {tile_idx} 推理异常: {e}")
                continue

            pred = result.pred_instances.cpu()
            mask = pred.scores >= score_thr
            if mask.sum() == 0:
                continue

            bboxes = pred.bboxes[mask].numpy()  # [N, 4] in (x1, y1, x2, y2)
            scores = pred.scores[mask].numpy()
            labels = pred.labels[mask].numpy().astype(int)

            # 如果做了 padding，需裁剪回原始 tile 范围（仅 x2,y2 可能超）
            if pad_to_tile_size:
                bboxes[:, [2, 3]] = np.minimum(bboxes[:, [2, 3]], [x_max - x_min, y_max - y_min])

            all_detections.append({
                'boxes': bboxes.tolist(),
                'scores': scores.tolist(),
                'labels': labels.tolist(),
                'x_off': x_min,
                'y_off': y_min
            })

            print(f"   🧩 Tile {tile_idx}: 检测到 {len(bboxes)} 个目标")

    # === 合并到原图坐标系 ===
    merged_boxes = []
    merged_scores = []
    merged_labels = []

    for det in all_detections:
        x_off, y_off = det['x_off'], det['y_off']
        for bbox, score, label in zip(det['boxes'], det['scores'], det['labels']):
            x1, y1, x2, y2 = bbox
            # 保证不超出原图边界（可选）
            x1 = max(0, x1 + x_off)
            y1 = max(0, y1 + y_off)
            x2 = min(orig_w, x2 + x_off)
            y2 = min(orig_h, y2 + y_off)
            if x2 <= x1 or y2 <= y1:
                continue
            merged_boxes.append([x1, y1, x2, y2])
            merged_scores.append(float(score))
            merged_labels.append(int(label))

    print(f"📊 合并前共 {len(merged_boxes)} 个检测框")

    # === 全局 NMS ===
    if not merged_boxes:
        kept_boxes, kept_scores, kept_labels = [], [], []
    else:
        boxes_np = np.array(merged_boxes, dtype=np.float32)
        scores_np = np.array(merged_scores, dtype=np.float32)

        # 使用 cv2 NMS（兼容不同版本）
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes_np.tolist(),
            scores=scores_np.tolist(),
            score_threshold=score_thr - 0.05,  # 略低于阈值，防止漏框
            nms_threshold=nms_thr
        )

        # 处理 OpenCV 版本差异
        if len(indices) == 0:
            kept_boxes, kept_scores, kept_labels = [], [], []
        else:
            if isinstance(indices[0], (list, np.ndarray)):
                indices = [i[0] for i in indices]
            else:
                indices = indices.flatten().tolist()

            kept_boxes = [merged_boxes[i] for i in indices]
            kept_scores = [merged_scores[i] for i in indices]
            kept_labels = [merged_labels[i] for i in indices]

    print(f"✅ NMS 后保留 {len(kept_boxes)} 个检测框")

    # === 保存 COCO 格式 JSON ===
    json_results = []
    for box, score, label in zip(kept_boxes, kept_scores, kept_labels):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        # ⚠️ 注意：MMDetection 默认类别从 0 开始，COCO 通常从 1 开始？
        # 如果你的模型训练时类别是 0-based，且 COCO label 也是 0-based，则保留。
        # 否则需 +1。这里假设 config 中 category_id 与模型一致。
        json_results.append({
            "image_id": int(image_id),
            "category_id": int(label),  # 如需 +1，改为 label + 1
            "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
            "score": round(score, 4)
        })

    os.makedirs(os.path.dirname(output_json) if os.path.dirname(output_json) else '.', exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    print(f"✅ 检测结果已保存: {output_json} ({len(json_results)} 个目标)")

    # === 可视化 ===
    if kept_boxes:
        pred_instances = InstanceData()
        pred_instances.bboxes = np.array(kept_boxes, dtype=np.float32)
        pred_instances.scores = np.array(kept_scores, dtype=np.float32)
        pred_instances.labels = np.array(kept_labels, dtype=np.int64)

        final_result = DetDataSample()
        final_result.pred_instances = pred_instances

        visualizer = DetLocalVisualizer()
        # 👉 关键：设置 classes 才能显示类别名
        if hasattr(model, 'dataset_meta') and 'classes' in model.dataset_meta:
            visualizer.dataset_meta = model.dataset_meta
        else:
            # 若未定义，尝试从 checkpoint 或 config 推断（简化处理）
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
        print(f"✅ 可视化图像已保存: {output_img}")
    else:
        cv2.imwrite(output_img, full_img)
        print(f"⚠️ 无检测结果，原图已保存: {output_img}")

    return json_results, model


# ========== 使用示例 ==========
if __name__ == "__main__":
    config_file = r'work_dirs/2D_/2D_.py'
    checkpoint_file = r'work_dirs/2D_/epoch_50.pth'
    img_path = r"result_9blocks.jpg"

    detect_and_save_tiled_mmdet(
        config_file=config_file,
        checkpoint_file=checkpoint_file,
        img_path=img_path,
        output_json="detection_result.json",
        output_img="detection_result.jpg",
        device='cuda:0',
        score_thr=0.5,
        nms_thr=0.5,
        tile_size=2048,
        overlap=128,
        image_id=1,
        pad_to_tile_size=True  # 推荐开启
    )