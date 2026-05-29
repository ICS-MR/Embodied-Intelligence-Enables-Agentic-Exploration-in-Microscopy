import json
import re
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import base64
import requests
import time
import cv2
import os
import io

# ===== 配置参数 =====
API_KEY = "sk-fe8622aafd6f42df95d2274f996a8e68"
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
MODEL_NAME = "qwen3-vl-235b-a22b-instruct"

# 最大允许输入边长（Qwen-VL 推荐 ≤ 4096）
MAX_INPUT_SIZE = 512
# 图像质量压缩参数（0-100，越低质量越小）
IMAGE_QUALITY = 80  # 可根据需要调整，建议 70-90
# 是否启用渐进式JPEG（进一步减小体积）
PROGRESSIVE_JPEG = True


def encode_image_from_pil(image, quality=95, progressive=False):
    """从PIL图像直接编码为base64，同时压缩质量"""
    try:
        # 创建字节流
        img_byte_arr = io.BytesIO()
        # 保存为JPEG并设置质量参数
        image.save(
            img_byte_arr,
            format='JPEG',
            quality=quality,
            progressive=progressive,
            optimize=True  # 启用优化
        )
        # 重置指针到开头
        img_byte_arr.seek(0)
        # 编码为base64
        return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"❌ 图像编码失败: {str(e)}")
        return None


def encode_image(image_path):
    """兼容原有接口的文件编码函数（也加入质量压缩）"""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # 确保为RGB模式
            return encode_image_from_pil(img, IMAGE_QUALITY, PROGRESSIVE_JPEG)
    except Exception as e:
        print(f"❌ 读取并编码图像文件失败: {str(e)}")
        return None


def call_qwen_vl_api(image_b64, queries):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_b64}"
                    }
                },
                {
                    "type": "text",
                    "text": f"请精确检测图中以下类别的物体：{', '.join(queries)}。要求：\n"
                            """输出必须是严格符合 JSON 数组格式的字符串，且仅包含以下字段：
                            - "label": 类别名称（字符串）
                            - "bbox": 边界框 [x_min, y_min, x_max, y_max]（整数列表）
                            不要输出任何额外文字、解释、代码块标记（如 ```json）、换行符或空格。
                            输出示例：[
                            {
                                "label":"cell",
                                "bbox":[100,200,300,400]
                            }
                            ]"""
                }
            ]
        }
    ]

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.01,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        else:
            print(f"API返回异常: {json.dumps(result, indent=2)}")
            return None

    except Exception as e:
        print(f"API请求失败: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"API错误详情: {e.response.text}")
        return None


def parse_detection_results(api_response, image_width, image_height, detection_threshold):
    """解析 API 返回，将 Qwen-VL 的 0-999 坐标转为实际像素坐标"""
    try:
        if not api_response:
            return [], [], []

        # 提取 JSON
        json_match = re.search(r'(\[.*\])', api_response, re.DOTALL)
        if not json_match:
            print("❌ 无法找到有效的JSON结构")
            return [], [], []

        detections = json.loads(json_match.group(1))

        formatted_detections = []
        for det in detections:
            if "bbox_2d" in det:
                formatted_detections.append({"label": "cell", "bbox": det["bbox_2d"]})
            elif "label" in det and "bbox" in det:
                formatted_detections.append({"label": det["label"], "bbox": det["bbox"]})

        boxes = []
        scores = []
        labels = []

        for det in formatted_detections:
            bbox = det["bbox"]
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue

            orig_x_min, orig_y_min, orig_x_max, orig_y_max = [float(x) for x in bbox]
            label = det["label"]

            # Qwen-VL 返回的是 0-999 的相对坐标
            x_min = (orig_x_min / 999.0) * image_width
            y_min = (orig_y_min / 999.0) * image_height
            x_max = (orig_x_max / 999.0) * image_width
            y_max = (orig_y_max / 999.0) * image_height

            # 边界裁剪
            x_min = max(0, min(x_min, image_width - 1))
            y_min = max(0, min(y_min, image_height - 1))
            x_max = max(1, min(x_max, image_width))
            y_max = max(1, min(y_max, image_height))

            if x_min >= x_max or y_min >= y_max:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(1.0)  # 所有 score 为 1.0
            labels.append(label)

        return boxes, scores, labels

    except Exception as e:
        print(f"🔥 解析检测结果时发生错误: {str(e)}")
        return [], [], []


def draw_boxes(image, boxes, scores, labels):
    """在图像上绘制检测框（支持中文）"""
    draw = ImageDraw.Draw(image)

    font = None
    font_paths = [
        "simhei.ttf",
        "/System/Library/Fonts/PingFang.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "arial.ttf"
    ]
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 16)
            break
        except:
            continue
    if font is None:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        draw.rectangle(box, outline=(0, 255, 0), width=2)
        label_text = f"{label} {score:.2f}"  # 增加类别显示
        try:
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x0, y0 = box[0], box[1] - text_height - 4
            draw.rectangle([x0 - 2, y0 - 2, x0 + text_width + 2, y0 + text_height + 2], fill=(0, 0, 0, 180))
            draw.text((x0, y0), label_text, fill="white", font=font)
        except:
            # fallback if textbbox fails
            draw.text((box[0], box[1] - 20), label_text, fill="white", font=font)

    return image


def save_results(boxes, scores, labels, output_json):
    results = []
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        results.append({
            "id": i + 1,
            "label": label,
            "confidence": round(float(score), 4),
            "bbox": {
                "x_min": round(float(box[0]), 2),
                "y_min": round(float(box[1]), 2),
                "x_max": round(float(box[2]), 2),
                "y_max": round(float(box[3]), 2)
            }
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 检测结果已保存到: {os.path.abspath(output_json)}")


def vlm_inference(IMAGE_PATH, OUTPUT_IMAGE, OUTPUT_JSON, DETECTION_THRESHOLD, QUERY_TEXTS):
    print("===== Qwen-VL 细胞检测系统 (缩放+质量压缩) =====")
    print(f"🔬 检测类别: {', '.join(QUERY_TEXTS)}")
    print(f"🎯 置信度阈值: {DETECTION_THRESHOLD}")
    print(f"🖼️ 图像压缩质量: {IMAGE_QUALITY}%")
    print(f"⚡ 渐进式JPEG: {'开启' if PROGRESSIVE_JPEG else '关闭'}")

    # 1. 读取原始图像
    try:
        original_image = Image.open(IMAGE_PATH).convert("RGB")
        orig_w, orig_h = original_image.size
        orig_size = os.path.getsize(IMAGE_PATH) / 1024 / 1024  # MB
        print(f"📊 原始图像: {orig_w}x{orig_h}, 大小: {orig_size:.2f} MB")
    except Exception as e:
        print(f"❌ 打开图像失败: {e}")
        print(f"🔍 请检查文件是否存在: {os.path.abspath(IMAGE_PATH)}")
        return

    # 2. 缩放图像（如过大）
    scale_factor = 1.0
    if max(orig_w, orig_h) > MAX_INPUT_SIZE:
        scale_factor = MAX_INPUT_SIZE / max(orig_w, orig_h)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        input_image = original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"🔄 图像过大，已缩放至: {new_w}x{new_h} (缩放因子: {scale_factor:.4f})")
    else:
        input_image = original_image
        new_w, new_h = orig_w, orig_h
        print("✅ 图像尺寸在允许范围内，无需缩放")

    # 3. 直接编码缩放后的图像（带质量压缩）
    print("🔄 正在编码图像（带质量压缩）...")
    image_b64 = encode_image_from_pil(input_image, IMAGE_QUALITY, PROGRESSIVE_JPEG)

    if not image_b64:
        print("❌ 图像编码失败，终止流程")
        return

    # 计算压缩后的大小
    compressed_size = len(image_b64) * 3 / 4 / 1024  # base64编码后体积增加约33%
    print(f"📦 压缩后图像大小: {compressed_size:.2f} KB")

    # 4. 调用 API
    print("🔄 正在调用 Qwen-VL API...")
    start_time = time.time()
    api_response = call_qwen_vl_api(image_b64, QUERY_TEXTS)
    elapsed = time.time() - start_time
    print(f"⏱️ API 响应时间: {elapsed:.2f} 秒")

    if api_response is None:
        print("❌ API 未返回有效结果，终止流程")
        return

    # 5. 解析检测结果（基于缩放后图像）
    boxes, scores, labels = parse_detection_results(api_response, new_w, new_h, DETECTION_THRESHOLD)
    print(f"✅ 在缩放图像上检测到 {len(boxes)} 个目标")

    # 6. 坐标还原到原始图像
    if scale_factor != 1.0:
        print("🔄 正在将检测框坐标还原到原始图像尺寸...")
        original_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            original_boxes.append([
                x1 / scale_factor,
                y1 / scale_factor,
                x2 / scale_factor,
                y2 / scale_factor
            ])
        boxes = original_boxes

    # 7. 保存结果
    save_results(boxes, scores, labels, OUTPUT_JSON)

    # 8. 绘制并保存标注图
    print("🖼️ 正在绘制检测结果...")
    result_image = draw_boxes(original_image.copy(), boxes, scores, labels)
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_IMAGE)), exist_ok=True)
    # 保存结果图时也可压缩
    result_image.save(
        OUTPUT_IMAGE,
        quality=90,  # 结果图质量稍高
        progressive=True,
        optimize=True
    )
    print(f"🖼️ 标注图像已保存至: {os.path.abspath(OUTPUT_IMAGE)}")

    # 9. 显示结果（可选）
    try:
        cv_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        screen_width = 1920
        max_width = min(1200, screen_width - 100)
        scale = max_width / cv_image.shape[1]
        new_height = int(cv_image.shape[0] * scale)
        cv_image_resized = cv2.resize(cv_image, (max_width, new_height))
        cv2.imshow("Qwen-VL 检测结果 (缩放+质量压缩)", cv_image_resized)
        print("\n🖼️ 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"⚠️ 显示图像失败（不影响保存）: {e}")


if __name__ == "__main__":
    # ===== 用户配置区 =====
    IMAGE_PATH = r"pos_4x.ome.png"
    OUTPUT_IMAGE = "output_scaled.jpg"
    OUTPUT_JSON = "detections_scaled.json"
    DETECTION_THRESHOLD = 0.3  # 实际未用于过滤（因 score 恒为 1.0），保留兼容性
    QUERY_TEXTS = ["cell"]

    # 可在这里临时调整压缩参数
    # IMAGE_QUALITY = 75  # 更低的质量，更小的体积
    # PROGRESSIVE_JPEG = True

    vlm_inference(IMAGE_PATH, OUTPUT_IMAGE, OUTPUT_JSON, DETECTION_THRESHOLD, QUERY_TEXTS)