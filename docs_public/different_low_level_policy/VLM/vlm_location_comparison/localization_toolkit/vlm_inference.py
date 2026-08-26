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
from pathlib import Path
from urllib.request import getproxies

# Fallback credentials used only when the local API config file is absent.
# Prefer filling docs_public/different_low_level_policy/VLM/vlm_api_config.json
# (copy it from vlm_api_config.example.json) instead of editing these.
API_KEY = "<your-vlm-api-key>"
API_URL = "<your-vlm-api-endpoint>"
MODEL_NAME = "<your-vlm-model-name>"

# Local API configuration file (gitignored):
# docs_public/different_low_level_policy/VLM/vlm_api_config.json
DEFAULT_API_CONFIG_PATH = Path(__file__).resolve().parents[2] / "vlm_api_config.json"

# Maximum input edge length.
MAX_INPUT_SIZE = 512
# JPEG quality from 0 to 100; lower values produce smaller payloads.
IMAGE_QUALITY = 80  # Recommended range: 70-90.
# Progressive JPEG further reduces the encoded payload size.
PROGRESSIVE_JPEG = True


def encode_image_from_pil(image, quality=95, progressive=False):
    """Encode a PIL image as a compressed base64 JPEG."""
    try:
        img_byte_arr = io.BytesIO()
        image.save(
            img_byte_arr,
            format='JPEG',
            quality=quality,
            progressive=progressive,
            optimize=True
        )
        img_byte_arr.seek(0)
        return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Image encoding failed: {e}") from e


def encode_image(image_path):
    """Read an image file and encode it as a compressed base64 JPEG."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return encode_image_from_pil(img, IMAGE_QUALITY, PROGRESSIVE_JPEG)
    except Exception as e:
        raise ValueError(f"Unable to read and encode image {os.path.abspath(image_path)}: {e}") from e


def _load_api_config(config_path=None):
    """Load VLM API credentials from the local config file, falling back to module constants.

    The config file is a JSON object with api_key / api_url / model_name keys.
    Pass `config_path` explicitly for testing; otherwise the default gitignored
    file under docs_public/different_low_level_policy/VLM/ is used.
    """
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"VLM API config file not found: {path}")
    else:
        path = DEFAULT_API_CONFIG_PATH

    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"VLM API config file is not valid JSON: {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"VLM API config file must contain a JSON object: {path}")
        values = {
            "API_KEY": str(payload.get("api_key") or "").strip(),
            "API_URL": str(payload.get("api_url") or "").strip(),
            "MODEL_NAME": str(payload.get("model_name") or "").strip(),
        }
        source = str(path)
    else:
        values = {
            "API_KEY": str(API_KEY).strip(),
            "API_URL": str(API_URL).strip(),
            "MODEL_NAME": str(MODEL_NAME).strip(),
        }
        source = "module constants"

    unconfigured = [
        name for name, value in values.items()
        if not value or (value.startswith("<") and value.endswith(">"))
    ]
    if unconfigured:
        raise RuntimeError(
            "VLM API configuration contains placeholders. Fill "
            f"{', '.join(unconfigured)} in the local config file "
            f"{DEFAULT_API_CONFIG_PATH} (copy from vlm_api_config.example.json)."
        )
    return values["API_KEY"], values["API_URL"], values["MODEL_NAME"]


def _proxy_summary():
    proxies = getproxies()
    if not proxies:
        return "none"
    redacted = {}
    for key, value in proxies.items():
        proxy_value = str(value)
        if "://" in proxy_value and "@" in proxy_value:
            scheme, rest = proxy_value.split("://", 1)
            proxy_value = f"{scheme}://***:***@{rest.split('@', 1)[1]}"
        redacted[key] = proxy_value
    return ", ".join(f"{key}={value}" for key, value in sorted(redacted.items()))


def call_qwen_vl_api(image_b64, queries, *, use_env_proxy=True):
    api_key, api_url, model_name = _load_api_config()
    headers = {
        "Authorization": f"Bearer {api_key}",
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
                    "text": f"Precisely detect objects in these categories: {', '.join(queries)}. Requirements:\n"
                            """Return a strict JSON object whose top-level "detections" field is an array.
                            Each detection must contain only these fields:
                            - "label": category name as a string
                            - "bbox": bounding box [x_min, y_min, x_max, y_max], normalized to 0-999
                            - "confidence": confidence score between 0 and 1
                            Do not return explanations, extra text, or code fences.
                            Example: {"detections":[{"label":"cell","bbox":[100,200,300,400],"confidence":0.9}]}"""
                }
            ]
        }
    ]

    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.01,
        "response_format": {"type": "json_object"}
    }

    # OpenAI-compatible chat endpoints expect the /chat/completions path; the
    # configured api_url may be a base URL (e.g. https://host/v1).
    endpoint = str(api_url).strip().rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"

    try:
        session = requests.Session()
        session.trust_env = bool(use_env_proxy)
        response = session.post(endpoint, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        raise RuntimeError(f"VLM API response has no choices: {json.dumps(result, ensure_ascii=False)}")

    except requests.RequestException as exc:
        detail = ""
        if exc.response is not None:
            detail = f" Response: {exc.response.text}"
        if isinstance(exc, requests.ProxyError):
            detail += (
                " Python proxy settings: "
                f"{_proxy_summary()}. Retry with --no-env-proxy to bypass environment proxies, "
                "or fix the local proxy before rerunning."
            )
        raise RuntimeError(f"VLM API request failed: {exc}.{detail}") from exc


def parse_detection_results(api_response, image_width, image_height, detection_threshold):
    """Parse a VLM response and convert normalized coordinates to pixels."""
    try:
        if not api_response:
            raise ValueError("VLM response is empty")

        cleaned = str(api_response).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            json_match = re.search(r'(\{.*\}|\[.*\])', cleaned, re.DOTALL)
            if not json_match:
                raise ValueError("VLM response does not contain valid JSON")
            payload = json.loads(json_match.group(1))

        if isinstance(payload, dict):
            detections = payload.get("detections", payload.get("objects", payload.get("results", [])))
        else:
            detections = payload
        if not isinstance(detections, list):
            raise ValueError("VLM detection payload must be a JSON array")

        formatted_detections = []
        for det in detections:
            if "bbox_2d" in det:
                formatted_detections.append({
                    "label": det.get("label", "cell"),
                    "bbox": det["bbox_2d"],
                    "confidence": det.get("confidence", det.get("score", 1.0)),
                })
            elif "label" in det and "bbox" in det:
                formatted_detections.append(det)

        boxes = []
        scores = []
        labels = []

        for det in formatted_detections:
            bbox = det["bbox"]
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue

            orig_x_min, orig_y_min, orig_x_max, orig_y_max = [float(x) for x in bbox]
            label = det["label"]
            score = float(det.get("confidence", det.get("score", 1.0)))
            if not 0.0 <= score <= 1.0 or score < float(detection_threshold):
                continue

            # The VLM returns relative coordinates in the 0-999 range.
            x_min = (orig_x_min / 999.0) * image_width
            y_min = (orig_y_min / 999.0) * image_height
            x_max = (orig_x_max / 999.0) * image_width
            y_max = (orig_y_max / 999.0) * image_height

            # Clip coordinates to the image bounds.
            x_min = max(0, min(x_min, image_width - 1))
            y_min = max(0, min(y_min, image_height - 1))
            x_max = max(1, min(x_max, image_width))
            y_max = max(1, min(y_max, image_height))

            if x_min >= x_max or y_min >= y_max:
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(score)
            labels.append(label)

        return boxes, scores, labels

    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"Failed to parse VLM detection response: {exc}") from exc


def draw_boxes(image, boxes, scores, labels):
    """Draw detection boxes and labels on an image."""
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
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    for box, score, label in zip(boxes, scores, labels):
        draw.rectangle(box, outline=(0, 255, 0), width=2)
        label_text = f"{label} {score:.2f}"
        try:
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x0, y0 = box[0], box[1] - text_height - 4
            draw.rectangle([x0 - 2, y0 - 2, x0 + text_width + 2, y0 + text_height + 2], fill=(0, 0, 0, 180))
            draw.text((x0, y0), label_text, fill="white", font=font)
        except (AttributeError, OSError, ValueError):
            # Older Pillow versions or unusual fonts can fail textbbox; keep the saved visualization usable.
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
    print(f"Saved detection results to {os.path.abspath(output_json)}")


def vlm_inference(
    IMAGE_PATH,
    OUTPUT_IMAGE,
    OUTPUT_JSON,
    DETECTION_THRESHOLD,
    QUERY_TEXTS,
    *,
    use_env_proxy=True,
    show_window=False,
):
    print("===== VLM localization (resize and JPEG compression) =====")
    print(f"Target categories: {', '.join(QUERY_TEXTS)}")
    print(f"Confidence threshold: {DETECTION_THRESHOLD}")
    print(f"JPEG quality: {IMAGE_QUALITY}%")
    print(f"Progressive JPEG: {'enabled' if PROGRESSIVE_JPEG else 'disabled'}")

    # 1. Read the original image.
    try:
        with Image.open(IMAGE_PATH) as source_image:
            original_image = source_image.convert("RGB")
        orig_w, orig_h = original_image.size
        orig_size = os.path.getsize(IMAGE_PATH) / 1024 / 1024  # MB
        print(f"Original image: {orig_w}x{orig_h}, {orig_size:.2f} MB")
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unable to open input image {os.path.abspath(IMAGE_PATH)}: {exc}") from exc

    # 2. Resize images that exceed the input limit.
    scale_factor = 1.0
    if max(orig_w, orig_h) > MAX_INPUT_SIZE:
        scale_factor = MAX_INPUT_SIZE / max(orig_w, orig_h)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        input_image = original_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        print(f"Resized image to {new_w}x{new_h} (scale factor: {scale_factor:.4f})")
    else:
        input_image = original_image
        new_w, new_h = orig_w, orig_h
        print("Image dimensions are within the input limit; no resize required")

    # 3. Encode the resized image with JPEG compression.
    print("Encoding image with JPEG compression...")
    image_b64 = encode_image_from_pil(input_image, IMAGE_QUALITY, PROGRESSIVE_JPEG)

    if not image_b64:
        raise RuntimeError("Image encoding returned an empty payload")

    # Estimate the decoded JPEG size from the base64 payload.
    compressed_size = len(image_b64) * 3 / 4 / 1024
    print(f"Compressed image size: {compressed_size:.2f} KB")

    # 4. Call the VLM API.
    print("Calling VLM API...")
    start_time = time.time()
    api_response = call_qwen_vl_api(image_b64, QUERY_TEXTS, use_env_proxy=use_env_proxy)
    elapsed = time.time() - start_time
    print(f"API response time: {elapsed:.2f} seconds")

    # 5. Parse detections in resized-image coordinates.
    boxes, scores, labels = parse_detection_results(api_response, new_w, new_h, DETECTION_THRESHOLD)
    print(f"Detected {len(boxes)} objects in the resized image")

    # 6. Map coordinates back to the original image.
    if scale_factor != 1.0:
        print("Mapping detection coordinates to the original image size...")
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

    # 7. Save structured results.
    save_results(boxes, scores, labels, OUTPUT_JSON)

    # 8. Draw and save the annotated image.
    print("Drawing detection results...")
    result_image = draw_boxes(original_image.copy(), boxes, scores, labels)
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_IMAGE)), exist_ok=True)
    result_image.save(
        OUTPUT_IMAGE,
        quality=90,
        progressive=True,
        optimize=True
    )
    print(f"Saved annotated image to {os.path.abspath(OUTPUT_IMAGE)}")

    if show_window:
        try:
            cv_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            screen_width = 1920
            max_width = min(1200, screen_width - 100)
            scale = max_width / cv_image.shape[1]
            new_height = int(cv_image.shape[0] * scale)
            cv_image_resized = cv2.resize(cv_image, (max_width, new_height))
            cv2.imshow("VLM detection results", cv_image_resized)
            print("\nPress any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as exc:
            print(f"Unable to display image; saved outputs are unaffected: {exc}")

    return len(boxes)
