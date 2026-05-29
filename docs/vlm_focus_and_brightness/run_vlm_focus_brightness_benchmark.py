from __future__ import annotations

import base64
import csv
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bootstrap.config import load_runtime_settings
from utils.runtime_core import release_resources, setup_microscope
from utils.runtime_factory import initialize_system_components

try:
    RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_LANCZOS = Image.LANCZOS


# ====================== Run configuration ======================
# Edit this block directly, then run:
#   .venv\Scripts\python.exe scripts\run_vlm_focus_brightness_benchmark.py
RUN_CONFIG: dict[str, Any] = {
    "enabled": True,
    "mode": "both",  # "focus", "brightness", or "both"
    "trial_count": 1,
    "show_preview_window": True,
    "output_dir": "test_docx/vlm_focus_brightness",
    "channel": "1-NONE",
    "exposure_ms": 10.0,
    "brightness": 100,
    "capture_source": "snap",  # "snap" uses synchronized acquisition; "preview" uses live preview cache.
    "initial_z_um": None,
    "preview_timeout_seconds": 8.0,
    "settle_seconds": 0.35,
    "mosaic_subimage_size_px": 360,
    "vlm_temperature": 0.0,
    "vlm_max_tokens": 80,
    "focus": {
        "enabled": True,
        "initial_step_um": 50.0,
        "min_step_um": 2.0,
        "max_iterations": 4,
        "candidate_offsets": [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    },
    "brightness_search": {
        "enabled": True,
        "initial_step": 10,
        "min_step": 1,
        "max_iterations": 4,
        "candidate_offsets": [-4, -3, -2, -1, 0, 1, 2, 3, 4],
    },
}


POSITIONS_EN = [
    "top-left",
    "top-center",
    "top-right",
    "middle-left",
    "center",
    "middle-right",
    "bottom-left",
    "bottom-center",
    "bottom-right",
]


@dataclass
class Candidate:
    filename: str
    value: float
    image_path: Path


def say(message: str) -> None:
    print(message, flush=True)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(value), float(upper)))


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "unknown")).strip("-") or "unknown"


def wait_for_preview_frame(microscope: Any, timeout_seconds: float) -> np.ndarray:
    deadline = time.monotonic() + float(timeout_seconds)
    while time.monotonic() < deadline:
        frame = microscope.get_live_preview_image()
        if frame is not None:
            return np.asarray(frame)
        time.sleep(0.1)
    raise RuntimeError("Live preview did not produce a frame within the timeout window")


def capture_frame(microscope: Any, capture_source: str, preview_timeout_seconds: float) -> np.ndarray:
    source = str(capture_source or "snap").lower()
    if source == "snap":
        if not hasattr(microscope, "_snap_image_preserving_preview"):
            raise RuntimeError("Microscope controller does not expose _snap_image_preserving_preview().")
        return np.asarray(microscope._snap_image_preserving_preview())
    if source == "preview":
        return wait_for_preview_frame(microscope, preview_timeout_seconds)
    raise ValueError(f"Unsupported capture_source: {capture_source!r}. Use 'snap' or 'preview'.")


def normalize_image_for_jpeg(image: np.ndarray) -> Image.Image:
    arr = np.asarray(image)
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]

    if np.issubdtype(arr.dtype, np.integer) and arr.dtype == np.uint8:
        out = arr
    else:
        arr_float = arr.astype(np.float32, copy=False)
        finite = arr_float[np.isfinite(arr_float)]
        if finite.size == 0:
            out = np.zeros(arr_float.shape, dtype=np.uint8)
        else:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            if hi <= lo:
                lo = float(np.min(finite))
                hi = float(np.max(finite))
            scaled = (arr_float - lo) / (hi - lo + 1e-8)
            out = np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

    if out.ndim == 2:
        return Image.fromarray(out, mode="L").convert("RGB")
    if out.ndim == 3:
        return Image.fromarray(out).convert("RGB")
    raise ValueError(f"Unsupported preview frame shape: {arr.shape}")


def save_preview_frame(image: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalize_image_for_jpeg(image).save(output_path, "JPEG", quality=90, optimize=True)


def create_mosaic(candidates: list[Candidate], output_path: Path, subimage_size_px: int) -> None:
    if len(candidates) != 9:
        raise ValueError(f"Expected exactly 9 candidates, got {len(candidates)}")
    tile_size = int(subimage_size_px)
    border = 2
    label_h = 24
    cell_w = tile_size + border * 2
    cell_h = tile_size + border * 2 + label_h
    mosaic = Image.new("RGB", (cell_w * 3, cell_h * 3), (235, 235, 235))
    draw = ImageDraw.Draw(mosaic)

    for idx, candidate in enumerate(candidates):
        with Image.open(candidate.image_path) as opened_image:
            image = opened_image.convert("RGB")
        image.thumbnail((tile_size, tile_size), RESAMPLE_LANCZOS)
        tile = Image.new("RGB", (tile_size, tile_size), (255, 255, 255))
        x_pad = (tile_size - image.width) // 2
        y_pad = (tile_size - image.height) // 2
        tile.paste(image, (x_pad, y_pad))

        x = (idx % 3) * cell_w
        y = (idx // 3) * cell_h
        mosaic.paste(tile, (x + border, y + border))
        draw.rectangle([x, y, x + cell_w - 1, y + cell_h - 1], outline=(220, 0, 0), width=border)
        draw.text((x + 6, y + tile_size + border + 4), candidate.filename, fill=(0, 0, 0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(output_path, "JPEG", quality=88, optimize=True)


def image_to_data_url(path: Path) -> str:
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{payload}"


def extract_candidate_name(raw_text: str, valid_names: Iterable[str]) -> str:
    text = str(raw_text or "").strip()
    names = list(valid_names)
    if text in names:
        return text
    for name in names:
        if name in text:
            return name
    match = re.search(r"[\w.-]+\.jpg", text, flags=re.IGNORECASE)
    if match and match.group(0) in names:
        return match.group(0)
    raise ValueError(f"VLM returned an invalid candidate: {text!r}; valid candidates: {names}")


def query_vlm(
    client: Any,
    model_name: str,
    mosaic_path: Path,
    candidates: list[Candidate],
    task: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str, str]:
    mapping = "\n".join(
        f"- {position}: {candidate.filename}, value={candidate.value:g}"
        for position, candidate in zip(POSITIONS_EN, candidates)
    )
    if task == "focus":
        prompt = f"""
You are a professional microscope image sharpness evaluator. The images may be out of focus, and you need to select the sharpest subimage from the 3x3 mosaic.


3x3 mosaic position-to-filename mapping:
{mapping}

Strictly follow these output requirements:
1. Return only the full filename of the sharpest subimage, including the suffix, for example "candidate_01_value_3940.jpg".
2. Do not add any extra content: no explanation, no quotes, and no formatting.
3. If multiple subimages have similar sharpness, prefer the one closest to the center.
        """
    elif task == "brightness":
        prompt = f"""
You are a professional microscope image brightness evaluator. Select the subimage with the most appropriate brightness from the 3x3 mosaic.

3x3 mosaic position-to-filename mapping:
{mapping}

Strictly follow these output requirements:
1. Return only the full filename of the subimage with the most appropriate brightness, including the suffix, for example "candidate_01_value_100.jpg".
2. Do not add any extra content: no explanation, no quotes, and no formatting.
3. If multiple subimages have similarly appropriate brightness, prefer the one closest to the center.
        """
    else:
        raise ValueError(f"Unsupported VLM task: {task}")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.strip()},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(mosaic_path), "detail": "high"}},
                ],
            }
        ],
        temperature=float(temperature),
        max_tokens=int(max_tokens),
    )
    raw = response.choices[0].message.content.strip()
    return extract_candidate_name(raw, [item.filename for item in candidates]), raw


def build_candidates(center: float, step: float, offsets: list[int], lower: float, upper: float) -> list[float]:
    values: list[float] = []
    for offset in offsets:
        value = clamp(center + float(offset) * float(step), lower, upper)
        if value not in values:
            values.append(value)
    while len(values) < 9:
        values.append(values[-1])
    return values[:9]


def run_vlm_search(
    microscope: Any,
    *,
    client: Any,
    model_name: str,
    output_dir: Path,
    task: str,
    get_current: Callable[[], float],
    set_value: Callable[[float], None],
    lower: float,
    upper: float,
    initial_step: float,
    min_step: float,
    max_iterations: int,
    candidate_offsets: list[int],
    settle_seconds: float,
    preview_timeout_seconds: float,
    capture_source: str,
    mosaic_subimage_size_px: int,
    vlm_temperature: float,
    vlm_max_tokens: int,
) -> dict[str, Any]:
    center = clamp(get_current(), lower, upper)
    step = float(initial_step)
    iterations: list[dict[str, Any]] = []
    best_value = center

    for iteration in range(1, int(max_iterations) + 1):
        iteration_dir = output_dir / f"{task}_iter_{iteration:02d}"
        values = build_candidates(center, step, candidate_offsets, lower, upper)
        candidates: list[Candidate] = []
        say(f"[ACTION] {task} iteration {iteration}: center={center:g}, step={step:g}")

        for idx, value in enumerate(values):
            set_value(float(value))
            time.sleep(float(settle_seconds))
            frame = capture_frame(microscope, capture_source, preview_timeout_seconds)
            filename = f"candidate_{idx + 1:02d}_value_{value:g}.jpg"
            image_path = iteration_dir / filename
            save_preview_frame(frame, image_path)
            candidates.append(Candidate(filename=filename, value=float(value), image_path=image_path))

        mosaic_path = iteration_dir / "mosaic.jpg"
        create_mosaic(candidates, mosaic_path, mosaic_subimage_size_px)
        selected_name, raw_response = query_vlm(
            client,
            model_name,
            mosaic_path,
            candidates,
            task,
            vlm_temperature,
            vlm_max_tokens,
        )
        selected = next(item for item in candidates if item.filename == selected_name)
        best_value = float(selected.value)
        iterations.append(
            {
                "iteration": iteration,
                "center_before": center,
                "step": step,
                "selected_filename": selected.filename,
                "selected_value": best_value,
                "vlm_raw_response": raw_response,
                "mosaic_path": str(mosaic_path),
                "candidates": [
                    {"filename": item.filename, "value": item.value, "image_path": str(item.image_path)}
                    for item in candidates
                ],
            }
        )
        say(f"[INFO] {task} VLM selected {selected.filename}, value={best_value:g}")

        center_candidate = candidates[len(candidates) // 2]
        if selected.filename == center_candidate.filename:
            step /= 2.0
        else:
            center = best_value

        if step < float(min_step):
            break

    set_value(best_value)
    return {"task": task, "selected_value": best_value, "iterations": iterations}


def append_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trial", "task", "selected_value", "iterations", "started_at", "finished_at", "status", "error"]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_once(config: dict[str, Any]) -> Path:
    settings = load_runtime_settings()
    runtime_context = initialize_system_components(settings.model.Simulation_mode)
    microscope = runtime_context.env_olympus
    preview_manager = None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = slugify(settings.model.vlm_model_name)
    run_dir = Path(config["output_dir"]) / f"{timestamp}__vlm_focus_brightness__model-{model_slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    try:
        say("[ACTION] Initializing microscope from existing runtime configuration")
        setup_microscope(microscope, settings.startup)

        if config.get("initial_z_um") is not None:
            microscope.set_z_position(float(config["initial_z_um"]))
        if config.get("channel"):
            microscope.set_channel(str(config["channel"]))
        if config.get("exposure_ms") is not None:
            microscope.set_exposure(float(config["exposure_ms"]))
        if str(config.get("channel", "1-NONE")) == "1-NONE" and config.get("brightness") is not None:
            microscope.set_brightness(int(config["brightness"]))

        say("[ACTION] Starting live preview")
        microscope.start_preview()
        wait_for_preview_frame(microscope, float(config["preview_timeout_seconds"]))
        say("[INFO] Live preview is active")

        if config.get("show_preview_window"):
            from utils.preview_process import PreviewProcessManager

            try:
                preview_manager = PreviewProcessManager(
                    microscope.get_live_preview_image,
                    window_name=getattr(microscope, "preview_window_name", "micro live"),
                )
                preview_manager.start()
                say("[INFO] Local live preview window started")
            except Exception as exc:
                preview_manager = None
                say(f"[INFO] Local live preview window unavailable: {exc}")

        mode = str(config.get("mode", "both")).lower()
        tasks = []
        if mode in ("focus", "both") and config["focus"].get("enabled", True):
            tasks.append("focus")
        if mode in ("brightness", "both") and config["brightness_search"].get("enabled", True):
            tasks.append("brightness")

        for trial in range(1, int(config.get("trial_count", 1)) + 1):
            for task in tasks:
                started_at = datetime.now().isoformat(timespec="seconds")
                task_dir = run_dir / f"trial_{trial:02d}" / task
                try:
                    if task == "focus":
                        search_cfg = config["focus"]
                        result = run_vlm_search(
                            microscope,
                            client=runtime_context.vlm_client,
                            model_name=settings.model.vlm_model_name,
                            output_dir=task_dir,
                            task="focus",
                            get_current=lambda: float(microscope.get_z_position()),
                            set_value=lambda value: microscope.set_z_position(float(value)),
                            lower=float(getattr(microscope, "Min_Z_position", -1e9)),
                            upper=float(getattr(microscope, "Max_Z_position", 1e9)),
                            initial_step=float(search_cfg["initial_step_um"]),
                            min_step=float(search_cfg["min_step_um"]),
                            max_iterations=int(search_cfg["max_iterations"]),
                            candidate_offsets=list(search_cfg["candidate_offsets"]),
                            settle_seconds=float(config["settle_seconds"]),
                            preview_timeout_seconds=float(config["preview_timeout_seconds"]),
                            capture_source=str(config.get("capture_source", "snap")),
                            mosaic_subimage_size_px=int(config["mosaic_subimage_size_px"]),
                            vlm_temperature=float(config["vlm_temperature"]),
                            vlm_max_tokens=int(config["vlm_max_tokens"]),
                        )
                    else:
                        if microscope.get_channel() != "1-NONE":
                            raise RuntimeError("Brightness VLM search requires brightfield channel '1-NONE'.")
                        search_cfg = config["brightness_search"]
                        result = run_vlm_search(
                            microscope,
                            client=runtime_context.vlm_client,
                            model_name=settings.model.vlm_model_name,
                            output_dir=task_dir,
                            task="brightness",
                            get_current=lambda: float(microscope.get_brightness()),
                            set_value=lambda value: microscope.set_brightness(int(round(value))),
                            lower=float(getattr(microscope, "Min_brightness", 0)),
                            upper=float(getattr(microscope, "Max_brightness", 100)),
                            initial_step=float(search_cfg["initial_step"]),
                            min_step=float(search_cfg["min_step"]),
                            max_iterations=int(search_cfg["max_iterations"]),
                            candidate_offsets=list(search_cfg["candidate_offsets"]),
                            settle_seconds=float(config["settle_seconds"]),
                            preview_timeout_seconds=float(config["preview_timeout_seconds"]),
                            capture_source=str(config.get("capture_source", "snap")),
                            mosaic_subimage_size_px=int(config["mosaic_subimage_size_px"]),
                            vlm_temperature=float(config["vlm_temperature"]),
                            vlm_max_tokens=int(config["vlm_max_tokens"]),
                        )
                    result_path = task_dir / "result.json"
                    result_path.parent.mkdir(parents=True, exist_ok=True)
                    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    summary_rows.append(
                        {
                            "trial": trial,
                            "task": task,
                            "selected_value": result["selected_value"],
                            "iterations": len(result["iterations"]),
                            "started_at": started_at,
                            "finished_at": datetime.now().isoformat(timespec="seconds"),
                            "status": "success",
                            "error": "",
                        }
                    )
                except Exception as exc:
                    summary_rows.append(
                        {
                            "trial": trial,
                            "task": task,
                            "selected_value": "",
                            "iterations": "",
                            "started_at": started_at,
                            "finished_at": datetime.now().isoformat(timespec="seconds"),
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    raise
    finally:
        append_summary_csv(run_dir / "summary.csv", summary_rows)
        (run_dir / "summary.json").write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        if preview_manager is not None:
            try:
                preview_manager.stop()
            except Exception:
                pass
        release_resources(runtime_context)

    return run_dir


def main() -> None:
    if not RUN_CONFIG.get("enabled", True):
        raise SystemExit("RUN_CONFIG['enabled'] is False.")
    run_dir = run_once(RUN_CONFIG)
    say(f"[DONE] VLM focus/brightness benchmark saved to: {run_dir}")


if __name__ == "__main__":
    main()
