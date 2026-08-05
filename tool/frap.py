from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from core_tool.spatial_metadata import load_ome_spatial_metadata
from tool.base import BaseTool, tool_func


def _import_pyautogui():
    try:
        import pyautogui
    except Exception as exc:
        raise RuntimeError("pyautogui is required for FRAP GUI control.") from exc
    return pyautogui


def _import_pygetwindow():
    try:
        import pygetwindow
    except Exception as exc:
        raise RuntimeError("pygetwindow is required for FRAP GUI control.") from exc
    return pygetwindow


class Frap(BaseTool):
    """Minimal GUI-driven FRAP helper."""

    planning_hint = (
        "Use this tool for GUI-driven laser manipulation with a compact API: "
        "laser_on, laser_off, laser_position, cell_detection, and cell_contourextraction."
    )
    execution_hint = (
        "laser_position expects physical stage coordinates in microns and requires the laser to be off. "
        "cell_detection and cell_contourextraction use the latest image in the output directory."
    )

    def __init__(self, storage_manager=None, output_dir: str = "./output") -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir
        self._laser_enabled = False
        self._default_profile_filename = "frap_ui_profile.json"

    @tool_func
    def laser_on(self, power: float, duration: float) -> None:
        """
        Activate laser for precise cell manipulation.

        Args:
            power: Laser intensity (percentage of max power, 0.0-100.0)
            duration: Exposure time in milliseconds (ms)
        """
        if not (0.0 <= float(power) <= 100.0):
            raise ValueError("power must be between 0.0 and 100.0")
        if float(duration) <= 0:
            raise ValueError("duration must be positive")
        self._run_named_action("laser_on", fallback="enable_frap")
        self._laser_enabled = True

    @tool_func
    def laser_off(self) -> None:
        """Immediately deactivate laser emission."""
        self._run_named_action("laser_off", fallback="disable_frap")
        self._laser_enabled = False

    @tool_func
    def laser_position(self, x: int, y: int) -> None:
        """
        Set laser focal point coordinates.

        Args:
            x: X-axis position (microns)
            y: Y-axis position (microns)
        """
        if self._laser_enabled:
            raise RuntimeError("laser_position requires the laser to be OFF during positioning.")

        image_path = self._resolve_latest_image_path()
        metadata = self._load_spatial_metadata(image_path)
        centered_x_px, centered_y_px = self._stage_to_centered_pixels(float(x), float(y), image_path, metadata)
        profile, window_info = self._prepare_laser_runtime_context()
        target = self._resolve_laser_target(centered_x_px, centered_y_px, profile)
        self._activate_window_if_needed(profile)
        self._run_actions(profile.get("workflow", {}).get("pre_point_actions", []), profile, window_info)
        self._click_laser_target(
            profile=profile,
            window_info=window_info,
            roi_offset_x=target["roi_offset_x"],
            roi_offset_y=target["roi_offset_y"],
        )
        self._run_actions(profile.get("workflow", {}).get("post_point_actions", []), profile, window_info)

    @tool_func
    def cell_detection(self) -> dict:
        """
        Detect and return the position of target cell.

        Returns:
            Dictionary containing cell coordinates:
            - 'x': X-axis position (microns)
            - 'y': Y-axis position (microns)
        """
        contour = self.cell_contourextraction()
        centroid = contour.get("centroid")
        if not isinstance(centroid, dict):
            return {}
        return {"x": float(centroid["x"]), "y": float(centroid["y"])}

    @tool_func
    def cell_contourextraction(self) -> dict:
        """
        Extract cell membrane contour from current image.

        Returns:
            Dictionary containing contour data:
            - 'points': List of (x,y) tuples (microns)
            - 'area': Cell area in square microns (um^2)
            - 'perimeter': Cell perimeter in microns (um)
        """
        image_path = self._resolve_latest_image_path()
        image = self._prepare_analysis_image(self._read_image_array(str(image_path)))
        metadata = self._load_spatial_metadata(image_path)
        component_mask = self._segment_largest_component(image)
        if component_mask is None:
            return {}

        boundary_points = self._extract_boundary_points(component_mask)
        if not boundary_points:
            return {}

        image_height, image_width = component_mask.shape
        rows, cols = np.nonzero(component_mask)
        centroid_row = float(np.mean(rows))
        centroid_col = float(np.mean(cols))
        pixel_area_um2 = float(metadata["pixel_size_x_um"]) * float(metadata["pixel_size_y_um"])
        perimeter_um = float(len(boundary_points)) * (
            float(metadata["pixel_size_x_um"]) + float(metadata["pixel_size_y_um"])
        ) / 2.0

        points = [
            self._image_pixel_to_stage_point(float(col), float(row), image_width, image_height, metadata)
            for row, col in boundary_points
        ]
        centroid_x, centroid_y = self._image_pixel_to_stage_point(
            centroid_col,
            centroid_row,
            image_width,
            image_height,
            metadata,
        )
        return {
            "points": points,
            "area": float(np.count_nonzero(component_mask)) * pixel_area_um2,
            "perimeter": perimeter_um,
            "centroid": {"x": centroid_x, "y": centroid_y},
            "coordinate_system": "stage_microns",
            "source_image": str(image_path),
        }

    def _load_profile(self) -> dict:
        path = Path(self.output_dir, self._default_profile_filename).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Default FRAP UI profile not found: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        controls = payload.get("controls", {})
        workflow = payload.get("workflow", {})
        options = payload.get("options", {})
        image_region = payload.get("image_region", {})
        if not isinstance(controls, dict) or not isinstance(workflow, dict) or not isinstance(options, dict):
            raise ValueError("Invalid FRAP profile structure")
        return {
            "window_title_keyword": str(payload.get("window_title_keyword", "")).strip(),
            "image_region": {
                "left": int(image_region.get("left", 0)),
                "top": int(image_region.get("top", 0)),
                "width": int(image_region.get("width", 0)),
                "height": int(image_region.get("height", 0)),
            },
            "controls": controls,
            "workflow": {
                "pre_point_actions": list(workflow.get("pre_point_actions", [])),
                "post_point_actions": list(workflow.get("post_point_actions", [])),
            },
            "options": {
                "activate_before_action": bool(options.get("activate_before_action", True)),
                "click_interval_sec": float(options.get("click_interval_sec", 0.15)),
                "move_duration_sec": float(options.get("move_duration_sec", 0.0)),
                "flip_x": bool(options.get("flip_x", False)),
                "flip_y": bool(options.get("flip_y", False)),
            },
        }

    def _prepare_laser_runtime_context(self) -> tuple[dict, dict]:
        profile = self._load_profile()
        keyword = profile["window_title_keyword"]
        if not keyword:
            raise ValueError("FRAP profile window_title_keyword must not be empty")
        return profile, self._wait_for_window(keyword)

    def _run_named_action(self, preferred: str, *, fallback: str) -> None:
        profile = self._load_profile()
        controls = profile["controls"]
        action_name = preferred if preferred in controls else fallback
        if action_name not in controls:
            raise ValueError(f"FRAP profile does not define control '{preferred}' or '{fallback}'.")
        window_info = self._wait_for_window(profile["window_title_keyword"])
        self._activate_window_if_needed(profile)
        self._run_actions([action_name], profile, window_info)

    def _wait_for_window(self, title_keyword: str, timeout_sec: float = 15.0, poll_interval_sec: float = 0.2) -> dict:
        deadline = time.time() + float(timeout_sec)
        while True:
            windows = _import_pygetwindow().getWindowsWithTitle(str(title_keyword))
            visible = [
                window for window in windows
                if getattr(window, "width", 0) > 0 and getattr(window, "height", 0) > 0
            ]
            if visible:
                window = visible[0]
                return {
                    "title": str(getattr(window, "title", "")),
                    "left": int(getattr(window, "left", 0)),
                    "top": int(getattr(window, "top", 0)),
                    "width": int(getattr(window, "width", 0)),
                    "height": int(getattr(window, "height", 0)),
                }
            if time.time() >= deadline:
                raise RuntimeError(f"No visible window matched title keyword within {timeout_sec:.2f}s: {title_keyword}")
            time.sleep(float(poll_interval_sec))

    def _activate_window_if_needed(self, profile: dict) -> None:
        if not bool(profile["options"].get("activate_before_action", True)):
            return
        window_info = self._wait_for_window(profile["window_title_keyword"])
        matched = _import_pygetwindow().getWindowsWithTitle(window_info["title"])
        if not matched:
            raise RuntimeError(f"Window disappeared before activation: {window_info['title']}")
        window = matched[0]
        if hasattr(window, "isMinimized") and getattr(window, "isMinimized", False):
            try:
                window.restore()
            except Exception:
                pass
        window.activate()
        time.sleep(0.2)

    def _run_actions(self, action_items: list[Any], profile: dict, window_info: dict) -> None:
        for item in action_items:
            action = self._resolve_action(item, profile)
            action_type = str(action.get("type", "")).strip().lower()

            if action_type == "point":
                self._click_window_relative(
                    window_info,
                    offset_x=int(action["x"]),
                    offset_y=int(action["y"]),
                    move_duration_sec=float(profile["options"]["move_duration_sec"]),
                    button=str(action.get("button", "left")),
                    clicks=int(action.get("clicks", 1)),
                )
            elif action_type == "hotkey":
                self._press_hotkey(action.get("keys", []))
            elif action_type == "text":
                self._type_text(str(action.get("text", "")), float(action.get("interval_sec", 0.0)))
            elif action_type == "wait":
                time.sleep(max(float(action.get("seconds", 0.0)), 0.0))
            else:
                raise ValueError(f"Unsupported FRAP action type: {action_type}")

            if action_type == "point" and float(profile["options"]["click_interval_sec"]) > 0:
                time.sleep(float(profile["options"]["click_interval_sec"]))

    def _resolve_action(self, item: Any, profile: dict) -> dict:
        if isinstance(item, str):
            if item not in profile["controls"]:
                raise ValueError(f"FRAP profile control not found: {item}")
            return dict(profile["controls"][item])
        if isinstance(item, dict) and "control" in item:
            name = str(item.get("control", "")).strip()
            if name not in profile["controls"]:
                raise ValueError(f"FRAP profile control not found: {name}")
            return dict(profile["controls"][name])
        if not isinstance(item, dict):
            raise ValueError("FRAP workflow actions must be strings or action dictionaries")
        return item

    def _click_laser_target(self, profile: dict, window_info: dict, roi_offset_x: int, roi_offset_y: int) -> None:
        image_region = profile["image_region"]
        self._click_window_relative(
            window_info,
            offset_x=int(image_region["left"]) + int(roi_offset_x),
            offset_y=int(image_region["top"]) + int(roi_offset_y),
            move_duration_sec=float(profile["options"]["move_duration_sec"]),
        )

    def _click_window_relative(
        self,
        window_info: dict,
        *,
        offset_x: int,
        offset_y: int,
        move_duration_sec: float,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        window_left = int(window_info["left"])
        window_top = int(window_info["top"])
        window_width = int(window_info["width"])
        window_height = int(window_info["height"])
        if not (0 <= int(offset_x) < window_width and 0 <= int(offset_y) < window_height):
            raise ValueError(
                f"Relative click offset is outside the window bounds: offset=({offset_x}, {offset_y}) "
                f"window=({window_width}, {window_height})"
            )
        screen_x = window_left + int(offset_x)
        screen_y = window_top + int(offset_y)
        pyautogui = _import_pyautogui()
        original_pause = getattr(pyautogui, "PAUSE", 0.0)
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.moveTo(screen_x, screen_y, duration=max(float(move_duration_sec), 0.0))
            pyautogui.click(screen_x, screen_y, clicks=max(int(clicks), 1), button=str(button))
        finally:
            pyautogui.PAUSE = original_pause

    def _press_hotkey(self, keys: list[str] | tuple[str, ...] | str) -> None:
        if isinstance(keys, str):
            key_list = [keys]
        else:
            key_list = [str(item).strip() for item in keys if str(item).strip()]
        if not key_list:
            raise ValueError("At least one hotkey key is required")
        pyautogui = _import_pyautogui()
        original_pause = getattr(pyautogui, "PAUSE", 0.0)
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.hotkey(*key_list, interval=0.0)
        finally:
            pyautogui.PAUSE = original_pause

    def _type_text(self, text: str, interval_sec: float = 0.0) -> None:
        pyautogui = _import_pyautogui()
        original_pause = getattr(pyautogui, "PAUSE", 0.0)
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.write(str(text), interval=max(float(interval_sec), 0.0))
        finally:
            pyautogui.PAUSE = original_pause

    def _resolve_input_path(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(self.output_dir, candidate).expanduser()
        return candidate.resolve()

    def _resolve_latest_image_path(self) -> Path:
        candidates: list[Path] = []
        output_root = Path(self.output_dir).expanduser().resolve()

        if self.storage_manager is not None and hasattr(self.storage_manager, "read_log"):
            try:
                registered = self.storage_manager.read_log(True)
            except Exception:
                registered = {}
            if isinstance(registered, dict):
                for meta in registered.values():
                    if not isinstance(meta, dict):
                        continue
                    filename = str(meta.get("filename", "") or "").strip()
                    if not filename:
                        continue
                    path = Path(output_root, filename).resolve()
                    if path.is_file():
                        candidates.append(path)

        if output_root.exists():
            for path in output_root.rglob("*"):
                if path.is_file():
                    suffixes = "".join(path.suffixes).lower()
                    if suffixes.endswith((".ome.tif", ".ome.tiff", ".tif", ".tiff", ".png", ".jpg", ".jpeg")):
                        candidates.append(path.resolve())

        if not candidates:
            raise FileNotFoundError("No current image is available in the FRAP output directory.")
        return max(set(candidates), key=lambda item: item.stat().st_mtime)

    def _load_spatial_metadata(self, image_path: str | Path) -> dict[str, float | bool]:
        return load_ome_spatial_metadata(image_path, require_stage_positions=True)

    def _read_image_array(self, image_path: str) -> np.ndarray:
        resolved_path = self._resolve_input_path(image_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Image file not found: {resolved_path}")
        suffixes = "".join(resolved_path.suffixes).lower()
        if suffixes.endswith((".tif", ".tiff")):
            return np.asarray(tifffile.imread(resolved_path))
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(f"matplotlib is required to load '{resolved_path.name}'.") from exc
        return np.asarray(plt.imread(resolved_path))

    def _prepare_analysis_image(self, image_array: np.ndarray) -> np.ndarray:
        array = np.asarray(image_array)
        if array.ndim >= 4:
            array = array.reshape((-1,) + array.shape[-2:])[0]
        elif array.ndim == 3 and array.shape[-1] not in {3, 4}:
            array = array[0]
        if array.ndim == 3 and array.shape[-1] in {3, 4}:
            array = np.mean(array[..., :3], axis=2)
        if array.ndim != 2:
            raise ValueError(f"Unsupported image dimensions for analysis: {np.asarray(image_array).shape}")
        array = array.astype(np.float32)
        min_value = float(np.min(array))
        max_value = float(np.max(array))
        if max_value <= min_value:
            return np.zeros_like(array, dtype=np.uint8)
        return np.clip((array - min_value) / (max_value - min_value) * 255.0, 0, 255).astype(np.uint8)

    def _segment_largest_component(self, analysis_image: np.ndarray) -> np.ndarray | None:
        image = np.asarray(analysis_image, dtype=np.uint8)
        threshold_low = int(np.percentile(image, 40))
        threshold_high = int(np.percentile(image, 60))
        candidates = [
            self._largest_connected_component(image <= threshold_low),
            self._largest_connected_component(image >= threshold_high),
        ]
        candidates = [mask for mask in candidates if mask is not None]
        if not candidates:
            return None
        return max(candidates, key=lambda mask: int(np.count_nonzero(mask)))

    def _largest_connected_component(self, binary_mask: np.ndarray) -> np.ndarray | None:
        mask = np.asarray(binary_mask, dtype=bool)
        if mask.ndim != 2 or not np.any(mask):
            return None

        height, width = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        best_points: list[tuple[int, int]] = []
        best_interior_points: list[tuple[int, int]] = []

        for row in range(height):
            for col in range(width):
                if not mask[row, col] or visited[row, col]:
                    continue

                queue: deque[tuple[int, int]] = deque([(row, col)])
                visited[row, col] = True
                points: list[tuple[int, int]] = []

                while queue:
                    current_row, current_col = queue.popleft()
                    points.append((current_row, current_col))
                    for next_row, next_col in (
                        (current_row - 1, current_col),
                        (current_row + 1, current_col),
                        (current_row, current_col - 1),
                        (current_row, current_col + 1),
                    ):
                        if (
                            0 <= next_row < height
                            and 0 <= next_col < width
                            and mask[next_row, next_col]
                            and not visited[next_row, next_col]
                        ):
                            visited[next_row, next_col] = True
                            queue.append((next_row, next_col))

                touches_border = any(
                    point_row in {0, height - 1} or point_col in {0, width - 1}
                    for point_row, point_col in points
                )
                if not touches_border and len(points) > len(best_interior_points):
                    best_interior_points = points
                if len(points) > len(best_points):
                    best_points = points

        selected = best_interior_points or best_points
        if not selected:
            return None
        component_mask = np.zeros_like(mask, dtype=bool)
        for row, col in selected:
            component_mask[row, col] = True
        return component_mask

    def _extract_boundary_points(self, component_mask: np.ndarray) -> list[tuple[int, int]]:
        mask = np.asarray(component_mask, dtype=bool)
        height, width = mask.shape
        boundary: list[tuple[int, int]] = []
        for row in range(height):
            for col in range(width):
                if not mask[row, col]:
                    continue
                neighbors = (
                    row == 0
                    or row == height - 1
                    or col == 0
                    or col == width - 1
                    or not mask[row - 1, col]
                    or not mask[row + 1, col]
                    or not mask[row, col - 1]
                    or not mask[row, col + 1]
                )
                if neighbors:
                    boundary.append((row, col))
        return boundary

    def _image_pixel_to_centered_point(
        self,
        *,
        x_px: float,
        y_px: float,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float]:
        center_x = (float(image_width) - 1.0) / 2.0
        center_y = (float(image_height) - 1.0) / 2.0
        return float(x_px) - center_x, float(y_px) - center_y

    def _centered_to_image_pixel(
        self,
        *,
        x_px: float,
        y_px: float,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int]:
        center_x = (float(image_width) - 1.0) / 2.0
        center_y = (float(image_height) - 1.0) / 2.0
        absolute_x = center_x + float(x_px)
        absolute_y = center_y + float(y_px)
        if not (0.0 <= absolute_x <= float(image_width - 1)):
            raise ValueError(f"x_px is outside image bounds: {x_px}")
        if not (0.0 <= absolute_y <= float(image_height - 1)):
            raise ValueError(f"y_px is outside image bounds: {y_px}")
        return int(round(absolute_x)), int(round(absolute_y))

    def _image_pixel_to_stage_point(
        self,
        x_px: float,
        y_px: float,
        image_width: int,
        image_height: int,
        metadata: dict[str, float | bool],
    ) -> tuple[float, float]:
        center_x_px = (float(image_width) - 1.0) / 2.0
        center_y_px = (float(image_height) - 1.0) / 2.0
        return (
            float(metadata["center_x_um"]) + (float(x_px) - center_x_px) * float(metadata["pixel_size_x_um"]),
            float(metadata["center_y_um"]) + (float(y_px) - center_y_px) * float(metadata["pixel_size_y_um"]),
        )

    def _stage_to_centered_pixels(
        self,
        x_um: float,
        y_um: float,
        image_path: str | Path,
        metadata: dict[str, float | bool] | None = None,
    ) -> tuple[float, float]:
        spatial = metadata or self._load_spatial_metadata(image_path)
        image = self._prepare_analysis_image(self._read_image_array(str(image_path)))
        image_height, image_width = image.shape
        image_center_x_px = (float(image_width) - 1.0) / 2.0
        image_center_y_px = (float(image_height) - 1.0) / 2.0
        image_x_px = image_center_x_px + (float(x_um) - float(spatial["center_x_um"])) / float(spatial["pixel_size_x_um"])
        image_y_px = image_center_y_px + (float(y_um) - float(spatial["center_y_um"])) / float(spatial["pixel_size_y_um"])
        return self._image_pixel_to_centered_point(
            x_px=image_x_px,
            y_px=image_y_px,
            image_width=image_width,
            image_height=image_height,
        )

    def _resolve_laser_target(self, x_px: float, y_px: float, profile: dict) -> dict[str, int]:
        image_region = profile["image_region"]
        options = profile["options"]
        image_x_px, image_y_px = self._centered_to_image_pixel(
            x_px=float(x_px),
            y_px=float(y_px),
            image_width=int(image_region["width"]),
            image_height=int(image_region["height"]),
        )
        roi_offset_x = self._map_image_axis_to_region_axis(
            image_x_px, int(image_region["width"]), bool(options.get("flip_x", False))
        )
        roi_offset_y = self._map_image_axis_to_region_axis(
            image_y_px, int(image_region["height"]), bool(options.get("flip_y", False))
        )
        return {"roi_offset_x": roi_offset_x, "roi_offset_y": roi_offset_y}

    def _map_image_axis_to_region_axis(self, pixel_value: int, axis_extent: int, flip_axis: bool) -> int:
        clamped = max(0, min(int(pixel_value), int(axis_extent - 1)))
        return int(axis_extent - 1 - clamped) if flip_axis else clamped
