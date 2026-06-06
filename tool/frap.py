from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from tool.base import BaseTool, tool_func


def _import_pyautogui():
    try:
        import pyautogui
    except Exception as exc:
        raise RuntimeError(
            "pyautogui is required for FRAP click execution, but it is unavailable."
        ) from exc
    return pyautogui


def _import_pygetwindow():
    try:
        import pygetwindow
    except Exception as exc:
        raise RuntimeError(
            "pygetwindow is required for FRAP window inspection, but it is unavailable."
        ) from exc
    return pygetwindow


class Frap(BaseTool):
    """
    High-level FRAP helper for image-guided ROI panel control.

    The public interface is intentionally small so code-generation models can
    call a few stable methods:
    - enable_frap()
    - disable_frap()
    - laser_position(x_px, y_px)
    - cell_contourextraction(image_path)

    The laser position uses image-centered pixel coordinates. The tool handles
    the internal mapping from the image coordinate system to the FRAP software
    ROI panel described by a saved GUI profile.
    """

    planning_hint = (
        "Use this tool for a compact FRAP workflow API built from atomic "
        "capabilities: enable_frap, disable_frap, laser_position, and "
        "cell_contourextraction. "
        "Complex behaviors are not always native one-shot operations. "
        "When no single API exactly matches the requested outcome, first consider "
        "whether the task can be achieved by composing these atomic capabilities "
        "into a valid short plan, potentially using standard Python computation "
        "and generated point sequences."
    )

    execution_hint = (
        "Treat the public API methods as atomic capabilities. Call "
        "enable_frap() before laser_position(). laser_position(x_px, y_px) "
        "performs a single-point FRAP action in image-centered pixel "
        "coordinates. For trajectories, geometric patterns, or other complex "
        "behaviors, use standard Python computation, control flow, and "
        "intermediate point-sequence generation, then call laser_position() "
        "repeatedly for each point. Use cell_contourextraction(image_path) when "
        "contour- or centroid-based target selection is needed from an input "
        "OME-TIFF image. Call disable_frap() when the sequence finishes."
    )

    def __init__(self, storage_manager=None, output_dir: str = "./output") -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir
        self._frap_enabled = False
        self._default_profile_filename = "frap_ui_profile.json"

    def create_ui_profile(
        self,
        profile_name: str,
        window_title_keyword: str,
        image_left: int,
        image_top: int,
        image_width: int,
        image_height: int,
        controls: dict | None = None,
        workflow: dict | None = None,
        activate_before_action: bool = True,
        click_interval_sec: float = 0.15,
        move_duration_sec: float = 0.0,
        flip_x: bool = False,
        flip_y: bool = False,
    ) -> dict:
        """
        Create a GUI profile for a FRAP-capable desktop application.

        Args:
            profile_name: Human-readable profile label.
            window_title_keyword: Partial window title used to attach to the application.
            image_left: Left offset of the image region relative to the window.
            image_top: Top offset of the image region relative to the window.
            image_width: Width of the image region relative to the window.
            image_height: Height of the image region relative to the window.
            controls: Optional mapping of named GUI actions.
            workflow: Optional workflow configuration with pre/post action lists.
            activate_before_action: Whether to activate the window before a batch of actions.
            click_interval_sec: Default delay between click actions.
            move_duration_sec: Default mouse move duration.
            flip_x: Whether to mirror the x axis when mapping image points into the image region.
            flip_y: Whether to mirror the y axis when mapping image points into the image region.

        Returns:
            A normalized FRAP GUI profile dictionary.
        """
        profile = {
            "profile_kind": "frap_ui_profile",
            "profile_name": str(profile_name),
            "window_title_keyword": str(window_title_keyword),
            "image_region": {
                "left": int(image_left),
                "top": int(image_top),
                "width": int(image_width),
                "height": int(image_height),
            },
            "controls": controls or {},
            "workflow": workflow or {},
            "options": {
                "activate_before_action": bool(activate_before_action),
                "click_interval_sec": float(click_interval_sec),
                "move_duration_sec": float(move_duration_sec),
                "flip_x": bool(flip_x),
                "flip_y": bool(flip_y),
            },
        }
        return self._normalize_ui_profile(profile)

    def save_ui_profile(self, ui_profile: dict, filename: str = "frap_ui_profile.json") -> str:
        """
        Save a FRAP GUI profile as JSON.

        Args:
            ui_profile: GUI profile dictionary.
            filename: Output JSON filename.

        Returns:
            Absolute path to the saved profile.
        """
        payload = self._normalize_ui_profile(ui_profile)
        return self._save_json_document(payload, filename, "FRAP UI profile")

    def load_ui_profile(self, filename: str) -> dict:
        """
        Load a saved FRAP GUI profile.

        Args:
            filename: Relative or absolute path to a GUI profile JSON file.

        Returns:
            Parsed and normalized GUI profile dictionary.
        """
        payload = self._load_json_document(filename, expected_kind="frap_ui_profile")
        return self._normalize_ui_profile(payload)

    def wait_for_window(
        self,
        title_keyword: str,
        timeout_sec: float = 15.0,
        poll_interval_sec: float = 0.2,
    ) -> dict:
        """
        Wait until a visible desktop window matching the title keyword appears.

        Args:
            title_keyword: Partial window title used to locate the application window.
            timeout_sec: Maximum time to wait.
            poll_interval_sec: Delay between polling attempts.

        Returns:
            A dictionary containing the matched window title and bounds.
        """
        if timeout_sec < 0:
            raise ValueError("timeout_sec must be non-negative")
        if poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")

        deadline = time.time() + float(timeout_sec)
        while True:
            try:
                return self.inspect_window(title_keyword)
            except RuntimeError:
                if time.time() >= deadline:
                    raise RuntimeError(
                        f"No visible window matched title keyword within {timeout_sec:.2f}s: {title_keyword}"
                    ) from None
                time.sleep(float(poll_interval_sec))

    def activate_window(self, title_keyword: str) -> dict:
        """
        Activate a desktop window by title keyword.

        Args:
            title_keyword: Partial window title used to locate the application window.
        Returns:
            A dictionary describing the executed activation.
        """
        window_info = self.inspect_window(title_keyword)
        result = {
            "title_keyword": str(title_keyword),
            "title": window_info["title"],
            "left": int(window_info["left"]),
            "top": int(window_info["top"]),
            "width": int(window_info["width"]),
            "height": int(window_info["height"]),
        }

        pygetwindow = _import_pygetwindow()
        matched_windows = pygetwindow.getWindowsWithTitle(window_info["title"])
        if not matched_windows:
            raise RuntimeError(f"Window disappeared before activation: {window_info['title']}")

        window = matched_windows[0]
        if hasattr(window, "isMinimized") and getattr(window, "isMinimized", False):
            try:
                window.restore()
            except Exception:
                pass
        window.activate()
        time.sleep(0.2)
        result["status"] = "activated"
        return result

    def press_hotkey(
        self,
        keys: list[str] | tuple[str, ...] | str,
        interval_sec: float = 0.0,
    ) -> dict:
        """
        Press one or more keyboard shortcuts.

        Args:
            keys: A string key name or a list/tuple of keys passed to pyautogui.hotkey.
            interval_sec: Delay between keys in the hotkey sequence.
        Returns:
            A dictionary describing the executed hotkey.
        """
        key_list = self._normalize_hotkey_keys(keys)
        if interval_sec < 0:
            raise ValueError("interval_sec must be non-negative")

        result = {
            "keys": key_list,
            "interval_sec": float(interval_sec),
        }

        pyautogui = _import_pyautogui()
        original_pause = getattr(pyautogui, "PAUSE", 0.0)
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.hotkey(*key_list, interval=max(float(interval_sec), 0.0))
        finally:
            pyautogui.PAUSE = original_pause

        result["status"] = "pressed"
        return result

    def type_text(self, text: str, interval_sec: float = 0.0) -> dict:
        """
        Type text into the currently focused application.

        Args:
            text: Text to type.
            interval_sec: Delay between characters.
        Returns:
            A dictionary describing the executed text entry.
        """
        if interval_sec < 0:
            raise ValueError("interval_sec must be non-negative")

        result = {
            "text": str(text),
            "interval_sec": float(interval_sec),
        }

        pyautogui = _import_pyautogui()
        original_pause = getattr(pyautogui, "PAUSE", 0.0)
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.write(str(text), interval=max(float(interval_sec), 0.0))
        finally:
            pyautogui.PAUSE = original_pause

        result["status"] = "typed"
        return result

    def run_ui_actions(
        self,
        profile_path: str,
        action_names: list[str] | tuple[str, ...],
    ) -> dict:
        """
        Run a named batch of GUI actions described by a saved UI profile.

        Args:
            profile_path: Relative or absolute path to a saved UI profile JSON file.
            action_names: Control names to execute in order.
        Returns:
            A dictionary summarizing the executed action batch.
        """
        if not isinstance(action_names, (list, tuple)) or not action_names:
            raise ValueError("action_names must be a non-empty list or tuple of control names")

        profile = self.load_ui_profile(profile_path)
        window_info = self.wait_for_window(profile["window_title_keyword"])
        actions: list[dict[str, Any]] = []
        if profile["options"]["activate_before_action"]:
            actions.append(self.activate_window(profile["window_title_keyword"]))

        actions.extend(
            self._execute_action_sequence(
                list(action_names),
                profile,
                window_info,
            )
        )
        return {
            "status": "ok",
            "profile_name": profile["profile_name"],
            "window": window_info,
            "actions": actions,
        }

    @tool_func
    def enable_frap(self) -> dict:
        """
        Open or switch the FRAP software into FRAP mode for the current session.

        This method is intended to operate the FRAP software UI, not merely to
        set an internal flag. When a saved default FRAP UI profile contains an
        ``enable_frap`` control, that GUI action is executed as part of this step.

        Args:
        Returns:
            A dictionary describing the resulting FRAP software mode state.
        """
        action_result = self._run_named_profile_action("enable_frap")
        self._frap_enabled = True

        return {
            "status": "ok",
            "frap_enabled": self._frap_enabled,
            "mode_action": action_result["mode_action"],
            "action": action_result["action"],
        }

    @tool_func
    def disable_frap(self) -> dict:
        """
        Close or exit FRAP mode in the FRAP software for the current session.

        This method is intended to operate the FRAP software UI, not merely to
        clear an internal flag. When a saved default FRAP UI profile contains a
        ``disable_frap`` control, that GUI action is executed as part of this step.

        Args:
        Returns:
            A dictionary describing the resulting FRAP software mode state.
        """
        action_result = self._run_named_profile_action("disable_frap")
        self._frap_enabled = False

        return {
            "status": "ok",
            "frap_enabled": self._frap_enabled,
            "mode_action": action_result["mode_action"],
            "action": action_result["action"],
        }

    @tool_func
    def laser_position(self, x_px: float, y_px: float) -> dict:
        """
        Move to an image-relative target position and trigger the FRAP click.

        Args:
            x_px: Horizontal pixel offset relative to the image center, where
                the image center is always ``(0, 0)``. Positive x moves right
                from the center. 
            y_px: Vertical pixel offset relative to the image center, where
                the image center is always ``(0, 0)``. Positive y moves down
                from the center. 
        Returns:
            A dictionary containing:
            - ``status``: String status, currently ``"ok"`` on success.
        """
        self._require_frap_enabled()
        profile, window_info = self._prepare_laser_runtime_context()
        target_info = self._resolve_laser_target(
            x_px=float(x_px),
            y_px=float(y_px),
            profile=profile,
        )
        activation_result = self._activate_profile_window_if_needed(profile)
        pre_actions = self._execute_profile_stage_actions(
            profile["workflow"]["pre_point_actions"],
            profile,
            window_info,
        )
        click_result = self._click_laser_target(
            profile=profile,
            window_info=window_info,
            roi_offset_x=target_info["roi_offset_x"],
            roi_offset_y=target_info["roi_offset_y"],
        )
        post_actions = self._execute_profile_stage_actions(
            profile["workflow"]["post_point_actions"],
            profile,
            window_info,
        )

        return {
            "status": "ok",
            "frap_enabled": self._frap_enabled,
            "coordinate_system": "image_centered_pixels",
            "input_target_px": {
                "x_px": float(x_px),
                "y_px": float(y_px),
            },
            "image_pixel_target": {
                "x_px": float(target_info["image_x_px"]),
                "y_px": float(target_info["image_y_px"]),
            },
            "roi_panel_offset_px": {
                "x_px": int(target_info["roi_offset_x"]),
                "y_px": int(target_info["roi_offset_y"]),
            },
            "window": window_info,
            "activation": activation_result,
            "pre_actions": pre_actions,
            "click": click_result,
            "post_actions": post_actions,
        }

    @tool_func
    def cell_contourextraction(self, image_path: str) -> dict:
        """
        Extract the dominant cell contour from an input image.

        Args:
            image_path: Relative or absolute path to the input image.

        Returns:
            Dictionary containing contour data in image-centered pixel coordinates:
            - ``points``: List of ``(x_px, y_px)`` tuples
            - ``area``: Cell area in square pixels
            - ``perimeter``: Cell perimeter in pixels
            - ``centroid_px``: Contour centroid in image-centered pixels

            Returns an empty dictionary if no cell-like component is detected.
        """
        image_array = self._read_image_array(image_path)
        analysis_image = self._prepare_analysis_image(image_array)
        component_mask = self._segment_largest_component(analysis_image)
        if component_mask is None:
            return {}

        boundary_points = self._extract_boundary_points(component_mask)
        if not boundary_points:
            return {}

        image_height, image_width = component_mask.shape
        component_rows, component_cols = np.nonzero(component_mask)
        centroid_row = float(np.mean(component_rows))
        centroid_col = float(np.mean(component_cols))

        contour_points = [
            self._image_pixel_to_centered_point(
                x_px=float(col),
                y_px=float(row),
                image_width=image_width,
                image_height=image_height,
            )
            for row, col in boundary_points
        ]
        centroid_px = self._image_pixel_to_centered_point(
            x_px=centroid_col,
            y_px=centroid_row,
            image_width=image_width,
            image_height=image_height,
        )

        return {
            "points": contour_points,
            "area": float(np.count_nonzero(component_mask)),
            "perimeter": float(len(boundary_points)),
            "centroid_px": {
                "x_px": float(centroid_px[0]),
                "y_px": float(centroid_px[1]),
            },
            "coordinate_system": "image_centered_pixels",
            "image_width": int(image_width),
            "image_height": int(image_height),
        }

    def inspect_window(self, title_keyword: str) -> dict:
        """
        Inspect a desktop window by title keyword.

        Args:
            title_keyword: Partial window title used to locate the FRAP software window.

        Returns:
            A dictionary containing the matched window title and bounds.
        """
        if not str(title_keyword).strip():
            raise ValueError("title_keyword must not be empty")

        pygetwindow = _import_pygetwindow()
        windows = pygetwindow.getWindowsWithTitle(str(title_keyword))
        visible_windows = [
            window
            for window in windows
            if getattr(window, "width", 0) > 0 and getattr(window, "height", 0) > 0
        ]
        if not visible_windows:
            raise RuntimeError(f"No visible window matched title keyword: {title_keyword}")

        window = visible_windows[0]
        return {
            "title": str(getattr(window, "title", "")),
            "left": int(getattr(window, "left", 0)),
            "top": int(getattr(window, "top", 0)),
            "width": int(getattr(window, "width", 0)),
            "height": int(getattr(window, "height", 0)),
        }

    def click_screen_point(
        self,
        screen_x: int,
        screen_y: int,
        move_duration_sec: float = 0.0,
    ) -> dict:
        """
        Click one point on the screen.

        Args:
            screen_x: Horizontal screen coordinate.
            screen_y: Vertical screen coordinate.
            move_duration_sec: Mouse move duration before clicking.
        Returns:
            A dictionary describing the executed click.
        """
        result = {
            "screen_x": int(screen_x),
            "screen_y": int(screen_y),
            "move_duration_sec": float(move_duration_sec),
        }

        pyautogui = _import_pyautogui()
        original_pause = getattr(pyautogui, "PAUSE", 0.0)
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.moveTo(int(screen_x), int(screen_y), duration=max(float(move_duration_sec), 0.0))
            pyautogui.click(int(screen_x), int(screen_y))
        finally:
            pyautogui.PAUSE = original_pause

        result["status"] = "clicked"
        return result

    def _resolve_input_path(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path).expanduser()
        registered_path = self._resolve_registered_input_path(candidate)
        if registered_path is not None:
            return registered_path
        if not candidate.is_absolute():
            candidate = Path(self.output_dir, candidate).expanduser()
        return candidate.resolve()

    def _resolve_output_path(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(self.output_dir, candidate).expanduser()
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate.resolve()

    def _resolve_registered_input_path(self, candidate: Path) -> Path | None:
        if self.storage_manager is None or not hasattr(self.storage_manager, "read_log"):
            return None

        try:
            registered = self.storage_manager.read_log(True)
        except Exception:
            return None
        if not isinstance(registered, dict):
            return None

        candidate_text = str(candidate).strip()
        candidate_name = candidate.name.strip()
        if not candidate_text and not candidate_name:
            return None

        lookup_keys = [key for key in {candidate_text, candidate_name} if key]
        matched_meta: dict[str, Any] | None = None
        for key in lookup_keys:
            meta = registered.get(key)
            if isinstance(meta, dict):
                matched_meta = meta
                break

        if matched_meta is None:
            for meta in registered.values():
                if not isinstance(meta, dict):
                    continue
                filename = str(meta.get("filename", "") or "").strip()
                if filename and filename in lookup_keys:
                    matched_meta = meta
                    break

        if matched_meta is None:
            return None

        registered_name = str(matched_meta.get("filename", "") or "").strip()
        if not registered_name:
            return None

        resolved = Path(self.output_dir, registered_name).expanduser().resolve()
        return resolved

    def _save_json_document(self, payload: dict, filename: str, description: str) -> str:
        output_path = self._resolve_output_path(filename)
        output_path.write_text(
            json.dumps(self._to_json_ready(payload), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._register_output(output_path.name, description, "json")
        return str(output_path)

    def _load_json_document(self, filename: str, expected_kind: str) -> dict:
        resolved_path = self._resolve_input_path(filename)
        if not resolved_path.exists():
            raise FileNotFoundError(f"JSON file not found: {resolved_path}")
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        payload_kind = str(
            payload.get("plan_kind") or payload.get("mapping_kind") or payload.get("profile_kind") or ""
        )
        if payload_kind != expected_kind:
            raise ValueError(
                f"Expected '{expected_kind}' document, but '{resolved_path.name}' contains '{payload_kind or 'unknown'}'."
            )
        return payload

    def _register_output(self, filename: str, description: str, file_type: str) -> None:
        if self.storage_manager is None or not hasattr(self.storage_manager, "register_file"):
            return
        self.storage_manager.register_file(filename, description, "frap", file_type)

    def _read_image_array(self, image_path: str) -> np.ndarray:
        resolved_path = self._resolve_input_path(image_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Image file not found: {resolved_path}")

        suffixes = "".join(resolved_path.suffixes).lower()
        if suffixes.endswith(".tif") or suffixes.endswith(".tiff"):
            return np.asarray(tifffile.imread(resolved_path))

        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                f"Unable to load non-TIFF image '{resolved_path.name}' because matplotlib is unavailable."
            ) from exc
        return np.asarray(plt.imread(resolved_path))

    def _prepare_display_image(self, image_array: np.ndarray) -> np.ndarray:
        array = np.asarray(image_array)
        if array.ndim == 2:
            return self._normalize_image(array)
        if array.ndim == 3 and array.shape[-1] in {3, 4}:
            return self._normalize_image(array[..., :3])
        if array.ndim == 3:
            return self._normalize_image(array[0])
        if array.ndim == 4 and array.shape[-1] in {3, 4}:
            return self._normalize_image(array[0, ..., :3])
        if array.ndim >= 4:
            flattened = array.reshape((-1,) + array.shape[-2:])
            return self._normalize_image(flattened[0])
        raise ValueError(f"Unsupported image dimensions for preview: {array.ndim}")

    def _normalize_image(self, image_array: np.ndarray) -> np.ndarray:
        array = np.asarray(image_array)
        if array.dtype == np.uint8:
            return array
        array = array.astype(np.float32)
        min_value = float(np.min(array))
        max_value = float(np.max(array))
        if max_value <= min_value:
            return np.zeros_like(array, dtype=np.uint8)
        normalized = (array - min_value) / (max_value - min_value)
        return np.clip(normalized * 255.0, 0, 255).astype(np.uint8)

    def _prepare_analysis_image(self, image_array: np.ndarray) -> np.ndarray:
        display_image = self._prepare_display_image(image_array)
        if display_image.ndim == 2:
            return display_image
        return np.mean(display_image[..., :3], axis=2).astype(np.uint8)

    def _segment_largest_component(self, analysis_image: np.ndarray) -> np.ndarray | None:
        image = np.asarray(analysis_image, dtype=np.uint8)
        if image.ndim != 2:
            raise ValueError("analysis_image must be a 2D grayscale image")

        threshold = int(np.percentile(image, 40))
        dark_mask = image <= threshold
        bright_mask = image >= int(np.percentile(image, 60))

        candidates = [
            self._largest_connected_component(dark_mask),
            self._largest_connected_component(bright_mask),
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

                if len(points) > len(best_points):
                    best_points = points

        if not best_points:
            return None

        component_mask = np.zeros_like(mask, dtype=bool)
        for row, col in best_points:
            component_mask[row, col] = True
        return component_mask

    def _extract_boundary_points(self, component_mask: np.ndarray) -> list[tuple[int, int]]:
        mask = np.asarray(component_mask, dtype=bool)
        if mask.ndim != 2:
            raise ValueError("component_mask must be a 2D mask")

        height, width = mask.shape
        boundary: list[tuple[int, int]] = []
        for row in range(height):
            for col in range(width):
                if not mask[row, col]:
                    continue
                if (
                    row == 0
                    or col == 0
                    or row == height - 1
                    or col == width - 1
                    or not mask[row - 1, col]
                    or not mask[row + 1, col]
                    or not mask[row, col - 1]
                    or not mask[row, col + 1]
                ):
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
            raise ValueError(
                f"x_px is outside the image bounds for centered coordinates: {x_px} (width={image_width})"
            )
        if not (0.0 <= absolute_y <= float(image_height - 1)):
            raise ValueError(
                f"y_px is outside the image bounds for centered coordinates: {y_px} (height={image_height})"
            )
        return int(round(absolute_x)), int(round(absolute_y))

    def _map_image_axis_to_region_axis(self, *, pixel_value: int, axis_extent: int, flip_axis: bool) -> int:
        clamped = max(0, min(int(pixel_value), int(axis_extent - 1)))
        return int(axis_extent - 1 - clamped) if flip_axis else clamped

    def _require_frap_enabled(self) -> None:
        if not self._frap_enabled:
            raise RuntimeError("FRAP is not enabled. Call enable_frap() before laser_position().")

    def _prepare_laser_runtime_context(self) -> tuple[dict, dict]:
        profile = self._load_default_ui_profile(required=True)
        window_info = self.wait_for_window(profile["window_title_keyword"])
        return profile, window_info

    def _resolve_laser_target(self, *, x_px: float, y_px: float, profile: dict) -> dict[str, int]:
        image_region = profile["image_region"]
        options = profile["options"]
        region_width = int(image_region["width"])
        region_height = int(image_region["height"])

        image_x_px, image_y_px = self._centered_to_image_pixel(
            x_px=float(x_px),
            y_px=float(y_px),
            image_width=region_width,
            image_height=region_height,
        )
        roi_offset_x = self._map_image_axis_to_region_axis(
            pixel_value=image_x_px,
            axis_extent=region_width,
            flip_axis=bool(options.get("flip_x", False)),
        )
        roi_offset_y = self._map_image_axis_to_region_axis(
            pixel_value=image_y_px,
            axis_extent=region_height,
            flip_axis=bool(options.get("flip_y", False)),
        )
        return {
            "image_x_px": int(image_x_px),
            "image_y_px": int(image_y_px),
            "roi_offset_x": int(roi_offset_x),
            "roi_offset_y": int(roi_offset_y),
        }

    def _activate_profile_window_if_needed(self, profile: dict) -> dict | None:
        if not bool(profile["options"].get("activate_before_action", True)):
            return None
        return self.activate_window(profile["window_title_keyword"])

    def _execute_profile_stage_actions(
        self,
        action_names: list[Any],
        profile: dict,
        window_info: dict,
    ) -> list[dict[str, Any]]:
        return self._execute_action_sequence(
            action_names,
            profile,
            window_info,
        )

    def _click_laser_target(
        self,
        *,
        profile: dict,
        window_info: dict,
        roi_offset_x: int,
        roi_offset_y: int,
    ) -> dict:
        image_region = profile["image_region"]
        options = profile["options"]
        return self._click_window_relative_from_window(
            window_info,
            offset_x=int(image_region["left"]) + int(roi_offset_x),
            offset_y=int(image_region["top"]) + int(roi_offset_y),
            move_duration_sec=float(options["move_duration_sec"]),
            button="left",
            clicks=1,
        )

    def _load_default_ui_profile(self, *, required: bool) -> dict | None:
        try:
            return self.load_ui_profile(self._default_profile_filename)
        except FileNotFoundError:
            if required:
                raise FileNotFoundError(
                    "Default FRAP UI profile not found. Expected "
                    f"'{self._default_profile_filename}' in the FRAP output directory."
                ) from None
            return None

    def _run_named_profile_action(self, action_name: str) -> dict:
        profile = self._load_default_ui_profile(required=False)
        if profile is None:
            raise FileNotFoundError(
                "Default FRAP UI profile not found. "
                f"Expected '{self._default_profile_filename}' in the FRAP output directory."
            )

        controls = profile.get("controls", {})
        if action_name not in controls:
            raise ValueError(f"Default FRAP UI profile does not define control: {action_name}")

        return {
            "mode_action": "profile_action",
            "action": self.run_ui_actions(
                self._default_profile_filename,
                [action_name],
            ),
        }

    def _to_json_ready(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._to_json_ready(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_json_ready(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _normalize_ui_profile(self, ui_profile: dict) -> dict:
        if not isinstance(ui_profile, dict):
            raise ValueError("ui_profile must be a dictionary")

        profile_name = str(ui_profile.get("profile_name", "") or "").strip()
        if not profile_name:
            raise ValueError("ui_profile.profile_name must not be empty")

        window_title_keyword = str(ui_profile.get("window_title_keyword", "") or "").strip()
        if not window_title_keyword:
            raise ValueError("ui_profile.window_title_keyword must not be empty")

        image_region = ui_profile.get("image_region")
        if not isinstance(image_region, dict):
            raise ValueError("ui_profile.image_region must be a dictionary")
        image_left = int(image_region.get("left", 0))
        image_top = int(image_region.get("top", 0))
        image_width = int(image_region.get("width", 0))
        image_height = int(image_region.get("height", 0))
        if image_left < 0 or image_top < 0:
            raise ValueError("ui_profile.image_region left/top must be non-negative")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("ui_profile.image_region width/height must be positive")

        raw_controls = ui_profile.get("controls", {})
        if raw_controls is None:
            raw_controls = {}
        if not isinstance(raw_controls, dict):
            raise ValueError("ui_profile.controls must be a dictionary")
        controls = {
            str(name).strip(): self._normalize_action_spec(spec)
            for name, spec in raw_controls.items()
            if str(name).strip()
        }

        raw_workflow = ui_profile.get("workflow", {})
        if raw_workflow is None:
            raw_workflow = {}
        if not isinstance(raw_workflow, dict):
            raise ValueError("ui_profile.workflow must be a dictionary")
        workflow = {
            "pre_point_actions": self._normalize_action_list(raw_workflow.get("pre_point_actions", [])),
            "post_point_actions": self._normalize_action_list(raw_workflow.get("post_point_actions", [])),
        }

        raw_options = ui_profile.get("options", {})
        if raw_options is None:
            raw_options = {}
        if not isinstance(raw_options, dict):
            raise ValueError("ui_profile.options must be a dictionary")
        click_interval_sec = float(raw_options.get("click_interval_sec", 0.15))
        move_duration_sec = float(raw_options.get("move_duration_sec", 0.0))
        if click_interval_sec < 0:
            raise ValueError("ui_profile.options.click_interval_sec must be non-negative")
        if move_duration_sec < 0:
            raise ValueError("ui_profile.options.move_duration_sec must be non-negative")
        options = {
            "activate_before_action": bool(raw_options.get("activate_before_action", True)),
            "click_interval_sec": click_interval_sec,
            "move_duration_sec": move_duration_sec,
            "flip_x": bool(raw_options.get("flip_x", False)),
            "flip_y": bool(raw_options.get("flip_y", False)),
        }

        return {
            "profile_kind": "frap_ui_profile",
            "profile_name": profile_name,
            "window_title_keyword": window_title_keyword,
            "image_region": {
                "left": image_left,
                "top": image_top,
                "width": image_width,
                "height": image_height,
            },
            "controls": controls,
            "workflow": workflow,
            "options": options,
        }

    def _normalize_action_list(self, raw_actions: Any) -> list[Any]:
        if raw_actions is None:
            return []
        if isinstance(raw_actions, str):
            actions = [raw_actions]
        elif isinstance(raw_actions, (list, tuple)):
            actions = list(raw_actions)
        else:
            raise ValueError("Workflow action lists must be a string, list, or tuple")

        normalized: list[Any] = []
        for item in actions:
            if isinstance(item, str):
                action_name = item.strip()
                if not action_name:
                    raise ValueError("Workflow action names must not be empty")
                normalized.append(action_name)
                continue
            if isinstance(item, dict):
                if "control" in item:
                    control_name = str(item.get("control", "") or "").strip()
                    if not control_name:
                        raise ValueError("Workflow action control references must not be empty")
                    normalized.append({"control": control_name})
                    continue
                normalized.append(self._normalize_action_spec(item))
                continue
            raise ValueError("Each workflow action must be a control name or an action dictionary")
        return normalized

    def _normalize_action_spec(self, action_spec: Any) -> dict:
        if not isinstance(action_spec, dict):
            raise ValueError("Action specs must be dictionaries")

        action_type = str(action_spec.get("type", "") or "").strip().lower()
        if action_type == "point":
            button = str(action_spec.get("button", "left") or "left").strip().lower()
            if button not in {"left", "right", "middle"}:
                raise ValueError("Point action button must be one of: left, right, middle")
            clicks = int(action_spec.get("clicks", 1))
            if clicks <= 0:
                raise ValueError("Point action clicks must be positive")
            return {
                "type": "point",
                "x": int(action_spec["x"]),
                "y": int(action_spec["y"]),
                "button": button,
                "clicks": clicks,
                "description": str(action_spec.get("description", "") or ""),
            }

        if action_type == "hotkey":
            return {
                "type": "hotkey",
                "keys": self._normalize_hotkey_keys(action_spec.get("keys", [])),
                "description": str(action_spec.get("description", "") or ""),
            }

        if action_type == "wait":
            seconds = float(action_spec.get("seconds", 0.0))
            if seconds < 0:
                raise ValueError("Wait action seconds must be non-negative")
            return {
                "type": "wait",
                "seconds": seconds,
                "description": str(action_spec.get("description", "") or ""),
            }

        if action_type == "text":
            interval_sec = float(action_spec.get("interval_sec", 0.0))
            if interval_sec < 0:
                raise ValueError("Text action interval_sec must be non-negative")
            return {
                "type": "text",
                "text": str(action_spec.get("text", "") or ""),
                "interval_sec": interval_sec,
                "description": str(action_spec.get("description", "") or ""),
            }

        raise ValueError("Unsupported action type. Supported types: point, hotkey, wait, text")

    def _normalize_hotkey_keys(self, keys: list[str] | tuple[str, ...] | str) -> list[str]:
        if isinstance(keys, str):
            key_items = [keys]
        elif isinstance(keys, (list, tuple)):
            key_items = list(keys)
        else:
            raise ValueError("keys must be a string, list, or tuple")

        normalized = [str(item).strip() for item in key_items if str(item).strip()]
        if not normalized:
            raise ValueError("At least one hotkey key is required")
        return normalized

    def _click_window_relative_from_window(
        self,
        window_info: dict,
        *,
        offset_x: int,
        offset_y: int,
        move_duration_sec: float,
        button: str = "left",
        clicks: int = 1,
    ) -> dict:
        window_left = int(window_info["left"])
        window_top = int(window_info["top"])
        window_width = int(window_info["width"])
        window_height = int(window_info["height"])
        if str(button) not in {"left", "right", "middle"}:
            raise ValueError("button must be one of: left, right, middle")
        if int(clicks) <= 0:
            raise ValueError("clicks must be positive")
        if not (0 <= int(offset_x) < window_width and 0 <= int(offset_y) < window_height):
            raise ValueError(
                "Relative click offset is outside the window bounds: "
                f"offset=({offset_x}, {offset_y}) window=({window_width}, {window_height})"
            )

        screen_x = window_left + int(offset_x)
        screen_y = window_top + int(offset_y)
        if str(button) == "left" and int(clicks) == 1:
            click_result = self.click_screen_point(
                screen_x=screen_x,
                screen_y=screen_y,
                move_duration_sec=float(move_duration_sec),
            )
        else:
            pyautogui = _import_pyautogui()
            original_pause = getattr(pyautogui, "PAUSE", 0.0)
            try:
                pyautogui.PAUSE = 0.0
                pyautogui.moveTo(screen_x, screen_y, duration=max(float(move_duration_sec), 0.0))
                pyautogui.click(screen_x, screen_y, clicks=int(clicks), button=str(button))
            finally:
                pyautogui.PAUSE = original_pause
            click_result = {
                "screen_x": int(screen_x),
                "screen_y": int(screen_y),
                "move_duration_sec": float(move_duration_sec),
                "status": "clicked",
            }
        return {
            "title": str(window_info["title"]),
            "offset_x": int(offset_x),
            "offset_y": int(offset_y),
            "window_left": window_left,
            "window_top": window_top,
            "button": str(button),
            "clicks": int(clicks),
            **click_result,
        }

    def _execute_action_sequence(
        self,
        action_items: list[Any],
        ui_profile: dict,
        window_info: dict,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for item in action_items:
            action_spec = self._resolve_action_reference(item, ui_profile)
            action_type = action_spec["type"]

            if action_type == "point":
                result = self._click_window_relative_from_window(
                    window_info,
                    offset_x=int(action_spec["x"]),
                    offset_y=int(action_spec["y"]),
                    move_duration_sec=float(ui_profile["options"]["move_duration_sec"]),
                    button=str(action_spec["button"]),
                    clicks=int(action_spec["clicks"]),
                )
                result["action_type"] = "point"
                result["description"] = str(action_spec.get("description", ""))
                results.append(result)
                if float(ui_profile["options"]["click_interval_sec"]) > 0:
                    time.sleep(float(ui_profile["options"]["click_interval_sec"]))
                continue

            if action_type == "hotkey":
                result = self.press_hotkey(
                    action_spec["keys"],
                    interval_sec=0.0,
                )
                result["action_type"] = "hotkey"
                result["description"] = str(action_spec.get("description", ""))
                results.append(result)
                continue

            if action_type == "text":
                result = self.type_text(
                    action_spec["text"],
                    interval_sec=float(action_spec["interval_sec"]),
                )
                result["action_type"] = "text"
                result["description"] = str(action_spec.get("description", ""))
                results.append(result)
                continue

            if action_type == "wait":
                result = {
                    "action_type": "wait",
                    "seconds": float(action_spec["seconds"]),
                    "description": str(action_spec.get("description", "")),
                    "status": "waited",
                }
                results.append(result)
                if float(action_spec["seconds"]) > 0:
                    time.sleep(float(action_spec["seconds"]))
                continue

            raise ValueError(f"Unsupported normalized action type: {action_type}")
        return results

    def _resolve_action_reference(self, item: Any, ui_profile: dict) -> dict:
        if isinstance(item, str):
            control_name = item.strip()
            if control_name not in ui_profile["controls"]:
                raise ValueError(f"ui_profile control not found: {control_name}")
            return dict(ui_profile["controls"][control_name])

        if isinstance(item, dict) and "control" in item:
            control_name = str(item.get("control", "") or "").strip()
            if control_name not in ui_profile["controls"]:
                raise ValueError(f"ui_profile control not found: {control_name}")
            return dict(ui_profile["controls"][control_name])

        if isinstance(item, dict):
            return self._normalize_action_spec(item)

        raise ValueError("Workflow actions must be strings or action dictionaries")
