from __future__ import annotations

import json
import time
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
    General FRAP point-plan and GUI workflow executor.

    This tool does not generate patterns by itself. Instead, it validates,
    stores, previews, maps, and executes point plans that can be created with
    regular Python code. It also provides a generic GUI profile layer for
    software-driven FRAP workflows when no native API is available.
    """

    planning_hint = (
        "Use this tool when a task already has FRAP points or needs to save, "
        "validate, map, preview, or execute a point plan, or when a GUI-based "
        "FRAP software workflow needs to be driven through a calibrated window "
        "profile. Pattern design can be implemented in normal Python before "
        "calling this tool."
    )
    execution_hint = (
        "Create or load a frap_point_plan, validate it before execution, build a "
        "linear image-to-screen mapping or a frap_ui_profile, preview the points "
        "on the image, then run the click plan or GUI workflow in dry_run mode "
        "before real execution."
    )

    def __init__(self, storage_manager=None, output_dir: str = "./output") -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir

    @tool_func
    def create_empty_plan(self, image_width: int, image_height: int) -> dict:
        """
        Create an empty FRAP point plan.

        Args:
            image_width: Width of the source image in pixels.
            image_height: Height of the source image in pixels.

        Returns:
            A FRAP point plan dictionary with no targets.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")

        return {
            "plan_kind": "frap_point_plan",
            "image_width": int(image_width),
            "image_height": int(image_height),
            "targets": [],
        }

    @tool_func
    def validate_point_plan(
        self,
        point_plan: dict,
        border_margin_px: int = 0,
        min_spacing_px: int = 0,
        max_targets: int = 0,
        sort_mode: str = "none",
    ) -> dict:
        """
        Validate and normalize a FRAP point plan.

        Args:
            point_plan: A candidate point plan dictionary.
            border_margin_px: Reject points closer than this many pixels to the image border.
            min_spacing_px: Reject points that are too close to previously accepted points.
            max_targets: Keep at most this many points. Use 0 to disable.
            sort_mode: Sorting strategy applied before spacing and truncation.
                Supported values are "none", "xy", and "yx".

        Returns:
            A normalized FRAP point plan dictionary.
        """
        image_width, image_height = self._extract_plan_dimensions(point_plan)
        if border_margin_px < 0 or min_spacing_px < 0:
            raise ValueError("border_margin_px and min_spacing_px must be non-negative")
        if max_targets < 0:
            raise ValueError("max_targets must be non-negative")
        if sort_mode not in {"none", "xy", "yx"}:
            raise ValueError("sort_mode must be one of: none, xy, yx")

        raw_targets = point_plan.get("targets")
        if not isinstance(raw_targets, list):
            raise ValueError("point_plan.targets must be a list")

        parsed_targets: list[dict[str, Any]] = []
        skipped_out_of_bounds = 0
        skipped_border = 0
        for index, item in enumerate(raw_targets, start=1):
            normalized = self._normalize_target(item, fallback_index=index)
            x_px = float(normalized["x_px"])
            y_px = float(normalized["y_px"])
            if not (0.0 <= x_px <= float(image_width - 1) and 0.0 <= y_px <= float(image_height - 1)):
                skipped_out_of_bounds += 1
                continue
            if not self._point_inside_image(x_px, y_px, image_width, image_height, border_margin_px):
                skipped_border += 1
                continue
            parsed_targets.append(normalized)

        ordered_targets = self._sort_targets(parsed_targets, sort_mode)
        accepted_targets = self._apply_target_spacing(
            ordered_targets,
            min_spacing_px=min_spacing_px,
            max_targets=max_targets,
        )
        for target_index, target in enumerate(accepted_targets, start=1):
            target["target_id"] = target_index

        return {
            "plan_kind": "frap_point_plan",
            "image_width": int(image_width),
            "image_height": int(image_height),
            "targets": accepted_targets,
            "summary": {
                "input_target_count": len(raw_targets),
                "accepted_target_count": len(accepted_targets),
                "skipped_out_of_bounds": skipped_out_of_bounds,
                "skipped_border": skipped_border,
                "border_margin_px": int(border_margin_px),
                "min_spacing_px": int(min_spacing_px),
                "max_targets": int(max_targets),
                "sort_mode": sort_mode,
            },
        }

    @tool_func
    def save_point_plan(self, point_plan: dict, filename: str = "frap_points.json") -> str:
        """
        Save a validated or draft point plan as JSON.

        Args:
            point_plan: Point plan to serialize.
            filename: Output JSON filename.

        Returns:
            Absolute path to the saved point plan.
        """
        image_width, image_height = self._extract_plan_dimensions(point_plan)
        payload = {
            "plan_kind": "frap_point_plan",
            "image_width": int(image_width),
            "image_height": int(image_height),
            "targets": self._to_json_ready(point_plan.get("targets", [])),
        }
        if "summary" in point_plan:
            payload["summary"] = self._to_json_ready(point_plan["summary"])
        return self._save_json_document(payload, filename, "FRAP point plan")

    @tool_func
    def load_point_plan(self, filename: str) -> dict:
        """
        Load a saved FRAP point plan.

        Args:
            filename: Relative or absolute path to a point plan JSON file.

        Returns:
            Parsed point plan dictionary.
        """
        return self._load_json_document(filename, expected_kind="frap_point_plan")

    @tool_func
    def build_linear_mapping(
        self,
        image_width: int,
        image_height: int,
        screen_left: int,
        screen_top: int,
        screen_width: int,
        screen_height: int,
        flip_x: bool = False,
        flip_y: bool = False,
        mapping_name: str = "default",
    ) -> dict:
        """
        Define a linear image-to-screen mapping.

        Args:
            image_width: Width of the source image in pixels.
            image_height: Height of the source image in pixels.
            screen_left: Left edge of the target FRAP screen region.
            screen_top: Top edge of the target FRAP screen region.
            screen_width: Width of the target FRAP screen region.
            screen_height: Height of the target FRAP screen region.
            flip_x: Whether to mirror the x axis during mapping.
            flip_y: Whether to mirror the y axis during mapping.
            mapping_name: Human-readable mapping label.

        Returns:
            A linear mapping dictionary.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")
        if screen_width <= 0 or screen_height <= 0:
            raise ValueError("screen_width and screen_height must be positive")

        return {
            "mapping_kind": "frap_linear_mapping",
            "mapping_name": str(mapping_name),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "screen_left": int(screen_left),
            "screen_top": int(screen_top),
            "screen_width": int(screen_width),
            "screen_height": int(screen_height),
            "flip_x": bool(flip_x),
            "flip_y": bool(flip_y),
        }

    @tool_func
    def save_mapping(self, mapping: dict, filename: str = "frap_mapping.json") -> str:
        """
        Save a mapping definition as JSON.

        Args:
            mapping: Mapping definition to save.
            filename: Output JSON filename.

        Returns:
            Absolute path to the saved mapping file.
        """
        return self._save_json_document(mapping, filename, "FRAP mapping")

    @tool_func
    def load_mapping(self, filename: str) -> dict:
        """
        Load a saved mapping definition.

        Args:
            filename: Relative or absolute path to a mapping JSON file.

        Returns:
            Parsed mapping dictionary.
        """
        return self._load_json_document(filename, expected_kind="frap_linear_mapping")

    @tool_func
    def map_point_plan(self, point_plan: dict, mapping: dict, clamp_to_region: bool = True) -> dict:
        """
        Convert image-space points into screen-space click points.

        Args:
            point_plan: FRAP point plan.
            mapping: Linear image-to-screen mapping.
            clamp_to_region: Clamp clicks to the target screen region.

        Returns:
            A click plan dictionary.
        """
        image_width, image_height = self._extract_plan_dimensions(point_plan)
        mapping_width = int(mapping.get("image_width", 0))
        mapping_height = int(mapping.get("image_height", 0))
        if image_width != mapping_width or image_height != mapping_height:
            raise ValueError(
                "point plan image size does not match mapping image size: "
                f"plan=({image_width}, {image_height}) mapping=({mapping_width}, {mapping_height})"
            )

        left = int(mapping["screen_left"])
        top = int(mapping["screen_top"])
        region_width = int(mapping["screen_width"])
        region_height = int(mapping["screen_height"])
        right = left + region_width - 1
        bottom = top + region_height - 1
        flip_x = bool(mapping.get("flip_x", False))
        flip_y = bool(mapping.get("flip_y", False))

        click_targets: list[dict[str, Any]] = []
        for item in point_plan.get("targets", []):
            target = self._normalize_target(item, fallback_index=len(click_targets) + 1)
            x_norm = float(target["x_px"]) / max(float(image_width - 1), 1.0)
            y_norm = float(target["y_px"]) / max(float(image_height - 1), 1.0)
            if flip_x:
                x_norm = 1.0 - x_norm
            if flip_y:
                y_norm = 1.0 - y_norm

            screen_x = left + x_norm * max(float(region_width - 1), 0.0)
            screen_y = top + y_norm * max(float(region_height - 1), 0.0)
            if clamp_to_region:
                screen_x = min(max(screen_x, float(left)), float(right))
                screen_y = min(max(screen_y, float(top)), float(bottom))

            click_targets.append(
                {
                    "target_id": int(target["target_id"]),
                    "x_px": float(target["x_px"]),
                    "y_px": float(target["y_px"]),
                    "screen_x": int(round(screen_x)),
                    "screen_y": int(round(screen_y)),
                    "label": str(target["label"]),
                    "group": str(target["group"]),
                }
            )

        return {
            "plan_kind": "frap_click_plan",
            "mapping_name": str(mapping.get("mapping_name", "")),
            "image_width": int(image_width),
            "image_height": int(image_height),
            "screen_region": {
                "left": left,
                "top": top,
                "width": region_width,
                "height": region_height,
            },
            "target_count": len(click_targets),
            "targets": click_targets,
        }

    @tool_func
    def save_click_plan(self, click_plan: dict, filename: str = "frap_click_plan.json") -> str:
        """
        Save a click plan as JSON.

        Args:
            click_plan: Click plan to save.
            filename: Output JSON filename.

        Returns:
            Absolute path to the saved click plan.
        """
        return self._save_json_document(click_plan, filename, "FRAP click plan")

    @tool_func
    def load_click_plan(self, filename: str) -> dict:
        """
        Load a saved click plan.

        Args:
            filename: Relative or absolute path to a click plan JSON file.

        Returns:
            Parsed click plan dictionary.
        """
        return self._load_json_document(filename, expected_kind="frap_click_plan")

    @tool_func
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

    @tool_func
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

    @tool_func
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

    @tool_func
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

    @tool_func
    def activate_window(self, title_keyword: str, dry_run: bool = True) -> dict:
        """
        Activate a desktop window by title keyword.

        Args:
            title_keyword: Partial window title used to locate the application window.
            dry_run: When True, only report the planned activation.

        Returns:
            A dictionary describing the planned or executed activation.
        """
        window_info = self.inspect_window(title_keyword)
        result = {
            "title_keyword": str(title_keyword),
            "title": window_info["title"],
            "left": int(window_info["left"]),
            "top": int(window_info["top"]),
            "width": int(window_info["width"]),
            "height": int(window_info["height"]),
            "dry_run": bool(dry_run),
        }
        if dry_run:
            result["status"] = "planned"
            return result

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

    @tool_func
    def click_window_relative(
        self,
        title_keyword: str,
        offset_x: int,
        offset_y: int,
        move_duration_sec: float = 0.0,
        dry_run: bool = True,
    ) -> dict:
        """
        Click a point expressed as an offset relative to the attached window.

        Args:
            title_keyword: Partial window title used to locate the application window.
            offset_x: Horizontal offset from the window's left edge.
            offset_y: Vertical offset from the window's top edge.
            move_duration_sec: Mouse move duration before clicking.
            dry_run: When True, only report the planned click.

        Returns:
            A dictionary describing the planned or executed click.
        """
        window_info = self.inspect_window(title_keyword)
        return self._click_window_relative_from_window(
            window_info,
            offset_x=int(offset_x),
            offset_y=int(offset_y),
            move_duration_sec=float(move_duration_sec),
            button="left",
            clicks=1,
            dry_run=bool(dry_run),
        )

    @tool_func
    def press_hotkey(
        self,
        keys: list[str] | tuple[str, ...] | str,
        interval_sec: float = 0.0,
        dry_run: bool = True,
    ) -> dict:
        """
        Press one or more keyboard shortcuts.

        Args:
            keys: A string key name or a list/tuple of keys passed to pyautogui.hotkey.
            interval_sec: Delay between keys in the hotkey sequence.
            dry_run: When True, only report the planned hotkey action.

        Returns:
            A dictionary describing the planned or executed hotkey.
        """
        key_list = self._normalize_hotkey_keys(keys)
        if interval_sec < 0:
            raise ValueError("interval_sec must be non-negative")

        result = {
            "keys": key_list,
            "interval_sec": float(interval_sec),
            "dry_run": bool(dry_run),
        }
        if dry_run:
            result["status"] = "planned"
            return result

        pyautogui = _import_pyautogui()
        original_pause = getattr(pyautogui, "PAUSE", 0.0)
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.hotkey(*key_list, interval=max(float(interval_sec), 0.0))
        finally:
            pyautogui.PAUSE = original_pause

        result["status"] = "pressed"
        return result

    @tool_func
    def type_text(self, text: str, interval_sec: float = 0.0, dry_run: bool = True) -> dict:
        """
        Type text into the currently focused application.

        Args:
            text: Text to type.
            interval_sec: Delay between characters.
            dry_run: When True, only report the planned text entry.

        Returns:
            A dictionary describing the planned or executed text entry.
        """
        if interval_sec < 0:
            raise ValueError("interval_sec must be non-negative")

        result = {
            "text": str(text),
            "interval_sec": float(interval_sec),
            "dry_run": bool(dry_run),
        }
        if dry_run:
            result["status"] = "planned"
            return result

        pyautogui = _import_pyautogui()
        original_pause = getattr(pyautogui, "PAUSE", 0.0)
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.write(str(text), interval=max(float(interval_sec), 0.0))
        finally:
            pyautogui.PAUSE = original_pause

        result["status"] = "typed"
        return result

    @tool_func
    def build_mapping_from_ui_profile(
        self,
        point_plan: dict,
        ui_profile: dict,
        clamp_to_window: bool = True,
    ) -> dict:
        """
        Build a linear mapping from a point plan and a live GUI profile attachment.

        Args:
            point_plan: FRAP point plan.
            ui_profile: GUI profile containing a relative image region and window title.
            clamp_to_window: Whether to verify the image region stays within the attached window.

        Returns:
            A linear mapping dictionary using absolute screen coordinates.
        """
        normalized_profile = self._normalize_ui_profile(ui_profile)
        window_info = self.inspect_window(normalized_profile["window_title_keyword"])
        return self._build_mapping_from_ui_profile_with_window(
            point_plan,
            normalized_profile,
            window_info,
            clamp_to_window=bool(clamp_to_window),
        )

    @tool_func
    def run_ui_actions(
        self,
        profile_path: str,
        action_names: list[str] | tuple[str, ...],
        dry_run: bool = True,
    ) -> dict:
        """
        Run a named batch of GUI actions described by a saved UI profile.

        Args:
            profile_path: Relative or absolute path to a saved UI profile JSON file.
            action_names: Control names to execute in order.
            dry_run: When True, only report the planned actions.

        Returns:
            A dictionary summarizing the executed or planned action batch.
        """
        if not isinstance(action_names, (list, tuple)) or not action_names:
            raise ValueError("action_names must be a non-empty list or tuple of control names")

        profile = self.load_ui_profile(profile_path)
        window_info = self.wait_for_window(profile["window_title_keyword"])
        actions: list[dict[str, Any]] = []
        if profile["options"]["activate_before_action"]:
            actions.append(self.activate_window(profile["window_title_keyword"], dry_run=dry_run))

        actions.extend(
            self._execute_action_sequence(
                list(action_names),
                profile,
                window_info,
                dry_run=bool(dry_run),
            )
        )
        return {
            "status": "ok",
            "dry_run": bool(dry_run),
            "profile_name": profile["profile_name"],
            "window": window_info,
            "actions": actions,
        }

    @tool_func
    def run_frap_workflow(
        self,
        point_plan_path: str,
        profile_path: str,
        dry_run: bool = True,
    ) -> dict:
        """
        Run a GUI-driven FRAP workflow using a point plan and a saved UI profile.

        Args:
            point_plan_path: Relative or absolute path to a saved point plan JSON file.
            profile_path: Relative or absolute path to a saved UI profile JSON file.
            dry_run: When True, only report the planned GUI actions and clicks.

        Returns:
            A dictionary summarizing the FRAP GUI workflow result.
        """
        point_plan = self.load_point_plan(point_plan_path)
        profile = self.load_ui_profile(profile_path)
        window_info = self.wait_for_window(profile["window_title_keyword"])
        workflow = profile["workflow"]
        options = profile["options"]
        actions: list[dict[str, Any]] = []

        if options["activate_before_action"]:
            actions.append(self.activate_window(profile["window_title_keyword"], dry_run=dry_run))

        actions.extend(
            self._execute_action_sequence(
                workflow["pre_point_actions"],
                profile,
                window_info,
                dry_run=bool(dry_run),
            )
        )

        mapping = self._build_mapping_from_ui_profile_with_window(
            point_plan,
            profile,
            window_info,
            clamp_to_window=True,
        )
        click_plan = self.map_point_plan(point_plan, mapping, clamp_to_region=True)

        click_logs: list[dict[str, Any]] = []
        for index, target in enumerate(click_plan.get("targets", [])):
            click_result = self.click_screen_point(
                screen_x=int(target["screen_x"]),
                screen_y=int(target["screen_y"]),
                move_duration_sec=float(options["move_duration_sec"]),
                dry_run=bool(dry_run),
            )
            click_logs.append(
                {
                    "target_id": int(target["target_id"]),
                    "label": str(target["label"]),
                    "group": str(target["group"]),
                    **click_result,
                }
            )
            if (
                not dry_run
                and index < len(click_plan["targets"]) - 1
                and float(options["click_interval_sec"]) > 0
            ):
                time.sleep(float(options["click_interval_sec"]))

        actions.extend(
            self._execute_action_sequence(
                workflow["post_point_actions"],
                profile,
                window_info,
                dry_run=bool(dry_run),
            )
        )

        return {
            "status": "ok",
            "dry_run": bool(dry_run),
            "profile_name": profile["profile_name"],
            "window": window_info,
            "mapping": mapping,
            "click_plan": click_plan,
            "point_clicks": click_logs,
            "actions": actions,
        }

    @tool_func
    def preview_point_plan(
        self,
        image_path: str,
        point_plan_path: str,
        output_name: str = "frap_points_preview.png",
        marker_radius_px: int = 10,
        draw_labels: bool = True,
    ) -> str:
        """
        Draw a point plan on top of an image and save the preview.

        Args:
            image_path: Relative or absolute path to the source image.
            point_plan_path: Relative or absolute path to a saved point plan JSON file.
            output_name: Output preview PNG filename.
            marker_radius_px: Circle radius around each point.
            draw_labels: Whether to annotate labels or target ids.

        Returns:
            Absolute path to the saved preview image.
        """
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError("matplotlib is required to save FRAP preview images.") from exc

        point_plan = self.load_point_plan(point_plan_path)
        image_array = self._read_image_array(image_path)
        display_image = self._prepare_display_image(image_array)
        output_path = self._resolve_output_path(output_name)

        figure, axis = plt.subplots(figsize=(8, 8))
        try:
            if display_image.ndim == 2:
                axis.imshow(display_image, cmap="gray")
            else:
                axis.imshow(display_image)

            for item in point_plan.get("targets", []):
                target = self._normalize_target(item, fallback_index=0)
                x_px = float(target["x_px"])
                y_px = float(target["y_px"])
                circle = plt.Circle((x_px, y_px), marker_radius_px, color="red", fill=False, linewidth=1.5)
                axis.add_patch(circle)
                axis.scatter([x_px], [y_px], color="yellow", s=18)
                if draw_labels:
                    label_text = str(target["label"] or target["target_id"])
                    axis.text(
                        x_px + marker_radius_px * 0.6,
                        y_px - marker_radius_px * 0.6,
                        label_text,
                        color="white",
                        fontsize=8,
                        bbox={"facecolor": "black", "alpha": 0.6, "pad": 1},
                    )

            axis.set_title("FRAP point plan preview")
            axis.axis("off")
            figure.tight_layout()
            figure.savefig(output_path, dpi=160, bbox_inches="tight")
        finally:
            plt.close(figure)

        self._register_output(output_path.name, "FRAP point plan preview", "png")
        return str(output_path)

    @tool_func
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

    @tool_func
    def click_screen_point(
        self,
        screen_x: int,
        screen_y: int,
        move_duration_sec: float = 0.0,
        dry_run: bool = True,
    ) -> dict:
        """
        Click one point on the screen.

        Args:
            screen_x: Horizontal screen coordinate.
            screen_y: Vertical screen coordinate.
            move_duration_sec: Mouse move duration before clicking.
            dry_run: When True, return the planned click without touching the mouse.

        Returns:
            A dictionary describing the executed or planned click.
        """
        result = {
            "screen_x": int(screen_x),
            "screen_y": int(screen_y),
            "move_duration_sec": float(move_duration_sec),
            "dry_run": bool(dry_run),
        }
        if dry_run:
            result["status"] = "planned"
            return result

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

    @tool_func
    def execute_click_plan(
        self,
        click_plan_path: str,
        click_interval_sec: float = 0.15,
        activate_window: bool = False,
        window_title_keyword: str = "",
        move_duration_sec: float = 0.0,
        dry_run: bool = True,
    ) -> dict:
        """
        Execute a saved click plan by issuing sequential mouse clicks.

        Args:
            click_plan_path: Relative or absolute path to a saved click plan JSON file.
            click_interval_sec: Delay between clicks in seconds.
            activate_window: Whether to activate the FRAP window before clicking.
            window_title_keyword: Partial window title used when activate_window is True.
            move_duration_sec: Mouse move duration for each click.
            dry_run: When True, only simulate execution and return the planned clicks.

        Returns:
            A dictionary summarizing the click execution result.
        """
        if click_interval_sec < 0:
            raise ValueError("click_interval_sec must be non-negative")

        click_plan = self.load_click_plan(click_plan_path)
        window_info: dict[str, Any] | None = None
        if activate_window:
            if not str(window_title_keyword).strip():
                raise ValueError("window_title_keyword is required when activate_window=True")
            window_info = self.inspect_window(window_title_keyword)
            if not dry_run:
                pygetwindow = _import_pygetwindow()
                matched_windows = pygetwindow.getWindowsWithTitle(window_info["title"])
                if matched_windows:
                    matched_windows[0].activate()
                    time.sleep(0.2)

        clicks: list[dict[str, Any]] = []
        targets = list(click_plan.get("targets", []))
        for index, target in enumerate(targets):
            clicks.append(
                self.click_screen_point(
                    screen_x=int(target["screen_x"]),
                    screen_y=int(target["screen_y"]),
                    move_duration_sec=move_duration_sec,
                    dry_run=dry_run,
                )
            )
            if index < len(targets) - 1 and click_interval_sec > 0:
                time.sleep(click_interval_sec)

        return {
            "status": "ok",
            "dry_run": bool(dry_run),
            "executed_count": len(clicks),
            "click_interval_sec": float(click_interval_sec),
            "window": window_info,
            "clicks": clicks,
        }

    def _extract_plan_dimensions(self, point_plan: dict) -> tuple[int, int]:
        if not isinstance(point_plan, dict):
            raise ValueError("point_plan must be a dictionary")
        image_width = int(point_plan.get("image_width", 0))
        image_height = int(point_plan.get("image_height", 0))
        if image_width <= 0 or image_height <= 0:
            raise ValueError("point_plan must include positive image_width and image_height")
        return image_width, image_height

    def _normalize_target(self, raw_target: Any, fallback_index: int) -> dict[str, Any]:
        if not isinstance(raw_target, dict):
            raise ValueError("Each target must be a dictionary")
        if "x_px" not in raw_target or "y_px" not in raw_target:
            raise ValueError("Each target must include x_px and y_px")

        target_id = raw_target.get("target_id", fallback_index)
        label = raw_target.get("label", "")
        group = raw_target.get("group", "")
        return {
            "target_id": int(target_id),
            "x_px": float(raw_target["x_px"]),
            "y_px": float(raw_target["y_px"]),
            "label": str(label),
            "group": str(group),
        }

    def _sort_targets(self, targets: list[dict[str, Any]], sort_mode: str) -> list[dict[str, Any]]:
        if sort_mode == "none":
            return list(targets)
        if sort_mode == "xy":
            return sorted(targets, key=lambda item: (float(item["x_px"]), float(item["y_px"])))
        return sorted(targets, key=lambda item: (float(item["y_px"]), float(item["x_px"])))

    def _apply_target_spacing(
        self,
        targets: list[dict[str, Any]],
        *,
        min_spacing_px: int,
        max_targets: int,
    ) -> list[dict[str, Any]]:
        if not targets:
            return []

        spacing_sq = float(min_spacing_px * min_spacing_px)
        accepted: list[dict[str, Any]] = []
        for target in targets:
            point = (float(target["x_px"]), float(target["y_px"]))
            if min_spacing_px > 0 and any(
                self._distance_squared(point, (float(item["x_px"]), float(item["y_px"]))) < spacing_sq
                for item in accepted
            ):
                continue
            accepted.append(dict(target))
            if max_targets > 0 and len(accepted) >= max_targets:
                break
        return accepted

    def _point_inside_image(
        self,
        x_px: float,
        y_px: float,
        width: int,
        height: int,
        border_margin_px: int,
    ) -> bool:
        return (
            border_margin_px <= x_px <= (width - 1 - border_margin_px)
            and border_margin_px <= y_px <= (height - 1 - border_margin_px)
        )

    def _resolve_input_path(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(self.output_dir, candidate).expanduser()
        return candidate.resolve()

    def _resolve_output_path(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(self.output_dir, candidate).expanduser()
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate.resolve()

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

    def _distance_squared(self, point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
        dx = float(point_a[0]) - float(point_b[0])
        dy = float(point_a[1]) - float(point_b[1])
        return dx * dx + dy * dy

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
        dry_run: bool,
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
        if bool(dry_run) or (str(button) == "left" and int(clicks) == 1):
            click_result = self.click_screen_point(
                screen_x=screen_x,
                screen_y=screen_y,
                move_duration_sec=float(move_duration_sec),
                dry_run=bool(dry_run),
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
                "dry_run": bool(dry_run),
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

    def _build_mapping_from_ui_profile_with_window(
        self,
        point_plan: dict,
        ui_profile: dict,
        window_info: dict,
        *,
        clamp_to_window: bool,
    ) -> dict:
        image_width, image_height = self._extract_plan_dimensions(point_plan)
        image_region = ui_profile["image_region"]
        region_left = int(image_region["left"])
        region_top = int(image_region["top"])
        region_width = int(image_region["width"])
        region_height = int(image_region["height"])
        window_width = int(window_info["width"])
        window_height = int(window_info["height"])

        if clamp_to_window:
            if region_left < 0 or region_top < 0:
                raise ValueError("ui_profile image region offsets must be non-negative")
            if region_left + region_width > window_width or region_top + region_height > window_height:
                raise ValueError(
                    "ui_profile image region exceeds the attached window bounds: "
                    f"region=({region_left}, {region_top}, {region_width}, {region_height}) "
                    f"window=({window_width}, {window_height})"
                )

        return self.build_linear_mapping(
            image_width=image_width,
            image_height=image_height,
            screen_left=int(window_info["left"]) + region_left,
            screen_top=int(window_info["top"]) + region_top,
            screen_width=region_width,
            screen_height=region_height,
            flip_x=bool(ui_profile["options"].get("flip_x", False)),
            flip_y=bool(ui_profile["options"].get("flip_y", False)),
            mapping_name=str(ui_profile["profile_name"]),
        )

    def _execute_action_sequence(
        self,
        action_items: list[Any],
        ui_profile: dict,
        window_info: dict,
        *,
        dry_run: bool,
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
                    dry_run=bool(dry_run),
                )
                result["action_type"] = "point"
                result["description"] = str(action_spec.get("description", ""))
                results.append(result)
                if not dry_run and float(ui_profile["options"]["click_interval_sec"]) > 0:
                    time.sleep(float(ui_profile["options"]["click_interval_sec"]))
                continue

            if action_type == "hotkey":
                result = self.press_hotkey(
                    action_spec["keys"],
                    interval_sec=0.0,
                    dry_run=bool(dry_run),
                )
                result["action_type"] = "hotkey"
                result["description"] = str(action_spec.get("description", ""))
                results.append(result)
                continue

            if action_type == "text":
                result = self.type_text(
                    action_spec["text"],
                    interval_sec=float(action_spec["interval_sec"]),
                    dry_run=bool(dry_run),
                )
                result["action_type"] = "text"
                result["description"] = str(action_spec.get("description", ""))
                results.append(result)
                continue

            if action_type == "wait":
                result = {
                    "action_type": "wait",
                    "seconds": float(action_spec["seconds"]),
                    "dry_run": bool(dry_run),
                    "description": str(action_spec.get("description", "")),
                    "status": "planned" if dry_run else "waited",
                }
                results.append(result)
                if not dry_run and float(action_spec["seconds"]) > 0:
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
