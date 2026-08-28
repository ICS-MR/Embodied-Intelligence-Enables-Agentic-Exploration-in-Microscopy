from __future__ import annotations

import logging
import json
import math
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tool.base import BaseTool, tool_func

logger = logging.getLogger(__name__)


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


def _import_cv2():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("cv2 is required for FRAP GUI control and cell detection.") from exc
    return cv2


def _import_cellpose_2d():
    try:
        from core_tool.cellpose_tool import Cellpose2D
    except Exception as exc:
        raise RuntimeError("Cellpose is required for FRAP cell detection.") from exc
    return Cellpose2D


def _find_processes_by_name(process_name: str) -> list[Any]:
    try:
        import psutil
    except Exception as exc:
        raise RuntimeError("psutil is required for FRAP process verification.") from exc

    normalized = str(process_name or "").strip().lower()
    if not normalized:
        return []
    return [
        proc
        for proc in psutil.process_iter(["name"])
        if str(proc.info.get("name", "") or "").strip().lower() == normalized
    ]


@dataclass(frozen=True)
class _FrapCoordinateTransform:
    source_width: int
    source_height: int
    display_left: int
    display_top: int
    display_right: int
    display_bottom: int
    display_flip_x: bool
    display_flip_y: bool
    pixel_size_x_um: float
    pixel_size_y_um: float

    def __post_init__(self) -> None:
        if self.source_width <= 1 or self.source_height <= 1:
            raise ValueError("FRAP source image dimensions must be greater than one pixel.")
        if self.display_right <= self.display_left or self.display_bottom <= self.display_top:
            raise ValueError("FRAP display region must have positive width and height.")
        if self.pixel_size_x_um <= 0 or self.pixel_size_y_um <= 0:
            raise ValueError("FRAP physical pixel sizes must be positive.")

    @property
    def display_width(self) -> int:
        return self.display_right - self.display_left + 1

    @property
    def display_height(self) -> int:
        return self.display_bottom - self.display_top + 1

    def source_to_display(self, x_px: float, y_px: float) -> tuple[float, float]:
        self._validate_source_point(x_px, y_px)
        mapped_x = float(self.source_width - 1) - float(x_px) if self.display_flip_x else float(x_px)
        mapped_y = float(self.source_height - 1) - float(y_px) if self.display_flip_y else float(y_px)
        return (
            mapped_x * float(self.display_width - 1) / float(self.source_width - 1),
            mapped_y * float(self.display_height - 1) / float(self.source_height - 1),
        )

    def display_to_source(self, x_px: float, y_px: float) -> tuple[float, float]:
        self._validate_display_point(x_px, y_px)
        source_x = float(x_px) * float(self.source_width - 1) / float(self.display_width - 1)
        source_y = float(y_px) * float(self.source_height - 1) / float(self.display_height - 1)
        if self.display_flip_x:
            source_x = float(self.source_width - 1) - source_x
        if self.display_flip_y:
            source_y = float(self.source_height - 1) - source_y
        return source_x, source_y

    def source_to_screen(self, x_px: float, y_px: float) -> tuple[int, int]:
        display_x, display_y = self.source_to_display(x_px, y_px)
        return (
            int(round(float(self.display_left) + display_x)),
            int(round(float(self.display_top) + display_y)),
        )

    def source_to_view_um(self, x_px: float, y_px: float) -> tuple[float, float]:
        self._validate_source_point(x_px, y_px)
        center_x_px = float(self.source_width - 1) / 2.0
        center_y_px = float(self.source_height - 1) / 2.0
        return (
            (float(x_px) - center_x_px) * self.pixel_size_x_um,
            (float(y_px) - center_y_px) * self.pixel_size_y_um,
        )

    def view_um_to_source(self, x_um: float, y_um: float) -> tuple[float, float]:
        center_x_px = float(self.source_width - 1) / 2.0
        center_y_px = float(self.source_height - 1) / 2.0
        source_x = center_x_px + float(x_um) / self.pixel_size_x_um
        source_y = center_y_px + float(y_um) / self.pixel_size_y_um
        self._validate_source_point(source_x, source_y, coordinate_name="view target")
        return source_x, source_y

    def display_to_view_um(self, x_px: float, y_px: float) -> tuple[float, float]:
        source_x, source_y = self.display_to_source(x_px, y_px)
        return self.source_to_view_um(source_x, source_y)

    def view_um_to_screen(self, x_um: float, y_um: float) -> tuple[int, int]:
        source_x, source_y = self.view_um_to_source(x_um, y_um)
        return self.source_to_screen(source_x, source_y)

    def _validate_source_point(
        self,
        x_px: float,
        y_px: float,
        *,
        coordinate_name: str = "source image point",
    ) -> None:
        tolerance = 1e-6
        if not (
            -tolerance <= float(x_px) <= float(self.source_width - 1) + tolerance
            and -tolerance <= float(y_px) <= float(self.source_height - 1) + tolerance
        ):
            raise ValueError(
                f"FRAP {coordinate_name} is outside the current source image: "
                f"point=({x_px}, {y_px}) image=({self.source_width}, {self.source_height})"
            )

    def _validate_display_point(self, x_px: float, y_px: float) -> None:
        tolerance = 1e-6
        if not (
            -tolerance <= float(x_px) <= float(self.display_width - 1) + tolerance
            and -tolerance <= float(y_px) <= float(self.display_height - 1) + tolerance
        ):
            raise ValueError(
                "FRAP display point is outside the configured image region: "
                f"point=({x_px}, {y_px}) display=({self.display_width}, {self.display_height})"
            )


class Frap(BaseTool):
    """Provide FRAP control and field-of-view analysis operations."""

    _active_instance: Frap | None = None
    _ELLIPSE_POINT_COUNT = 36

    planning_hint = (
        "Use for FRAP workflows that start and stop FRAP, detect cells, extract contours, "
        "and photobleach selected field-centered micron coordinates. Spatial patterns can be "
        "executed as sampled coordinate trajectories derived from fitted contours or generated "
        "paths. Bleaching intensity is controlled by sampling point spacing."
    )
    execution_hint = (
        "Call laser_on before cell_detection, cell_contour_extraction, or laser_position; "
        "call laser_off after the full bleaching sequence is complete. Treat laser_position "
        "x and y as microns relative to the field center. Use detected cell centers, fitted "
        "contour points, or sampled trajectories as laser_position targets. Adjust relative "
        "bleaching dose by changing point spacing: increase point spacing to lower the dose; "
        "decrease point spacing to raise the dose."
    )

    def __init__(
        self,
        storage_manager=None,
        output_dir: str = "./output",
        launch_command: str | list[str] | None = None,
        launch_workdir: str = "",
        cellpose_model_type: str = "cpsam",
        cellpose_device: str | None = None,
    ) -> None:
        self._profile_path = Path(__file__).resolve().parents[1] / "docs_public" / "frap" / "frap_ui_profile.json"
        self._laser_enabled = False
        self._closed = False
        self._session_prepared = False
        self._storage_manager = storage_manager
        self._output_dir = str(output_dir)
        self._cellpose_model_type = str(cellpose_model_type).strip() or "cpsam"
        self._cellpose_device = str(cellpose_device).strip() if cellpose_device else None
        self._cellpose = None
        self._profile = self._load_profile()
        profile_launch_command = self._profile.get("launch_command") or None
        profile_launch_workdir = str(self._profile.get("launch_workdir", "")).strip()
        self._launch_command = self._normalize_launch_command(
            launch_command if launch_command is not None else profile_launch_command
        )
        self._launch_workdir = str(launch_workdir or profile_launch_workdir).strip()
        self._window_info: dict[str, Any] = {}
        type(self)._active_instance = self

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self) -> Frap:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def close(self) -> None:
        """Close the current cellSens window used by this FRAP helper."""
        if bool(getattr(self, "_closed", True)):
            return
        self._closed = True
        if type(self)._active_instance is self:
            type(self)._active_instance = None
        self.release_session()

    def release_session(self) -> None:
        """Release the current cellSens FRAP session while keeping this tool reusable."""
        was_prepared = bool(self._session_prepared or self._window_info)
        self._laser_enabled = False
        self._session_prepared = False
        profile = getattr(self, "_profile", None)
        if not profile:
            self._window_info = {}
            return
        try:
            window_info = self._wait_for_window(profile["window_title_keyword"], timeout_sec=1.0)
        except RuntimeError as exc:
            self._window_info = {}
            if was_prepared:
                raise RuntimeError(
                    "FRAP session was prepared, but the cellSens window was not found during release."
                ) from exc
            return
        try:
            self._activate_window_if_needed(profile, window_info=window_info)
            self._close_window(window_info, profile=profile)
        except Exception as exc:
            raise RuntimeError("Failed to release the FRAP cellSens session.") from exc
        finally:
            self._window_info = {}

    @tool_func
    def laser_on(self) -> None:
        """
        Turn on the FRAP operation switch.

        This method must be called before laser_position(), cell_detection(),
        or cell_contour_extraction(). It starts the FRAP operation.

        """
        self._ensure_prepared_once()
        self._run_control_sequence(
            profile=self._profile,
            control_names=("frap_start_button",),
        )
        settle_seconds = float(self._profile["options"].get("frap_start_settle_seconds", 0.0))
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        self._wait_for_state_reference("frap_start_state_check", "laser_on")
        self._laser_enabled = True

    @tool_func
    def laser_off(self) -> None:
        """
        Turn off the FRAP operation switch.

        Call this method after completing the laser_position() bleaching
        sequence. This stops FRAP operation but does not release the session.
        """
        if not self._session_prepared:
            try:
                self._window_info = self._wait_for_window(self._profile["window_title_keyword"], timeout_sec=1.0)
            except RuntimeError:
                self._laser_enabled = False
                return
            self._activate_window_if_needed(self._profile, window_info=self._window_info)
        else:
            self._window_info = self._wait_for_window(self._profile["window_title_keyword"])
            self._activate_window_if_needed(self._profile, window_info=self._window_info)
        self._run_control_sequence(
            profile=self._profile,
            control_names=("frap_stop_button",),
        )
        settle_seconds = float(self._profile["options"].get("frap_stop_settle_seconds", 0.0))
        if settle_seconds > 0:
            time.sleep(settle_seconds)
        self._wait_for_state_reference("frap_stop_state_check", "laser_off")
        self._laser_enabled = False

    @tool_func
    @staticmethod
    def laser_position(x: int, y: int) -> None:
        """
        Position the laser at the specified coordinates and perform one
        bleaching operation at that location.

        FRAP must be turned on with laser_on() before calling this method.
        The method may be called repeatedly to bleach multiple positions.

        Args:
            x: X-axis position in microns relative to the center of the current field of view.
            y: Y-axis position in microns relative to the center of the current field of view.
        """
        instance = Frap._require_active_instance()
        instance._laser_position_impl(x, y)

    def _require_laser_enabled(self, operation_name: str) -> None:
        if not self._laser_enabled:
            raise RuntimeError(f"{operation_name} requires FRAP to be started first.")

    def _laser_position_impl(self, x: int, y: int) -> None:
        self._require_laser_enabled("laser_position")
        self._ensure_prepared_once()

        options = self._profile["options"]
        transform = self._build_coordinate_transform()
        target_x, target_y = transform.view_um_to_screen(float(x), float(y))
        self._click_screen_absolute(
            absolute_x=target_x,
            absolute_y=target_y,
            move_duration_sec=float(options.get("move_duration_sec", 0.0)),
            button="left",
        )
        if float(options.get("click_interval_sec", 0.0)) > 0:
            time.sleep(float(options.get("click_interval_sec", 0.0)))

    @tool_func
    @staticmethod
    def cell_detection() -> dict:
        """
        Detect all usable cells in the current field of view.

        FRAP must be turned on with laser_on() before calling this method.

        Returns:
            Dictionary containing a ``cells`` list. Each item contains ``cell_id``
            plus ``x`` and ``y`` coordinates in microns relative to the field center.
            The list is empty when no usable cells are detected.
        """
        instance = Frap._require_active_instance()
        return instance._cell_detection_impl()

    def _cell_detection_impl(self) -> dict:
        self._require_laser_enabled("cell_detection")
        self._ensure_prepared_once()
        frame = self._capture_image_region(self._profile)
        analysis = self._segment_and_analyze_cells(frame)
        if not analysis:
            return {"cells": []}
        transform = self._build_coordinate_transform()
        self._validate_captured_frame(frame, transform)
        cells = []
        for cell_id, candidate in enumerate(analysis["candidates"], start=1):
            center_px = candidate["center_px"]
            x_um, y_um = transform.display_to_view_um(
                float(center_px["x"]),
                float(center_px["y"]),
            )
            cells.append(
                {
                    "cell_id": int(cell_id),
                    "x": x_um,
                    "y": y_um,
                }
            )
        return {"cells": cells}

    @tool_func
    @staticmethod
    def cell_contour_extraction() -> dict:
        """
        Extract all usable cell contours from the current field of view.

        FRAP must be turned on with laser_on() before calling this method.

        Returns:
            Dictionary containing a ``cells`` list. Each item contains ``cell_id``,
            and fitted ellipse ``points`` represented as ``[x, y]`` pairs in
            field-centered microns. The list is empty when no usable contours
            are extracted.
        """
        instance = Frap._require_active_instance()
        return instance._cell_contour_extraction_impl()

    def _cell_contour_extraction_impl(self) -> dict:
        self._require_laser_enabled("cell_contour_extraction")
        self._ensure_prepared_once()
        frame = self._capture_image_region(self._profile)
        analysis = self._segment_and_analyze_cells(frame)
        if not analysis:
            return {"cells": []}

        transform = self._build_coordinate_transform()
        self._validate_captured_frame(frame, transform)
        cells = []
        for candidate in analysis["candidates"]:
            contour_px = np.asarray(candidate["contour"], dtype=float).reshape(-1, 2)
            contour_points = [
                transform.display_to_view_um(float(point[0]), float(point[1]))
                for point in contour_px
            ]
            points = self._fit_ellipse_points(contour_points)
            if points is None:
                continue
            cells.append(
                {
                    "cell_id": int(len(cells) + 1),
                    "points": [[float(x), float(y)] for x, y in points],
                }
            )
        return {"cells": cells}

    @classmethod
    def _fit_ellipse_points(
        cls,
        contour_points: list[tuple[float, float]],
    ) -> list[tuple[float, float]] | None:
        if len(contour_points) < 5:
            return None

        cv2 = _import_cv2()
        contour = np.asarray(contour_points, dtype=np.float32).reshape(-1, 1, 2)
        try:
            (center_x, center_y), (axis_x, axis_y), angle_degrees = cv2.fitEllipse(contour)
        except cv2.error:
            return None

        if axis_x <= 0 or axis_y <= 0:
            return None

        angle = np.deg2rad(float(angle_degrees))
        cos_angle = float(np.cos(angle))
        sin_angle = float(np.sin(angle))
        theta_values = np.linspace(
            0.0,
            2.0 * np.pi,
            num=cls._ELLIPSE_POINT_COUNT,
            endpoint=False,
        )
        points = []
        for theta in theta_values:
            local_x = 0.5 * float(axis_x) * float(np.cos(theta))
            local_y = 0.5 * float(axis_y) * float(np.sin(theta))
            points.append(
                (
                    float(center_x + local_x * cos_angle - local_y * sin_angle),
                    float(center_y + local_x * sin_angle + local_y * cos_angle),
                )
            )
        return points

    @classmethod
    def _require_active_instance(cls) -> Frap:
        instance = cls._active_instance
        if instance is None:
            raise RuntimeError("Frap must be instantiated before using this static method.")
        return instance

    def _load_profile(self) -> dict:
        path = self._profile_path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"FRAP UI profile not found: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        image_region = payload.get("image_region", {})
        options = payload.get("options", {})
        if not isinstance(image_region, dict) or not isinstance(options, dict):
            raise ValueError("Invalid FRAP profile structure")

        startup_window_timeout_sec = float(options.get("startup_window_timeout_sec", 60.0))
        startup_poll_interval_sec = float(options.get("startup_poll_interval_sec", 1.0))
        startup_settle_seconds = float(options.get("startup_settle_seconds", 0.0))
        if startup_window_timeout_sec <= 0:
            raise ValueError("FRAP startup_window_timeout_sec must be greater than zero.")
        if startup_poll_interval_sec <= 0:
            raise ValueError("FRAP startup_poll_interval_sec must be greater than zero.")
        if startup_settle_seconds < 0:
            raise ValueError("FRAP startup_settle_seconds must be non-negative.")

        startup_match_threshold = float(options.get("startup_match_threshold", 0.8))
        if not 0.0 < startup_match_threshold <= 1.0:
            raise ValueError("FRAP startup_match_threshold must be within (0, 1].")
        startup_reference_checks_raw = options.get("startup_reference_checks", {})
        if not isinstance(startup_reference_checks_raw, dict):
            raise ValueError("FRAP startup_reference_checks must be an object.")
        unknown_check_keys = [
            key for key in startup_reference_checks_raw if key not in ("pre_click", "post_click")
        ]
        if unknown_check_keys:
            raise ValueError(
                "FRAP startup_reference_checks has unknown keys: "
                + ", ".join(sorted(unknown_check_keys))
            )
        startup_reference_checks = {
            "pre_click": self._parse_startup_reference_check(
                startup_reference_checks_raw.get("pre_click", {}), "pre_click"
            ),
            "post_click": self._parse_startup_reference_check(
                startup_reference_checks_raw.get("post_click", {}), "post_click"
            ),
        }

        required_region_keys = ("left", "top", "right", "bottom", "source_width", "source_height")
        missing_region_keys = [key for key in required_region_keys if key not in image_region]
        if missing_region_keys:
            raise ValueError(
                "FRAP profile image_region is missing required fields: "
                + ", ".join(missing_region_keys)
            )

        return {
            "window_title_keyword": str(payload.get("window_title_keyword", "")).strip(),
            "launch_command": payload.get("launch_command"),
            "launch_workdir": str(payload.get("launch_workdir", "")).strip(),
            "image_region": {
                "left": int(image_region.get("left", 0)),
                "top": int(image_region.get("top", 0)),
                "right": int(image_region.get("right", 0)),
                "bottom": int(image_region.get("bottom", 0)),
                "source_width": int(image_region.get("source_width", 0)),
                "source_height": int(image_region.get("source_height", 0)),
            },
            "controls": dict(payload.get("controls", {})),
            "options": {
                "activate_before_action": bool(options.get("activate_before_action", True)),
                "startup_window_timeout_sec": startup_window_timeout_sec,
                "startup_poll_interval_sec": startup_poll_interval_sec,
                "startup_settle_seconds": startup_settle_seconds,
                "startup_ready_settle_seconds": float(options.get("startup_ready_settle_seconds", 15.0)),
                "startup_match_threshold": startup_match_threshold,
                "frap_start_settle_seconds": float(options.get("frap_start_settle_seconds", 1.0)),
                "frap_stop_settle_seconds": float(options.get("frap_stop_settle_seconds", 1.5)),
                "frap_start_state_check": self._parse_startup_reference_check(
                    options.get("frap_start_state_check", {}), "frap_start_state_check"
                ),
                "frap_stop_state_check": self._parse_startup_reference_check(
                    options.get("frap_stop_state_check", {}), "frap_stop_state_check"
                ),
                "close_settle_seconds": float(options.get("close_settle_seconds", 2.0)),
                "close_window_timeout_sec": float(options.get("close_window_timeout_sec", 15.0)),
                "close_process_timeout_sec": float(options.get("close_process_timeout_sec", 30.0)),
                "startup_reference_checks": startup_reference_checks,
                "click_interval_sec": float(options.get("click_interval_sec", 0.15)),
                "move_duration_sec": float(options.get("move_duration_sec", 0.0)),
                "flip_x": bool(options.get("flip_x", False)),
                "flip_y": bool(options.get("flip_y", False)),
                "pixel_size_x_um": float(options.get("pixel_size_x_um", 0.0)),
                "pixel_size_y_um": float(options.get("pixel_size_y_um", 0.0)),
                "cellpose_channel": int(options.get("cellpose_channel", 1)),
                "cellpose_diameter": (
                    float(options["cellpose_diameter"])
                    if options.get("cellpose_diameter") is not None
                    else None
                ),
                "cellpose_area_percentile": float(options.get("cellpose_area_percentile", 60.0)),
                "cellpose_min_area_px": float(options.get("cellpose_min_area_px", 3000.0)),
                "cellpose_distance_weight": float(options.get("cellpose_distance_weight", 0.5)),
            },
        }

    def _build_coordinate_transform(self) -> _FrapCoordinateTransform:
        region = self._profile["image_region"]
        options = self._profile["options"]
        source_width = int(region["source_width"])
        source_height = int(region["source_height"])
        return _FrapCoordinateTransform(
            source_width=source_width,
            source_height=source_height,
            display_left=int(region["left"]),
            display_top=int(region["top"]),
            display_right=int(region["right"]),
            display_bottom=int(region["bottom"]),
            display_flip_x=bool(options.get("flip_x", False)),
            display_flip_y=bool(options.get("flip_y", False)),
            pixel_size_x_um=float(options["pixel_size_x_um"]),
            pixel_size_y_um=float(options["pixel_size_y_um"]),
        )

    @staticmethod
    def _parse_startup_reference_check(payload: Any, label: str) -> dict:
        if not isinstance(payload, dict):
            raise ValueError(
                f"FRAP startup reference check {label!r} must be an object with image and region."
            )
        image = str(payload.get("image", "")).strip()
        region: dict | None = None
        region_raw = payload.get("region")
        if image:
            if not isinstance(region_raw, dict):
                raise ValueError(
                    f"FRAP startup reference check {label!r} requires region when image is configured."
                )
            missing_region_keys = [
                key for key in ("left", "top", "right", "bottom") if key not in region_raw
            ]
            if missing_region_keys:
                raise ValueError(
                    f"FRAP startup reference check {label!r} region is missing required fields: "
                    + ", ".join(missing_region_keys)
                )
            left = int(region_raw.get("left", 0))
            top = int(region_raw.get("top", 0))
            right = int(region_raw.get("right", 0))
            bottom = int(region_raw.get("bottom", 0))
            if right - left <= 0 or bottom - top <= 0:
                raise ValueError(
                    f"FRAP startup reference check {label!r} region must have positive width and height."
                )
            region = {"left": left, "top": top, "right": right, "bottom": bottom}
        return {"image": image, "region": region}

    @staticmethod
    def _validate_captured_frame(frame: np.ndarray, transform: _FrapCoordinateTransform) -> None:
        height, width = np.asarray(frame).shape[:2]
        if width != transform.display_width or height != transform.display_height:
            raise ValueError(
                "FRAP captured GUI image dimensions do not match the configured display region: "
                f"captured=({width}, {height}) "
                f"configured=({transform.display_width}, {transform.display_height})"
            )

    def _normalize_launch_command(self, launch_command: Any) -> list[str]:
        if launch_command is None:
            env_command = os.environ.get("FRAP_CELL_SENS_LAUNCH_COMMAND", "").strip()
            if not env_command:
                return []
            return [part for part in shlex.split(env_command, posix=False) if str(part).strip()]
        if isinstance(launch_command, str):
            return [part for part in shlex.split(launch_command, posix=False) if str(part).strip()]
        if isinstance(launch_command, (list, tuple)):
            return [str(part).strip() for part in launch_command if str(part).strip()]
        raise ValueError("launch_command must be a string, list, tuple, or empty")

    def _ensure_prepared_once(self) -> None:
        if bool(getattr(self, "_closed", True)):
            raise RuntimeError("FRAP tool has been closed and cannot be reused.")

        if self._session_prepared:
            try:
                self._window_info = self._wait_for_window(self._profile["window_title_keyword"], timeout_sec=1.0)
                self._activate_window_if_needed(self._profile, window_info=self._window_info)
                return
            except RuntimeError:
                self._session_prepared = False
                self._window_info = {}

        self._window_info = self._ensure_window(self._profile)
        self._activate_window_if_needed(self._profile, window_info=self._window_info)
        self._window_info = self._wait_for_window(self._profile["window_title_keyword"])
        self._wait_for_startup_readiness("pre_click")
        self._prepare_frap_console(self._profile, self._window_info)
        self._verify_startup_reference("post_click")
        self._session_prepared = True

    def _ensure_window(self, profile: dict) -> dict:
        keyword = profile["window_title_keyword"]
        try:
            return self._wait_for_window(keyword)
        except RuntimeError:
            self._launch_cell_sens()
            return self._wait_for_window(
                keyword,
                timeout_sec=float(profile["options"]["startup_window_timeout_sec"]),
            )

    def _launch_cell_sens(self) -> None:
        launch_command = list(self._launch_command)
        if not launch_command:
            raise RuntimeError(
                "cellSens window not found and no launch_command was configured. "
                "Set launch_command in Frap.__init__ or FRAP_CELL_SENS_LAUNCH_COMMAND."
            )
        cwd = self._launch_workdir or None
        subprocess.Popen(
            launch_command,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            shell=False,
        )

    def _prepare_frap_console(self, profile: dict, window_info: dict) -> None:
        self._run_control_sequence(
            profile=profile,
            control_names=("bottom_frap_tab_button",),
        )

    def _wait_for_startup_readiness(self, check_name: str) -> None:
        options = self._profile["options"]
        ready_settle_seconds = float(options.get("startup_ready_settle_seconds", 15.0))
        timeout_sec = float(options.get("startup_window_timeout_sec", 60.0))
        poll_interval_sec = float(options.get("startup_poll_interval_sec", 1.0))
        deadline = time.time() + max(timeout_sec, 0.1)
        while True:
            try:
                self._verify_startup_reference(check_name)
                if ready_settle_seconds > 0:
                    time.sleep(ready_settle_seconds)
                return
            except RuntimeError:
                if time.time() >= deadline:
                    raise
                time.sleep(poll_interval_sec)

    def _verify_state_reference(self, check_name: str, state_name: str) -> None:
        options = self._profile["options"]
        check = options.get(check_name, {})
        reference_path_text = str(check.get("image", "")).strip()
        if not reference_path_text:
            logger.warning(
                "FRAP %s reference image is not configured; skipping this state check.",
                state_name,
            )
            return

        region = check.get("region")
        if not isinstance(region, dict):
            raise RuntimeError(f"FRAP {state_name} state check region is invalid.")

        reference_path = Path(reference_path_text).expanduser()
        if not reference_path.is_absolute():
            reference_path = Path(self._profile_path).parent / reference_path
        reference_path = reference_path.resolve()
        cv2 = _import_cv2()
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise RuntimeError(f"FRAP {state_name} reference image could not be read: {reference_path}")

        capture = self._capture_screen_region(region)
        capture_gray = cv2.cvtColor(capture, cv2.COLOR_RGB2GRAY) if capture.ndim == 3 else capture
        if capture_gray.shape != reference.shape:
            raise RuntimeError(
                f"FRAP {state_name} reference image size does not match its region: "
                f"reference=({reference.shape[1]}x{reference.shape[0]}) "
                f"region=({capture_gray.shape[1]}x{capture_gray.shape[0]})."
            )

        match = cv2.matchTemplate(capture_gray, reference, cv2.TM_CCOEFF_NORMED)
        score = float(np.max(match))
        threshold = float(options.get("startup_match_threshold", 0.8))
        if math.isfinite(score) and score >= threshold:
            logger.info(
                "FRAP %s visual verification passed. score=%.4f threshold=%.4f reference=%s",
                state_name,
                score,
                threshold,
                reference_path,
            )
            return

        evidence_path = self._save_startup_evidence(f"{state_name}", capture)
        raise RuntimeError(
            f"FRAP {state_name} visual verification failed: similarity score "
            f"{score:.4f} is below threshold {threshold:.4f}. Evidence screenshot saved to: {evidence_path}"
        )

    def _wait_for_state_reference(self, check_name: str, state_name: str) -> None:
        options = self._profile["options"]
        timeout_sec = float(options.get("startup_window_timeout_sec", 60.0))
        poll_interval_sec = float(options.get("startup_poll_interval_sec", 1.0))
        deadline = time.time() + max(timeout_sec, 0.1)
        while True:
            try:
                self._verify_state_reference(check_name, state_name)
                return
            except RuntimeError:
                if time.time() >= deadline:
                    raise
                time.sleep(poll_interval_sec)

    def _verify_startup_reference(self, check_name: str) -> None:
        options = self._profile["options"]
        check = options["startup_reference_checks"][check_name]
        reference_path_text = str(check["image"]).strip()
        if not reference_path_text:
            logger.warning(
                "FRAP startup %s reference image is not configured; skipping this startup check.",
                check_name,
            )
            return

        reference_path = Path(reference_path_text).expanduser()
        if not reference_path.is_absolute():
            reference_path = Path(self._profile_path).parent / reference_path
        reference_path = reference_path.resolve()
        cv2 = _import_cv2()
        reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise RuntimeError(
                f"FRAP startup {check_name} reference image could not be read: {reference_path}"
            )

        capture = self._capture_screen_region(check["region"])
        capture_gray = cv2.cvtColor(capture, cv2.COLOR_RGB2GRAY) if capture.ndim == 3 else capture
        if capture_gray.shape != reference.shape:
            raise RuntimeError(
                f"FRAP startup {check_name} reference image size does not match its region: "
                f"reference=({reference.shape[1]}x{reference.shape[0]}) "
                f"region=({capture_gray.shape[1]}x{capture_gray.shape[0]}). "
                "Recapture the reference image using the configured region."
            )

        match = cv2.matchTemplate(capture_gray, reference, cv2.TM_CCOEFF_NORMED)
        score = float(np.max(match))
        threshold = float(options["startup_match_threshold"])
        if math.isfinite(score) and score >= threshold:
            logger.info(
                "FRAP startup %s visual verification passed. score=%.4f threshold=%.4f reference=%s",
                check_name,
                score,
                threshold,
                reference_path,
            )
            return

        failure_hints = {
            "pre_click": "cellSens does not appear fully loaded; refusing to click the FRAP tab.",
            "post_click": (
                "cellSens may not be fully loaded, or the FRAP tab click did not open the FRAP console."
            ),
        }
        evidence_path = self._save_startup_evidence(check_name, capture)
        raise RuntimeError(
            f"FRAP startup {check_name} visual verification failed: similarity score "
            f"{score:.4f} is below threshold {threshold:.4f}. cellSens may not be fully "
            f"loaded. {failure_hints[check_name]} "
            f"Evidence screenshot saved to: {evidence_path}"
        )

    def _save_startup_evidence(self, check_name: str, capture: np.ndarray) -> Path:
        output_dir = Path(__file__).resolve().parents[1] / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        evidence_path = (
            output_dir / f"frap_startup_mismatch_{check_name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        )
        saved = _import_cv2().imwrite(str(evidence_path), capture)
        if not saved:
            raise RuntimeError(f"Failed to write FRAP startup evidence screenshot: {evidence_path}")
        return evidence_path

    def _capture_screen_region(self, region: dict) -> np.ndarray:
        left = int(region["left"])
        top = int(region["top"])
        width = int(region["right"]) - left + 1
        height = int(region["bottom"]) - top + 1
        if width <= 0 or height <= 0:
            raise ValueError("FRAP capture region must have positive width and height.")

        pyautogui = _import_pyautogui()
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        frame = np.asarray(screenshot)
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] >= 3:
            return frame[:, :, :3]
        raise RuntimeError(f"Unsupported screenshot shape for FRAP detection: {frame.shape}")

    def _capture_image_region(self, profile: dict) -> np.ndarray:
        return self._capture_screen_region(profile["image_region"])

    def _get_cellpose(self):
        if self._cellpose is None:
            cellpose_class = _import_cellpose_2d()
            self._cellpose = cellpose_class(self._storage_manager, self._output_dir)
            self._cellpose.cellpose_initialize(
                model_type=self._cellpose_model_type,
                device=self._cellpose_device,
            )
        return self._cellpose

    def _segment_and_analyze_cells(self, frame: np.ndarray) -> dict[str, Any] | None:
        image = np.asarray(frame)
        if image.ndim not in (2, 3):
            raise ValueError(f"Unsupported FRAP frame shape: {image.shape}")

        options = self._profile["options"]
        diameter = options.get("cellpose_diameter")
        masks = self._get_cellpose().segment(
            image,
            diameter=float(diameter) if diameter is not None else None,
            tile_size=max(image.shape[:2]),
            normalize={
                "lowhigh": None,
                "percentile": [1.0, 99.0],
                "normalize": True,
                "norm3D": True,
                "sharpen_radius": 0.0,
                "smooth_radius": 0.0,
                "tile_norm_blocksize": 0.0,
                "tile_norm_smooth3D": 0.0,
                "invert": False,
            },
        )
        return self._analyze_cellpose_masks(
            masks,
            image,
            fluorescence_channel=int(options.get("cellpose_channel", 1)),
            intensity_percentile=float(options.get("cellpose_area_percentile", 60.0)),
            min_area_px=float(options.get("cellpose_min_area_px", 3000.0)),
            distance_weight=float(options.get("cellpose_distance_weight", 0.5)),
        )

    @staticmethod
    def _analyze_cellpose_masks(
        masks: np.ndarray,
        frame: np.ndarray | None = None,
        *,
        fluorescence_channel: int = 1,
        intensity_percentile: float = 60.0,
        min_area_px: float | None = 3000.0,
        distance_weight: float = 0.5,
    ) -> dict[str, Any] | None:
        if not 0.0 <= float(intensity_percentile) <= 100.0:
            raise ValueError("Cellpose area intensity percentile must be between 0 and 100.")
        if min_area_px is not None and float(min_area_px) < 0.0:
            raise ValueError("Cellpose minimum area must be non-negative.")
        if not 0.0 <= float(distance_weight) <= 1.0:
            raise ValueError("Cellpose distance weight must be between 0 and 1.")
        cv2 = _import_cv2()
        mask_array = np.squeeze(np.asarray(masks))
        if mask_array.ndim != 2:
            raise ValueError(f"Cellpose masks must be 2D, got shape {mask_array.shape}")

        height, width = mask_array.shape
        center_x = (float(width) - 1.0) / 2.0
        center_y = (float(height) - 1.0) / 2.0
        min_area = max(64.0, float(width * height) * 0.00002)
        if min_area_px is not None:
            min_area = max(min_area, float(min_area_px))
        max_area = max(min_area + 1.0, float(width * height) * 0.25)
        edge_margin = max(5, int(round(min(width, height) * 0.01)))
        intensity = None
        if frame is not None:
            frame_array = np.asarray(frame)
            if frame_array.ndim == 2:
                intensity = frame_array.astype(np.float32, copy=False)
            elif frame_array.ndim == 3 and 0 <= fluorescence_channel < frame_array.shape[2]:
                intensity = frame_array[:, :, fluorescence_channel].astype(np.float32, copy=False)
            else:
                raise ValueError(f"Unsupported FRAP fluorescence frame shape: {frame_array.shape}")
        candidates: list[dict[str, Any]] = []
        labels = np.unique(mask_array)
        for label in labels[labels > 0]:
            instance_mask = (mask_array == label).astype(np.uint8)
            area = float(np.count_nonzero(instance_mask))
            if area < float(min_area) or area > float(max_area):
                continue

            contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            touches_edge = (
                x <= edge_margin
                or y <= edge_margin
                or x + w >= width - edge_margin
                or y + h >= height - edge_margin
            )
            if touches_edge:
                continue

            refined_mask = instance_mask
            if intensity is not None:
                instance_values = intensity[instance_mask > 0]
                threshold = float(np.percentile(instance_values, intensity_percentile))
                bright_mask = ((instance_mask > 0) & (intensity >= threshold)).astype(np.uint8)
                bright_mask = cv2.morphologyEx(
                    bright_mask,
                    cv2.MORPH_OPEN,
                    np.ones((5, 5), np.uint8),
                    iterations=1,
                )
                bright_mask = cv2.morphologyEx(
                    bright_mask,
                    cv2.MORPH_CLOSE,
                    np.ones((3, 3), np.uint8),
                    iterations=1,
                )
                bright_contours, _ = cv2.findContours(
                    bright_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if bright_contours:
                    largest = max(bright_contours, key=cv2.contourArea)
                    largest = cv2.convexHull(largest)
                    refined_mask = np.zeros_like(instance_mask)
                    cv2.drawContours(refined_mask, [largest], -1, 1, thickness=-1)

            refined_area = float(np.count_nonzero(refined_mask))
            if refined_area < float(min_area):
                continue
            refined_contours, _ = cv2.findContours(
                refined_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not refined_contours:
                continue
            contour = max(refined_contours, key=cv2.contourArea)
            refined_x, refined_y, refined_w, refined_h = cv2.boundingRect(contour)
            ys, xs = np.nonzero(refined_mask)
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())
            distance = float(np.hypot(centroid_x - center_x, centroid_y - center_y))
            fluorescence = (
                float(np.sum(intensity[refined_mask > 0]))
                if intensity is not None
                else float(refined_area)
            )
            contour_points = contour.reshape(-1, 2).astype(int)
            candidates.append(
                {
                    "label": int(label),
                    "distance_to_center_px": distance,
                    "area_px": refined_area,
                    "cellpose_area_px": float(area),
                    "fluorescence_signal": fluorescence,
                    "bbox_px": {
                        "left": int(refined_x),
                        "top": int(refined_y),
                        "width": int(refined_w),
                        "height": int(refined_h),
                    },
                    "center_px": {
                        "x": float(centroid_x),
                        "y": float(centroid_y),
                    },
                    "center_offset_px": {
                        "x": float(centroid_x - center_x),
                        "y": float(centroid_y - center_y),
                    },
                    "touches_edge": bool(touches_edge),
                    "contour": contour_points,
                }
            )

        if not candidates:
            return None

        max_distance = float(np.hypot(center_x, center_y)) or 1.0
        max_fluorescence = max(float(item["fluorescence_signal"]) for item in candidates) or 1.0
        for item in candidates:
            distance_score = max(
                0.0,
                1.0 - float(item["distance_to_center_px"]) / max_distance,
            )
            fluorescence_score = float(item["fluorescence_signal"]) / max_fluorescence
            item["distance_score"] = distance_score
            item["fluorescence_score"] = fluorescence_score
            item["score"] = (
                float(distance_weight) * distance_score
                + (1.0 - float(distance_weight)) * fluorescence_score
            )

        candidates.sort(key=lambda item: (float(item["score"]), float(item["area_px"])), reverse=True)
        return {
            "frame": {
                "width": int(width),
                "height": int(height),
                "center_x": float(center_x),
                "center_y": float(center_y),
            },
            "candidates": candidates,
            "candidate_count": len(candidates),
        }

    def _close_window(self, window_info: dict, profile: dict | None = None) -> None:
        window_width = int(window_info["width"])
        window_height = int(window_info["height"])
        if window_width <= 0 or window_height <= 0:
            return
        settle_seconds = 2.0
        close_timeout_sec = 15.0
        process_timeout_sec = 30.0
        if profile is not None:
            settle_seconds = float(profile.get("options", {}).get("close_settle_seconds", settle_seconds))
            close_timeout_sec = float(profile.get("options", {}).get("close_window_timeout_sec", close_timeout_sec))
            process_timeout_sec = float(profile.get("options", {}).get("close_process_timeout_sec", process_timeout_sec))
            close_control = profile.get("controls", {}).get("cellsens_close_button")
            if isinstance(close_control, dict) and str(close_control.get("type", "")).strip().lower() == "point":
                try:
                    self._click_screen_absolute(
                        absolute_x=int(close_control["x"]),
                        absolute_y=int(close_control["y"]),
                        move_duration_sec=float(profile["options"].get("move_duration_sec", 0.0)),
                        button=str(close_control.get("button", "left")),
                    )
                    if settle_seconds > 0:
                        time.sleep(settle_seconds)
                    try:
                        self._wait_for_window(profile["window_title_keyword"], timeout_sec=max(close_timeout_sec, 0.1))
                    except RuntimeError:
                        self._wait_for_process_exit(process_name="SisXV.exe", timeout_sec=process_timeout_sec)
                        return
                except Exception:
                    logger.warning("FRAP GUI close button click failed; falling back to Alt+F4.", exc_info=True)
        pyautogui = _import_pyautogui()
        original_pause = getattr(pyautogui, "PAUSE", 0.0)
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.hotkey("alt", "f4", interval=0.0)
        finally:
            pyautogui.PAUSE = original_pause
        self._wait_for_process_exit(process_name="SisXV.exe", timeout_sec=process_timeout_sec)

    def _wait_for_process_exit(self, process_name: str, timeout_sec: float = 30.0, poll_interval_sec: float = 0.5) -> None:
        deadline = time.time() + float(timeout_sec)
        while True:
            if not _find_processes_by_name(process_name):
                return
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Process still running after {timeout_sec:.2f}s: {process_name}"
                )
            time.sleep(float(poll_interval_sec))

    def _wait_for_window(self, title_keyword: str, timeout_sec: float = 15.0, poll_interval_sec: float = 0.2) -> dict:
        deadline = time.time() + float(timeout_sec)
        while True:
            windows = _import_pygetwindow().getWindowsWithTitle(str(title_keyword))
            visible = [
                window for window in windows
                if getattr(window, "width", 0) > 0 and getattr(window, "height", 0) > 0
            ]
            minimized = [
                window for window in visible
                if bool(getattr(window, "isMinimized", False))
            ]
            if minimized:
                for window in minimized:
                    try:
                        window.restore()
                    except Exception:
                        pass
                time.sleep(0.2)
                windows = _import_pygetwindow().getWindowsWithTitle(str(title_keyword))
                visible = [
                    window for window in windows
                    if getattr(window, "width", 0) > 0 and getattr(window, "height", 0) > 0
                ]
            if visible:
                visible.sort(
                    key=lambda window: (
                        not bool(getattr(window, "isMinimized", False)),
                        int(getattr(window, "width", 0)) * int(getattr(window, "height", 0)),
                        bool(getattr(window, "isActive", False)),
                    ),
                    reverse=True,
                )
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

    def _activate_window_if_needed(self, profile: dict, window_info: dict | None = None) -> None:
        if not bool(profile["options"].get("activate_before_action", True)):
            return
        if window_info is None:
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

    def _run_control_sequence(
        self,
        *,
        profile: dict,
        control_names: tuple[str, ...],
    ) -> None:
        for control_name in control_names:
            action = self._get_point_control(profile["controls"], control_name)
            self._click_screen_absolute(
                absolute_x=int(action["x"]),
                absolute_y=int(action["y"]),
                move_duration_sec=float(profile["options"]["move_duration_sec"]),
                button=str(action.get("button", "left")),
                clicks=int(action.get("clicks", 1)),
            )
            if float(profile["options"]["click_interval_sec"]) > 0:
                time.sleep(float(profile["options"]["click_interval_sec"]))

    @staticmethod
    def _get_point_control(controls: dict, control_name: str) -> dict:
        if control_name not in controls:
            raise ValueError(f"FRAP profile control not found: {control_name}")
        action = dict(controls[control_name])
        action_type = str(action.get("type", "")).strip().lower()
        if action_type != "point":
            raise ValueError(f"FRAP control must be a point action: {control_name}")
        return action

    def _click_screen_absolute(
        self,
        *,
        absolute_x: int,
        absolute_y: int,
        move_duration_sec: float,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        pyautogui = _import_pyautogui()
        screen_width, screen_height = pyautogui.size()
        if not (0 <= int(absolute_x) < int(screen_width) and 0 <= int(absolute_y) < int(screen_height)):
            raise ValueError(
                f"Absolute click position is outside the screen bounds: position=({absolute_x}, {absolute_y}) "
                f"screen=({screen_width}, {screen_height})"
            )
        original_pause = getattr(pyautogui, "PAUSE", 0.0)
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.moveTo(int(absolute_x), int(absolute_y), duration=max(float(move_duration_sec), 0.0))
            time.sleep(0.1)
            for _ in range(max(int(clicks), 1)):
                pyautogui.mouseDown(int(absolute_x), int(absolute_y), button=str(button))
                time.sleep(0.05)
                pyautogui.mouseUp(int(absolute_x), int(absolute_y), button=str(button))
                time.sleep(0.05)
        finally:
            pyautogui.PAUSE = original_pause

