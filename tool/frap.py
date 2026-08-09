from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


def _import_cv2():
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("cv2 is required for FRAP cell detection and contour extraction.") from exc
    return cv2


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
    stage_axis_sign_x: int
    stage_axis_sign_y: int
    center_x_um: float
    center_y_um: float
    pixel_size_x_um: float
    pixel_size_y_um: float

    def __post_init__(self) -> None:
        if self.source_width <= 1 or self.source_height <= 1:
            raise ValueError("FRAP source image dimensions must be greater than one pixel.")
        if self.display_right <= self.display_left or self.display_bottom <= self.display_top:
            raise ValueError("FRAP display region must have positive width and height.")
        if self.stage_axis_sign_x not in {-1, 1} or self.stage_axis_sign_y not in {-1, 1}:
            raise ValueError("FRAP stage axis signs must each be either -1 or 1.")
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

    def source_to_stage(self, x_px: float, y_px: float) -> tuple[float, float]:
        self._validate_source_point(x_px, y_px)
        center_x_px = float(self.source_width - 1) / 2.0
        center_y_px = float(self.source_height - 1) / 2.0
        return (
            self.center_x_um
            + self.stage_axis_sign_x * (float(x_px) - center_x_px) * self.pixel_size_x_um,
            self.center_y_um
            + self.stage_axis_sign_y * (float(y_px) - center_y_px) * self.pixel_size_y_um,
        )

    def stage_to_source(self, x_um: float, y_um: float) -> tuple[float, float]:
        center_x_px = float(self.source_width - 1) / 2.0
        center_y_px = float(self.source_height - 1) / 2.0
        source_x = center_x_px + self.stage_axis_sign_x * (
            float(x_um) - self.center_x_um
        ) / self.pixel_size_x_um
        source_y = center_y_px + self.stage_axis_sign_y * (
            float(y_um) - self.center_y_um
        ) / self.pixel_size_y_um
        self._validate_source_point(source_x, source_y, coordinate_name="stage target")
        return source_x, source_y

    def display_to_stage(self, x_px: float, y_px: float) -> tuple[float, float]:
        source_x, source_y = self.display_to_source(x_px, y_px)
        return self.source_to_stage(source_x, source_y)

    def stage_to_screen(self, x_um: float, y_um: float) -> tuple[int, int]:
        source_x, source_y = self.stage_to_source(x_um, y_um)
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
    """cellSens FRAP GUI helper."""

    _active_instance: Frap | None = None

    planning_hint = (
        "Use this tool for cellSens FRAP control through the FRAP panel: "
        "open the FRAP tab first, then use laser_on, laser_position, laser_off, "
        "cell_detection, and cell_contour_extraction."
    )
    execution_hint = (
        "Instantiate the tool to ensure cellSens is available and focused. "
        "After opening cellSens, select the FRAP tab first. laser_on clicks the "
        "FRAP start button, laser_position performs a single click inside the live "
        "field-of-view region, laser_off closes cellSens, and cell_detection / "
        "cell_contour_extraction analyze the current field image for a usable target."
    )

    def __init__(
        self,
        storage_manager=None,
        output_dir: str = "./output",
        launch_command: str | list[str] | None = None,
        launch_workdir: str = "",
    ) -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir
        self._default_profile_filename = "frap_ui_profile.json"
        self._laser_enabled = False
        self._launch_command = self._normalize_launch_command(launch_command)
        self._launch_workdir = str(launch_workdir).strip()
        self._profile = self._load_profile()
        self._window_info = self._ensure_window(self._profile)
        self._activate_window_if_needed(self._profile)
        self._prepare_frap_console(self._profile, self._window_info)
        type(self)._active_instance = self

    @tool_func
    def laser_on(self, power: float, duration: float) -> None:
        """
        Open FRAP mode in cellSens.

        Args:
            power: Retained for API compatibility; validated but not applied directly.
            duration: Retained for API compatibility; validated but not applied directly.
        """
        if not (0.0 <= float(power) <= 100.0):
            raise ValueError("power must be between 0.0 and 100.0")
        if float(duration) <= 0:
            raise ValueError("duration must be positive")

        self._window_info = self._ensure_window(self._profile)
        self._activate_window_if_needed(self._profile)
        self._prepare_frap_console(self._profile, self._window_info)
        self._run_control_sequence(
            profile=self._profile,
            window_info=self._window_info,
            control_names=("frap_start_button",),
            fallback_names=(("frap_start_button", "laser_on", "start", "FRAP"),),
        )
        self._laser_enabled = True

    @tool_func
    def laser_off(self) -> None:
        """Close cellSens."""
        self._window_info = self._ensure_window(self._profile)
        self._activate_window_if_needed(self._profile)
        self._laser_enabled = False
        self._close_window(self._window_info)

    @tool_func
    @staticmethod
    def laser_position(x: int, y: int) -> None:
        """
        Set laser focal point coordinates.

        Args:
            x: Absolute X-axis stage position in microns.
            y: Absolute Y-axis stage position in microns.
        """
        instance = Frap._require_active_instance()
        instance._laser_position_impl(x, y)

    def _laser_position_impl(self, x: int, y: int) -> None:
        if not self._laser_enabled:
            raise RuntimeError("laser_position requires FRAP to be started first.")

        self._window_info = self._ensure_window(self._profile)
        self._activate_window_if_needed(self._profile)

        options = self._profile["options"]
        transform = self._build_coordinate_transform()
        target_x, target_y = transform.stage_to_screen(float(x), float(y))
        self._click_screen_absolute(
            self._window_info,
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
        Detect and return the position of the target cell.

        Returns:
            Dictionary containing absolute stage ``x`` and ``y`` coordinates in
            microns. Returns an empty dictionary when no usable cell is detected.
        """
        instance = Frap._require_active_instance()
        return instance._cell_detection_impl()

    def _cell_detection_impl(self) -> dict:
        self._window_info = self._ensure_window(self._profile)
        self._activate_window_if_needed(self._profile)
        frame = self._capture_image_region(self._profile)
        analysis = self._analyze_cell_candidates(frame)
        if not analysis:
            return {}
        transform = self._build_coordinate_transform()
        self._validate_captured_frame(frame, transform)
        center_px = analysis["best_candidate"]["center_px"]
        x_um, y_um = transform.display_to_stage(
            float(center_px["x"]),
            float(center_px["y"]),
        )
        return {"x": x_um, "y": y_um}

    @tool_func
    @staticmethod
    def cell_contour_extraction() -> dict:
        """
        Extract the target cell membrane contour from the current image.

        Returns:
            Dictionary containing absolute stage contour ``points`` in microns,
            ``area`` in square microns, and ``perimeter`` in microns. Returns an
            empty dictionary when no usable cell is detected.
        """
        instance = Frap._require_active_instance()
        return instance._cell_contour_extraction_impl()

    def _cell_contour_extraction_impl(self) -> dict:
        self._window_info = self._ensure_window(self._profile)
        self._activate_window_if_needed(self._profile)
        frame = self._capture_image_region(self._profile)
        analysis = self._analyze_cell_candidates(frame)
        if not analysis:
            return {}

        transform = self._build_coordinate_transform()
        self._validate_captured_frame(frame, transform)
        candidate = analysis["best_candidate"]
        contour_px = np.asarray(candidate["contour"], dtype=float).reshape(-1, 2)
        points = [
            transform.display_to_stage(float(point[0]), float(point[1]))
            for point in contour_px
        ]
        return {
            "points": points,
            "area": self._polygon_area_um2(points),
            "perimeter": self._contour_perimeter_um(points),
        }

    @classmethod
    def _require_active_instance(cls) -> Frap:
        instance = cls._active_instance
        if instance is None:
            raise RuntimeError("Frap must be instantiated before using this static method.")
        return instance

    def _load_profile(self) -> dict:
        path = Path(self.output_dir, self._default_profile_filename).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Default FRAP UI profile not found: {path}")

        payload = json.loads(path.read_text(encoding="utf-8"))
        image_region = payload.get("image_region", {})
        options = payload.get("options", {})
        if not isinstance(image_region, dict) or not isinstance(options, dict):
            raise ValueError("Invalid FRAP profile structure")

        required_region_keys = ("left", "top", "right", "bottom", "source_width", "source_height")
        missing_region_keys = [key for key in required_region_keys if key not in image_region]
        if missing_region_keys:
            raise ValueError(
                "FRAP profile image_region is missing required fields: "
                + ", ".join(missing_region_keys)
            )

        return {
            "window_title_keyword": str(payload.get("window_title_keyword", "")).strip(),
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
                "click_interval_sec": float(options.get("click_interval_sec", 0.15)),
                "move_duration_sec": float(options.get("move_duration_sec", 0.0)),
                "flip_x": bool(options.get("flip_x", False)),
                "flip_y": bool(options.get("flip_y", False)),
                "stage_axis_sign_x": options.get("stage_axis_sign_x"),
                "stage_axis_sign_y": options.get("stage_axis_sign_y"),
            },
        }

    def _build_coordinate_transform(self) -> _FrapCoordinateTransform:
        region = self._profile["image_region"]
        options = self._profile["options"]
        stage_axis_sign_x = self._parse_stage_axis_sign(options.get("stage_axis_sign_x"), "x")
        stage_axis_sign_y = self._parse_stage_axis_sign(options.get("stage_axis_sign_y"), "y")
        image_path = self._resolve_latest_ome_image_path()
        metadata = load_ome_spatial_metadata(
            image_path,
            require_stage_positions=True,
            require_pixel_sizes=True,
        )
        source_width = int(region["source_width"])
        source_height = int(region["source_height"])
        metadata_width = int(metadata.get("image_width_px", 0))
        metadata_height = int(metadata.get("image_height_px", 0))
        if metadata_width != source_width or metadata_height != source_height:
            raise ValueError(
                "FRAP source image dimensions do not match the latest OME image: "
                f"profile=({source_width}, {source_height}) "
                f"OME=({metadata_width}, {metadata_height}) path={image_path}"
            )
        return _FrapCoordinateTransform(
            source_width=source_width,
            source_height=source_height,
            display_left=int(region["left"]),
            display_top=int(region["top"]),
            display_right=int(region["right"]),
            display_bottom=int(region["bottom"]),
            display_flip_x=bool(options.get("flip_x", False)),
            display_flip_y=bool(options.get("flip_y", False)),
            stage_axis_sign_x=stage_axis_sign_x,
            stage_axis_sign_y=stage_axis_sign_y,
            center_x_um=float(metadata["center_x_um"]),
            center_y_um=float(metadata["center_y_um"]),
            pixel_size_x_um=float(metadata["pixel_size_x_um"]),
            pixel_size_y_um=float(metadata["pixel_size_y_um"]),
        )

    @staticmethod
    def _parse_stage_axis_sign(raw_value: Any, axis_name: str) -> int:
        if raw_value is None:
            raise RuntimeError(
                f"FRAP stage_axis_sign_{axis_name} is not calibrated. "
                "Set it to 1 or -1 in frap_ui_profile.json before using physical coordinates."
            )
        numeric_value = float(raw_value)
        if numeric_value not in {-1.0, 1.0}:
            raise ValueError(f"FRAP stage_axis_sign_{axis_name} must be either 1 or -1.")
        return int(numeric_value)

    def _resolve_latest_ome_image_path(self) -> Path:
        registered_candidates: list[Path] = []
        if self.storage_manager is not None and hasattr(self.storage_manager, "read_log"):
            try:
                registered = self.storage_manager.read_log(include_temp=True)
            except Exception:
                registered = {}
            if isinstance(registered, dict):
                for metadata in registered.values():
                    if not isinstance(metadata, dict):
                        continue
                    if metadata.get("created_by") != "microscope" or metadata.get("file_type") != "ome-tiff":
                        continue
                    filename = str(metadata.get("filename", "") or "").strip()
                    candidate = Path(self.output_dir, filename).expanduser().resolve()
                    if self._is_ome_tiff(candidate) and candidate.is_file():
                        registered_candidates.append(candidate)
        if registered_candidates:
            return max(set(registered_candidates), key=lambda path: path.stat().st_mtime)

        output_root = Path(self.output_dir).expanduser().resolve()
        filesystem_candidates = [
            path.resolve()
            for path in output_root.rglob("*")
            if path.is_file() and self._is_ome_tiff(path)
        ] if output_root.exists() else []
        if filesystem_candidates:
            return max(set(filesystem_candidates), key=lambda path: path.stat().st_mtime)
        raise FileNotFoundError(
            "FRAP coordinate conversion requires a current microscope OME-TIFF in the output directory."
        )

    @staticmethod
    def _is_ome_tiff(path: Path) -> bool:
        lowered_name = path.name.lower()
        return lowered_name.endswith(".ome.tif") or lowered_name.endswith(".ome.tiff")

    @staticmethod
    def _validate_captured_frame(frame: np.ndarray, transform: _FrapCoordinateTransform) -> None:
        height, width = np.asarray(frame).shape[:2]
        if width != transform.display_width or height != transform.display_height:
            raise ValueError(
                "FRAP captured GUI image dimensions do not match the configured display region: "
                f"captured=({width}, {height}) "
                f"configured=({transform.display_width}, {transform.display_height})"
            )

    @staticmethod
    def _polygon_area_um2(points: list[tuple[float, float]]) -> float:
        if len(points) < 3:
            return 0.0
        coordinates = np.asarray(points, dtype=float)
        x_values = coordinates[:, 0]
        y_values = coordinates[:, 1]
        return float(
            0.5
            * abs(
                np.dot(x_values, np.roll(y_values, -1))
                - np.dot(y_values, np.roll(x_values, -1))
            )
        )

    @staticmethod
    def _contour_perimeter_um(points: list[tuple[float, float]]) -> float:
        if len(points) < 2:
            return 0.0
        coordinates = np.asarray(points, dtype=float)
        closed = np.vstack((coordinates, coordinates[0]))
        segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        return float(segment_lengths.sum())

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

    def _ensure_window(self, profile: dict) -> dict:
        keyword = profile["window_title_keyword"]
        try:
            return self._wait_for_window(keyword)
        except RuntimeError:
            self._launch_cell_sens()
            return self._wait_for_window(keyword, timeout_sec=60.0)

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
            window_info=window_info,
            control_names=("bottom_frap_tab_button",),
            fallback_names=(("bottom_frap_tab_button", "frap_tab_button"),),
        )

    def _capture_image_region(self, profile: dict) -> np.ndarray:
        region = profile["image_region"]
        left = int(region["left"])
        top = int(region["top"])
        width = int(region["right"]) - left + 1
        height = int(region["bottom"]) - top + 1
        if width <= 0 or height <= 0:
            raise ValueError("FRAP image_region must have positive width and height.")

        pyautogui = _import_pyautogui()
        screenshot = pyautogui.screenshot(region=(left, top, width, height))
        frame = np.asarray(screenshot)
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] >= 3:
            return frame[:, :, :3]
        raise RuntimeError(f"Unsupported screenshot shape for FRAP detection: {frame.shape}")

    def _analyze_cell_candidates(self, frame: np.ndarray) -> dict[str, Any] | None:
        cv2 = _import_cv2()
        image = np.asarray(frame)
        if image.ndim == 3 and image.shape[2] >= 3:
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
        elif image.ndim == 2:
            gray = image
        else:
            raise ValueError(f"Unsupported FRAP frame shape: {image.shape}")

        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
        height, width = blurred.shape[:2]
        center_x = (float(width) - 1.0) / 2.0
        center_y = (float(height) - 1.0) / 2.0
        min_area = max(12.0, float(width * height) * 0.00001)
        max_area = max(min_area + 1.0, float(width * height) * 0.02)
        edge_margin = max(5, int(round(min(width, height) * 0.01)))

        modes = (
            ("bright", cv2.THRESH_BINARY),
            ("dark", cv2.THRESH_BINARY_INV),
        )
        mode_candidates: list[dict[str, Any]] = []
        for mode_name, threshold_type in modes:
            _, mask = cv2.threshold(blurred, 0, 255, threshold_type + cv2.THRESH_OTSU)
            candidates = self._extract_contour_candidates(
                mask,
                cv2=cv2,
                center_x=center_x,
                center_y=center_y,
                min_area=min_area,
                max_area=max_area,
                edge_margin=edge_margin,
                mode_name=mode_name,
            )
            if candidates:
                mode_candidates.append(
                    {
                        "mode": mode_name,
                        "candidates": candidates,
                        "best_score": float(candidates[0]["score"]),
                    }
                )

        if not mode_candidates:
            return None

        mode_candidates.sort(key=lambda item: item["best_score"], reverse=True)
        best_mode = mode_candidates[0]
        best_candidate = dict(best_mode["candidates"][0])
        ranked_candidates = list(best_mode["candidates"][:10])
        return {
            "frame": {
                "width": int(width),
                "height": int(height),
                "center_x": float(center_x),
                "center_y": float(center_y),
            },
            "mode": str(best_mode["mode"]),
            "candidates": ranked_candidates,
            "best_candidate": best_candidate,
            "candidate_count": len(best_mode["candidates"]),
        }

    def _extract_contour_candidates(
        self,
        mask: np.ndarray,
        *,
        cv2: Any,
        center_x: float,
        center_y: float,
        min_area: float,
        max_area: float,
        edge_margin: int,
        mode_name: str,
    ) -> list[dict[str, Any]]:
        working_mask = np.asarray(mask).copy()
        kernel_open = np.ones((3, 3), np.uint8)
        kernel_close = np.ones((5, 5), np.uint8)
        working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        working_mask = cv2.morphologyEx(working_mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        contours, _ = cv2.findContours(working_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[dict[str, Any]] = []
        height, width = working_mask.shape[:2]

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(min_area) or area > float(max_area):
                continue

            x, y, w, h = cv2.boundingRect(contour)
            moments = cv2.moments(contour)
            if abs(float(moments.get("m00", 0.0))) < 1e-9:
                continue

            centroid_x = float(moments["m10"] / moments["m00"])
            centroid_y = float(moments["m01"] / moments["m00"])
            if not (0.0 <= centroid_x < float(width) and 0.0 <= centroid_y < float(height)):
                continue

            touches_edge = (
                x <= edge_margin
                or y <= edge_margin
                or x + w >= width - edge_margin
                or y + h >= height - edge_margin
            )
            if touches_edge:
                continue
            shape_ratio = float(min(w, h)) / float(max(w, h)) if max(w, h) > 0 else 0.0
            distance = float(np.hypot(centroid_x - center_x, centroid_y - center_y))
            score = (1.0 + area / 250.0) * (0.5 + 0.5 * shape_ratio) / (1.0 + distance / 25.0)

            contour_points = contour.reshape(-1, 2).astype(int)
            candidates.append(
                {
                    "mode": mode_name,
                    "score": float(score),
                    "area_px": float(area),
                    "bbox_px": {
                        "left": int(x),
                        "top": int(y),
                        "width": int(w),
                        "height": int(h),
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
                    "shape_ratio": float(shape_ratio),
                    "contour": contour_points,
                }
            )

        candidates.sort(key=lambda item: (float(item["score"]), float(item["area_px"])), reverse=True)
        return candidates

    def _close_window(self, window_info: dict) -> None:
        window_width = int(window_info["width"])
        window_height = int(window_info["height"])
        if window_width <= 0 or window_height <= 0:
            return
        pyautogui = _import_pyautogui()
        original_pause = getattr(pyautogui, "PAUSE", 0.0)
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.hotkey("alt", "f4", interval=0.0)
        finally:
            pyautogui.PAUSE = original_pause

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

    def _run_control_sequence(
        self,
        *,
        profile: dict,
        window_info: dict,
        control_names: tuple[str, ...],
        fallback_names: tuple[tuple[str, ...], ...],
    ) -> None:
        for index, control_name in enumerate(control_names):
            action = self._resolve_control_action(
                profile["controls"],
                control_name,
                fallback_names[index] if index < len(fallback_names) else (control_name,),
            )
            self._run_actions([action], profile, window_info)

    def _resolve_control_action(
        self,
        controls: dict,
        preferred_name: str,
        fallback_names: tuple[str, ...],
    ) -> dict:
        for name in (preferred_name, *fallback_names):
            if name in controls:
                return dict(controls[name])
        fallback_display = ", ".join(repr(name) for name in (preferred_name, *fallback_names))
        raise ValueError(f"FRAP profile does not define any of: {fallback_display}")

    def _run_actions(self, action_items: list[Any], profile: dict, window_info: dict) -> None:
        for item in action_items:
            action = self._resolve_action(item, profile)
            action_type = str(action.get("type", "")).strip().lower()

            if action_type == "point":
                self._click_screen_absolute(
                    window_info,
                    absolute_x=int(action["x"]),
                    absolute_y=int(action["y"]),
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

    def _click_screen_absolute(
        self,
        window_info: dict,
        *,
        absolute_x: int,
        absolute_y: int,
        move_duration_sec: float,
        button: str = "left",
        clicks: int = 1,
    ) -> None:
        window_left = int(window_info["left"])
        window_top = int(window_info["top"])
        window_width = int(window_info["width"])
        window_height = int(window_info["height"])
        if not (
            window_left <= int(absolute_x) < window_left + window_width
            and window_top <= int(absolute_y) < window_top + window_height
        ):
            raise ValueError(
                f"Absolute click position is outside the window bounds: position=({absolute_x}, {absolute_y}) "
                f"window=({window_left}, {window_top}, {window_width}, {window_height})"
            )
        pyautogui = _import_pyautogui()
        original_pause = getattr(pyautogui, "PAUSE", 0.0)
        try:
            pyautogui.PAUSE = 0.0
            pyautogui.moveTo(int(absolute_x), int(absolute_y), duration=max(float(move_duration_sec), 0.0))
            pyautogui.click(int(absolute_x), int(absolute_y), clicks=max(int(clicks), 1), button=str(button))
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
