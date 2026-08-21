from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tool.base import BaseTool, tool_func
from .common import ImagingData, _coerce_detection_image_to_2d



class MicroscopeController(BaseTool):
    """Pure-simulation microscope platform (no Micro-Manager MMCore).

    All hardware state is kept in memory and image acquisition synthesizes
    deterministic arrays. Axis ranges come from the runtime configuration, so a
    real-microscope calibration (e.g. focus Z = 4323/4100 um) is accepted as long
    as it falls inside the configured range.
    """

    planning_hint = (
        "Use for simulated microscope hardware control and image acquisition: "
        "stage XY/Z focus, objective/channel/exposure/brightness, Z-stack and "
        "multi-dimensional acquisition. No real hardware is required."
    )
    execution_hint = (
        "The microscope is a pure simulation: state is in-memory and coordinates "
        "inside the configured ranges are accepted. Set focus Z to the calibration "
        "value (e.g. 4323 in the raw material pool, 4100 in the microwell array) and "
        "stage XY to target microns directly."
    )

    preview_window_name = "micro live (simulation)"

    def __init__(
        self,
        config_path: str,
        app_dir: str,
        output_path: str,
        storagemanger,
        *,
        system_config: Any = None,
        detection_targets: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        del config_path, app_dir
        if system_config is None:
            from bootstrap.config import load_runtime_settings

            system_config = load_runtime_settings().system
        self.system_config = system_config
        self.microscope_mode = "mock"

        self._storagemanger = storagemanger
        self.output_directory = str(output_path)
        self.detection_targets = {
            str(target_name): dict(spec)
            for target_name, spec in (detection_targets or {}).items()
        }
        self.objectives = dict(getattr(system_config, "objectives", {}))
        self.channels = dict(getattr(system_config, "channels", {}))
        self.objective_labels = dict(getattr(system_config, "objective_labels", {}))
        self.dichroic_colors = dict(getattr(system_config, "dichroic_colors", {}))

        # Axis and parameter ranges (mock overlay widens Z so calibration values fit).
        self.Max_X_position = float(getattr(system_config, "Max_X_position", 100000.0))
        self.Min_X_position = float(getattr(system_config, "Min_X_position", 0.0))
        self.Max_Y_position = float(getattr(system_config, "Max_Y_position", 70000.0))
        self.Min_Y_position = float(getattr(system_config, "Min_Y_position", 0.0))
        self.Max_Z_position = float(getattr(system_config, "Max_Z_position", 10000.0))
        self.Min_Z_position = float(getattr(system_config, "Min_Z_position", 0.0))
        self.Max_brightness = int(getattr(system_config, "Max_brightness", 250))
        self.Min_brightness = int(getattr(system_config, "Min_brightness", 0))
        self.Max_exposure = float(getattr(system_config, "Max_exposure", 1000))
        self.Min_exposure = float(getattr(system_config, "Min_exposure", 0))

        transmitted_light = dict(getattr(system_config, "transmitted_light", {}) or {})
        self.brightness_device = str(transmitted_light.get("device") or "").strip()
        self.brightness_property = str(transmitted_light.get("intensity_property") or "").strip()

        # Current state
        self.current_channel = ""
        self.current_objective = ""
        self.current_X_position = 0.0
        self.current_Y_position = 0.0
        self.current_Z_position = 0.0
        self.current_brightness = 0
        self.current_exposure_time = 0.0
        self._user_brightness = 0
        self._pixel_size = 0.1625

        # Acquisition parameters
        self.acquisition_positions: List[Dict[str, Any]] = []
        self.acquisition_channels: List[Dict[str, Any]] = []
        self.z_stack_params: Optional[Dict[str, float]] = None
        self.time_lapse_params: Optional[Dict[str, float]] = None

        # Preview / progress
        self.preview_running = False
        self._latest_display_frame: Optional[np.ndarray] = None
        self._task_progress_listener: Optional[Any] = None
        self._last_task_progress: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()
        self._frame_counter = 0
        self._initialized = False
        self._shutdown = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @tool_func
    def initialize(self) -> bool:
        """Initialize the simulated microscope (state bookkeeping only)."""
        os.makedirs(self.output_directory, exist_ok=True)
        with self._lock:
            self._initialized = True
            self._shutdown = False
        return True

    @tool_func
    def shutdown(self) -> None:
        """Release the simulated microscope and stop preview."""
        with self._lock:
            self.preview_running = False
            self._shutdown = True
            self._initialized = False

    # ------------------------------------------------------------------
    # Position control
    # ------------------------------------------------------------------
    @tool_func
    def set_x_y_position(self, x: float, y: float) -> None:
        """Set the motorized stage XY position (microns)."""
        x = max(self.Min_X_position, min(float(x), self.Max_X_position))
        y = max(self.Min_Y_position, min(float(y), self.Max_Y_position))
        with self._lock:
            self.current_X_position = x
            self.current_Y_position = y

    @tool_func
    def get_x_y_position(self) -> Tuple[float, float]:
        """Get the current stage XY position (microns)."""
        with self._lock:
            return float(self.current_X_position), float(self.current_Y_position)

    @tool_func
    def set_z_position(self, z: float) -> None:
        """Set the microscope focus Z position (microns)."""
        z = max(self.Min_Z_position, min(float(z), self.Max_Z_position))
        with self._lock:
            self.current_Z_position = z

    @tool_func
    def get_z_position(self) -> float:
        """Get the current microscope focus Z position (microns)."""
        with self._lock:
            return float(self.current_Z_position)


    # ------------------------------------------------------------------
    # Imaging parameters
    # ------------------------------------------------------------------
    @tool_func
    def set_exposure(self, exposure_time: float) -> None:
        """Set the camera exposure time (ms)."""
        exposure_time = max(self.Min_exposure, min(float(exposure_time), self.Max_exposure))
        with self._lock:
            self.current_exposure_time = exposure_time

    @tool_func
    def get_exposure(self) -> float:
        """Get the current camera exposure time (ms)."""
        with self._lock:
            return float(self.current_exposure_time)

    def _brightfield_label(self) -> str:
        """Return the configured label whose semantic key is 'brightfield' ('' if unknown)."""
        for key, item in self.channels.items():
            if str(key).strip().lower() == "brightfield" and isinstance(item, dict):
                return str(item.get("label") or "").strip()
        return "1-NONE"

    @tool_func
    def set_brightness(self, brightness: int) -> None:
        """Set the transmitted-light brightness (0..Max_brightness).

        In non-brightfield channels the transmitted light is off (0), mirroring
        the real controller / backup simulation.
        """
        if self.current_channel != self._brightfield_label():
            with self._lock:
                self.current_brightness = 0
            return
        brightness = max(self.Min_brightness, min(int(brightness), self.Max_brightness))
        with self._lock:
            self.current_brightness = brightness
            self._user_brightness = brightness

    @tool_func
    def get_brightness(self) -> int:
        """Get the current transmitted-light brightness."""
        with self._lock:
            return int(self.current_brightness)

    @tool_func
    def set_objective(self, objective_label: str) -> None:
        """Set the objective lens to a configured label (e.g. '4-LUCPLFLN40X')."""
        target_label = str(objective_label or "").strip()
        if not target_label:
            raise ValueError("Objective label cannot be empty")
        configured_labels = {
            str(item.get("label") or "").strip()
            for item in self.objectives.values()
            if isinstance(item, dict) and str(item.get("label") or "").strip()
        }
        if configured_labels and target_label not in configured_labels:
            raise ValueError(
                f"Objective label '{target_label}' is not present in the confirmed configuration. "
                f"Configured labels: {sorted(configured_labels)}"
            )
        with self._lock:
            self.current_objective = target_label
            magnification = self.objective_labels.get(target_label)
            if magnification:
                self._pixel_size = 1.6234 * 4.0 / float(magnification)

    @tool_func
    def get_objective(self) -> str:
        """Get the current objective label."""
        with self._lock:
            return self.current_objective

    @tool_func
    def set_channel(self, channel: str) -> None:
        """Set the filter set / channel to a configured label (e.g. '1-NONE' brightfield)."""
        target_label = str(channel or "").strip()
        if not target_label:
            raise ValueError("Channel label cannot be empty")
        configured_labels = {
            str(item.get("label") or "").strip()
            for item in self.channels.values()
            if isinstance(item, dict) and str(item.get("label") or "").strip()
        }
        if configured_labels and target_label not in configured_labels:
            raise ValueError(
                f"Channel label '{target_label}' is not present in the confirmed configuration. "
                f"Configured labels: {sorted(configured_labels)}"
            )
        with self._lock:
            if self.current_channel == self._brightfield_label():
                self._user_brightness = self.current_brightness
            self.current_channel = target_label
            if target_label == self._brightfield_label():
                self.current_brightness = self._user_brightness
            else:
                self.current_brightness = 0

    @tool_func
    def get_channel(self) -> str:
        """Get the current channel label."""
        with self._lock:
            return self.current_channel

    # ------------------------------------------------------------------
    # Acquisition configuration
    # ------------------------------------------------------------------
    @tool_func
    def add_acquisition_position(self, name: str, x: float, y: float, width: float, height: float) -> None:
        """Add a stage position to the automatic acquisition queue.

        Width/height describe the acquisition coverage and may be None when the
        executor defers to the current field of view (mirrors the real controller).
        """
        with self._lock:
            self.acquisition_positions.append(
                {
                    "name": str(name),
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                }
            )

    @tool_func
    def add_channels(self, channel: str, exposure: float) -> None:
        """Configure a channel and its exposure for acquisition."""
        with self._lock:
            self.acquisition_channels.append(
                {"channel": str(channel), "exposure": float(exposure)}
            )

    @tool_func
    def set_z_stack(self, z_start: float, z_end: float, z_step: float) -> None:
        """Configure the Z-stack acquisition range (microns)."""
        z_start = float(z_start)
        z_end = float(z_end)
        z_step = float(z_step)
        if z_step <= 0:
            raise ValueError(f"Z-stack step must be positive, got {z_step}")
        with self._lock:
            self.z_stack_params = {"z_start": z_start, "z_end": z_end, "z_step": z_step}

    @tool_func
    def set_time_series(self, num_frames: int, interval_sec: float) -> None:
        """Configure time-series acquisition (frames and interval seconds)."""
        num_frames = int(num_frames)
        interval_sec = float(interval_sec)
        if num_frames < 1:
            raise ValueError(f"Time-series frame count must be >= 1, got {num_frames}")
        with self._lock:
            self.time_lapse_params = {"num_frames": num_frames, "interval_sec": interval_sec}

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------
    @tool_func
    def run_acquisition(self) -> List[ImagingData]:
        """Perform automatic acquisition and return ImagingData for each position."""
        with self._lock:
            if not self.acquisition_positions:
                raise ValueError("Please add acquisition positions")
            if not self.acquisition_channels:
                raise ValueError("Please configure channels")

            positions = [dict(p) for p in self.acquisition_positions]
            channels = [dict(c) for c in self.acquisition_channels]
            z_params = (
                dict(self.z_stack_params)
                if self.z_stack_params
                else {"z_start": self.current_Z_position, "z_end": self.current_Z_position, "z_step": 1.0}
            )
            time_params = (
                dict(self.time_lapse_params)
                if self.time_lapse_params
                else {"num_frames": 1, "interval_sec": 0.0}
            )
            # Reset configured acquisition plan after use (mirrors real controller).
            self.acquisition_positions = []
            self.acquisition_channels = []
            self.z_stack_params = None
            self.time_lapse_params = None

        num_frames = int(time_params["num_frames"])
        z_positions = np.arange(
            float(z_params["z_start"]),
            float(z_params["z_end"]) + float(z_params["z_step"]) / 2.0,
            float(z_params["z_step"]),
        ).tolist()
        if not z_positions:
            z_positions = [float(z_params["z_start"])]

        os.makedirs(self.output_directory, exist_ok=True)
        results: List[ImagingData] = []
        for pos_index, position in enumerate(positions, start=1):
            data = self._synthesize_image(
                num_frames=num_frames,
                num_channels=len(channels),
                num_z=len(z_positions),
                height=256,
                width=256,
                seed=pos_index,
            )
            filename = f"{position['name']}.ome.tif"
            save_path = os.path.join(self.output_directory, filename)
            self._save_image(data, save_path)
            channel_names = [ch["channel"] for ch in channels]
            desc = (
                f'"channel_names": {channel_names}, '
                f'pixel_size: {self._pixel_size}, '
                f'"objective_label": {self.current_objective}, '
                f'center_x: {float(position["x"] or 0.0)}, '
                f'center_y: {float(position["y"] or 0.0)}'
            )
            self._storagemanger.register_file(filename, desc, "microscope", "ome-tiff")
            imaging_data = ImagingData(
                image=data,
                center_x=float(position["x"] or 0.0),
                center_y=float(position["y"] or 0.0),
                center_z=float(self.current_Z_position),
                objective_magnification=self.current_objective,
                pixel_size=self._pixel_size,
            )
            imaging_data.position_name = position["name"]
            results.append(imaging_data)
        return results

    def _synthesize_image(
        self,
        *,
        num_frames: int,
        num_channels: int,
        num_z: int,
        height: int,
        width: int,
        seed: int,
    ) -> np.ndarray:
        """Build a deterministic TCZYX array with a few bright spheroid-like blobs."""
        rng = np.random.default_rng(seed)
        base = np.zeros((height, width), dtype=np.float32)
        for _ in range(4):
            cy = int(rng.integers(40, height - 40))
            cx = int(rng.integers(40, width - 40))
            radius = float(rng.integers(4, 10))
            yy, xx = np.mgrid[0:height, 0:width]
            base += 180.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * radius * radius))
        data = np.zeros((num_frames, num_channels, num_z, height, width), dtype=np.float32)
        for t in range(num_frames):
            for c in range(num_channels):
                for z in range(num_z):
                    data[t, c, z] = base + rng.normal(0.0, 2.0, size=(height, width)).astype(np.float32)
        return data

    def _save_image(self, data: np.ndarray, save_path: str) -> None:
        """Save a TCZYX array as an OME-TIFF (fallback to plain TIFF)."""
        try:
            from aicsimageio.writers import OmeTiffWriter

            OmeTiffWriter.save(data, save_path, dim_order="TCZYX")
            return
        except Exception as exc:  # pragma: no cover - fallback path
            import logging

            logging.getLogger(__name__).debug("aicsimageio OME-TIFF save failed, using tifffile: %s", exc)
        try:
            import tifffile

            tifffile.imwrite(save_path, data)
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Failed to save simulated image to {save_path}: {exc}") from exc

    # ------------------------------------------------------------------
    # Autofocus / autobrightness
    # ------------------------------------------------------------------
    @tool_func
    def perform_autofocus(
        self,
        tolerance=0.5,
        use_auto_params=False,
        search_range=600.0,
        min_z: Optional[float] = None,
        max_z: Optional[float] = None,
    ) -> float:
        """Simulate autofocus: keep the current focus unchanged (no drift)."""
        del tolerance, use_auto_params, search_range
        current = self.current_Z_position
        if min_z is not None:
            current = max(current, float(min_z))
        if max_z is not None:
            current = min(current, float(max_z))
        best_z = max(self.Min_Z_position, min(self.Max_Z_position, current))
        with self._lock:
            self.current_Z_position = best_z
        return float(best_z)

    @tool_func
    def perform_autobrightness(
        self,
        tolerance: Optional[float] = None,
        target_high_percentile: float = 0.82,
        high_percentile: float = 99.5,
        max_saturation_ratio: float = 0.002,
        min_median_ratio: float = 0.08,
        max_iterations: int = 8,
        settle_time_sec: float = 0.15,
    ) -> int:
        """Simulate automatic brightness adjustment: set a mid-range value."""
        del (
            tolerance,
            target_high_percentile,
            high_percentile,
            max_saturation_ratio,
            min_median_ratio,
            max_iterations,
            settle_time_sec,
        )
        result = max(self.Min_brightness, min(self.Max_brightness, int(self.current_brightness) + 5))
        self.set_brightness(result)
        return int(result)

    # ------------------------------------------------------------------
    # Targets and wells
    # ------------------------------------------------------------------
    @tool_func
    def detect_targets_in_image(
        self,
        image_data: ImagingData,
        target_class: str,
        confidence_threshold: float = 0.5,
        device: Optional[Any] = None,
    ) -> List[Dict[str, float]]:
        """Simulate target detection: return two synthetic detections around the image center."""
        del device
        if not isinstance(image_data, ImagingData):
            raise TypeError("image_data must be an ImagingData instance")
        target_name = str(target_class or "").strip()
        if not target_name:
            raise ValueError("target_class cannot be empty")
        if image_data.pixel_size is None or float(image_data.pixel_size) <= 0:
            raise ValueError("image_data.pixel_size must be a positive number")

        image_2d = _coerce_detection_image_to_2d(image_data.image)
        pixel_size = float(image_data.pixel_size)
        image_f = image_2d.astype(np.float32, copy=False)
        if image_f.size == 0:
            return []
        img_max = float(np.max(image_f))
        if img_max <= 0:
            return []

        # Normalize to [0, 1] so one heuristic works across uint8/uint16 mock images.
        image_norm = image_f / (img_max + 1e-8)
        threshold = max(0.6, float(confidence_threshold))
        mask = image_norm >= threshold
        if not np.any(mask):
            return []

        h, w = image_2d.shape
        img_center_x_px = (w - 1) / 2.0
        img_center_y_px = (h - 1) / 2.0
        image_center_x_um = float(image_data.center_x)
        image_center_y_um = float(image_data.center_y)

        visited = np.zeros(mask.shape, dtype=bool)
        results: List[Dict[str, float]] = []
        max_regions = 8

        for seed_y, seed_x in np.argwhere(mask):
            if visited[int(seed_y), int(seed_x)]:
                continue
            stack = [(int(seed_y), int(seed_x))]
            component: List[Tuple[int, int]] = []
            visited[int(seed_y), int(seed_x)] = True
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if len(component) < 5:
                continue

            ys = np.array([pt[0] for pt in component], dtype=np.float32)
            xs = np.array([pt[1] for pt in component], dtype=np.float32)
            scores = image_norm[ys.astype(np.int32), xs.astype(np.int32)]
            score = float(np.mean(scores))
            if score < confidence_threshold:
                continue

            cx_px = float(np.mean(xs))
            cy_px = float(np.mean(ys))
            offset_x_um = (cx_px - img_center_x_px) * pixel_size
            # Match the OME/Cellpose/Fiji/MP285 image-to-stage convention (real controller).
            offset_y_um = (cy_px - img_center_y_px) * pixel_size
            results.append(
                {
                    "center_x_um": float(image_center_x_um + offset_x_um),
                    "center_y_um": float(image_center_y_um + offset_y_um),
                    "offset_x_um": float(offset_x_um),
                    "offset_y_um": float(offset_y_um),
                    "confidence": float(min(1.0, score)),
                }
            )
            if len(results) >= max_regions:
                break

        results.sort(key=lambda item: item["confidence"], reverse=True)
        return results

    @tool_func
    def load_target_locations(self, filename: str) -> List[Tuple[float, float, float, float]]:
        """Load target location regions [x, y, width, height] (microns) from a JSON file."""
        filepath = os.path.join(self.output_directory, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Target location file not found: {filepath}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Target location file is not valid JSON: {filepath}") from exc
        if not isinstance(loaded_data, list):
            raise ValueError(f"Target location file must contain a JSON array: {filepath}")

        regions = []
        for item in loaded_data:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                x, y, width, height = map(float, item)
                regions.append((x, y, width, height))
        return regions

    @tool_func
    def z_stack_range(self) -> Tuple[float, float]:
        """Return a recommended Z-stack range as (z_max, z_min) around the current focus.

        Mirrors the real controller's sharpness-vs-Z recommendation: a narrow
        window around the current focus position (40x -> +/-40 um).
        """
        half_width = 40.0
        current = self.current_Z_position
        z_max = min(float(self.Max_Z_position), current + half_width)
        z_min = max(float(self.Min_Z_position), current - half_width)
        if z_max <= z_min:
            z_max = z_min + 1.0
        return z_max, z_min

    @tool_func
    def create_96_wells_positions(self) -> List[Tuple[float, float]]:
        """Generate the standard 8x12 96-well plate XY positions (microns, 9 mm pitch)."""
        positions = []
        for row in range(8):
            for col in range(12):
                positions.append((float(col * 9000.0), float(row * 9000.0)))
        return positions

    @tool_func
    def create_24_wells_positions(self) -> List[Tuple[float, float]]:
        """Generate the standard 4x6 24-well plate XY positions (microns, ~19.3 mm pitch)."""
        positions = []
        for row in range(4):
            for col in range(6):
                positions.append((float(col * 19300.0), float(row * 19300.0)))
        return positions

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------
    def start_preview(self) -> None:
        """Start simulated live preview (synthesizes frames)."""
        with self._lock:
            self.preview_running = True
            self._latest_display_frame = self._synthesize_preview_frame()

    @tool_func
    def stop_preview(self) -> None:
        """Stop the live preview stream."""
        with self._lock:
            self.preview_running = False

    def get_live_preview_image(self) -> Optional[np.ndarray]:
        """Return the latest simulated preview frame."""
        with self._lock:
            if self._latest_display_frame is None:
                self._latest_display_frame = self._synthesize_preview_frame()
            return self._latest_display_frame.copy()

    def _synthesize_preview_frame(self) -> np.ndarray:
        self._frame_counter += 1
        rng = np.random.default_rng(self._frame_counter)
        frame = np.zeros((256, 256), dtype=np.uint8)
        cy = 96 + int(20 * np.sin(self._frame_counter * 0.3))
        cx = 128 + int(20 * np.cos(self._frame_counter * 0.2))
        yy, xx = np.mgrid[0:256, 0:256]
        frame += (120.0 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 12.0 * 12.0))).astype(np.uint8)
        frame += rng.integers(0, 12, size=(256, 256)).astype(np.uint8)
        return frame

    def get_transmitted_light_runtime_info(self) -> Dict[str, Any]:
        """Report simulated transmitted-light runtime information."""
        return {
            "device": self.brightness_device,
            "property": self.brightness_property,
            "control_kind": "mock_in_memory",
            "min": self.Min_brightness,
            "max": self.Max_brightness,
        }

    def _supports_transmitted_brightness(self) -> bool:
        return bool(self.brightness_device)

    # ------------------------------------------------------------------
    # FRAP hardware-owner handoff
    # ------------------------------------------------------------------
    def capture_handoff_state(self) -> Dict[str, Any]:
        """Capture current microscope state before handing hardware to another tool."""
        with self._lock:
            return {
                "x": float(self.current_X_position),
                "y": float(self.current_Y_position),
                "z": float(self.current_Z_position),
                "objective": self.current_objective,
                "channel": self.current_channel,
                "exposure": float(self.current_exposure_time),
                "brightness": int(self.current_brightness),
                "preview_running": bool(self.preview_running),
            }

    def release_for_handoff(self) -> None:
        """Release the simulated microscope for another hardware owner (no-op)."""
        with self._lock:
            self.preview_running = False

    def restore_after_handoff(self, state: Dict[str, Any]) -> None:
        """Restore microscope state after another hardware owner releases it."""
        with self._lock:
            if not isinstance(state, dict):
                raise ValueError("handoff state must be a dict")
            self.current_X_position = float(state.get("x", self.current_X_position))
            self.current_Y_position = float(state.get("y", self.current_Y_position))
            self.current_Z_position = float(state.get("z", self.current_Z_position))
            self.current_objective = str(state.get("objective", self.current_objective))
            self.current_channel = str(state.get("channel", self.current_channel))
            self.current_exposure_time = float(state.get("exposure", self.current_exposure_time))
            self.current_brightness = int(state.get("brightness", self.current_brightness))
            self.preview_running = bool(state.get("preview_running", False))

    # ------------------------------------------------------------------
    # Task progress
    # ------------------------------------------------------------------
    def set_task_progress_listener(self, listener: Optional[Any]) -> None:
        """Set a callback that receives task progress dictionaries."""
        with self._lock:
            self._task_progress_listener = listener

    def get_last_task_progress(self) -> Optional[Dict[str, Any]]:
        """Return the last emitted task progress dictionary."""
        with self._lock:
            return dict(self._last_task_progress) if self._last_task_progress else None

    def _emit_task_progress(self, **payload: Any) -> None:
        with self._lock:
            self._last_task_progress = dict(payload)
            listener = self._task_progress_listener
        if callable(listener):
            try:
                listener(dict(payload))
            except Exception:
                pass
