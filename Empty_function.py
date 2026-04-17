from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tool.base import BaseTool,tool_func
import math
from contextlib import contextmanager
import threading
from datetime import datetime
from queue import Queue  # Thread-safe queue for image transmission
from typing import Any, List, Dict, Tuple, Optional, Sequence, Literal
import numpy as np
import os
import pandas as pd

try:
    from aicsimageio.types import PhysicalPixelSizes
except Exception:
    @dataclass
    class PhysicalPixelSizes:
        Z: Optional[float] = None
        Y: Optional[float] = None
        X: Optional[float] = None


# Map objective magnification to objective labels
objective_labels = {
    '1-UPLFLN4XPH': 4,
    '2-SOB': 10,
    '3-LUCPLFLN20XRC': 20,
    '4-LUCPLFLN40X': 40,
    '5-LUCPLFLN60X': 60,
    '6-UPLSAPO30XS': 30
}


def _coerce_detection_image_to_2d(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return image_array

    squeezed = np.squeeze(image_array)
    if squeezed.ndim == 2:
        return squeezed

    raise ValueError("Only 2D grayscale image or singleton multidimensional image supported")

# Channel to color mapping (RGB values)
dichroic_colors = {
    '1-NONE': (128, 128, 128),  # Gray (brightfield)
    '2-U-FUNA': (0, 0, 255),  # Red
    '3-U-FBNA': (0, 255, 0),  # Green
    '4-U-FGNA': (255, 0, 0),  # Blue
}


@dataclass
class ImagingData:
    image: np.ndarray
    center_x: float
    center_y: float
    center_z: float
    objective_magnification: str
    pixel_size: Optional[float] = None
    position_name: str = ""


@dataclass
class ImageWithMetadata:
    dataset: Any
    center_x_um: float
    center_y_um: float
    center_z_um: float = 0.0
    pixel_size_x_um: float = 1.0
    pixel_size_y_um: float = 1.0

    @property
    def pixel_size_um(self) -> float:
        return float(self.pixel_size_x_um)


class MicroscopeController(BaseTool):
    """
    Core class for microscope control system, supporting real-time preview, image acquisition, 
    and multi-dimensional automated acquisition.
    All physical units are in micrometers (μm) by default, and time units are in milliseconds (ms) by default.
    Test version: Keep input/output structure without implementing actual hardware control functions
    """

    def __init__(self, config_path: str, app_dir: str, output_path: str, storagemanger):
        self._storagemanger = storagemanger
        self.app_dir = app_dir
        self.config_path = config_path

        self.device_lock = threading.RLock()

        self.camera_device = 'Camera-1'
        self.xy_stage_device = 'XYStage'
        self.objective_device = 'Objective'
        self.transmittedIllumination = 'TransmittedIllumination 2'
        self.focus_drive = 'FocusDrive'
        self.Dichroic = 'Dichroic 2'

        self.Max_X_position: float = 500000
        self.Min_X_position: float = 0
        self.Max_Y_position: float = 500000
        self.Min_Y_position: float = 0
        self.Max_Z_position: float = 10000
        self.Min_Z_position: float = 0
        self.Max_brightness = 250
        self.Min_brightness = 0
        self.Max_exposure = 1000
        self.Min_exposure = 0

        self.current_channel = '1-NONE'
        self.current_objective = '1-UPLFLN4XPH'
        self.current_X_position = 0.0
        self.current_Y_position = 0.0
        self.current_Z_position = 0.0
        self.current_brightness = 100
        self._user_brightness = self.current_brightness
        self.current_exposure_time = 100.0

        self.acquisition_positions: List[Dict] = []
        self.acquisition_channels: List[Dict] = []
        self.z_stack_params: Optional[Dict] = None
        self.time_lapse_params: Optional[Dict] = None
        self.output_directory: str = output_path

        self.pixel_size = 1.6234  # μm/pixel
        self.img_dtype = np.uint16
        self.current_img_height = 1024
        self.current_img_width = 1024
        self.is_16bit = True

        self.preview_running = False
        self.preview_auto_restart_enabled = True
        self.acquisition_thread = None
        self.image_queue = Queue(maxsize=5)
        self.preview_interval = 0.04
        self.preview_window_name = "micro live"
        self.is_continuous = False
        self.display_ready = threading.Event()

        self.acquisition_running = False
        self.acquisition_abort = False

        self.auto_contrast_enabled = True
        self.contrast_percentile = 0.1
        self.initialized = False
        self.laser_enabled = False

    def _deterministic_image(self, height: int | None = None, width: int | None = None, *, fill: int = 0) -> np.ndarray:
        image_height = height or self.current_img_height
        image_width = width or self.current_img_width
        values = np.arange(image_height * image_width, dtype=self.img_dtype).reshape(image_height, image_width)
        if fill:
            values = (values + np.array(fill, dtype=self.img_dtype)) % np.iinfo(self.img_dtype).max
        return values

    def _write_mock_output_file(self, filename: str, payload: str) -> Path:
        output_path = Path(self.output_directory, filename).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        return output_path

    @contextmanager
    def _acquisition_guard(self):
        previous_running = self.acquisition_running
        previous_auto_restart = self.preview_auto_restart_enabled
        self.acquisition_running = True
        self.preview_auto_restart_enabled = False
        try:
            yield
        finally:
            self.acquisition_running = previous_running
            self.preview_auto_restart_enabled = previous_auto_restart

    @tool_func
    def initialize(self):
        self.initialized = True
        os.makedirs(self.output_directory, exist_ok=True)
        return True

    @tool_func
    def set_x_y_position(self, x: float, y: float):
        print("Running function: set_x_y_position")
        x = max(self.Min_X_position, min(x, self.Max_X_position))
        y = max(self.Min_Y_position, min(y, self.Max_Y_position))
        self.current_X_position = x
        self.current_Y_position = y

    @tool_func
    def set_z_position(self, z: float):
        print("Running function: set_z_position")
        z = max(self.Min_Z_position, min(z, self.Max_Z_position))
        self.current_Z_position = z

    @tool_func
    def get_x_y_position(self) -> Tuple[float, float]:
        print("Running function: get_x_y_position")
        return (self.current_X_position, self.current_Y_position)

    @tool_func
    def get_z_position(self) -> float:
        print("Running function: get_z_position")
        return self.current_Z_position

    @tool_func
    def set_exposure(self, exposure_time: float):
        print("Running function: set_exposure")
        exposure_time = max(self.Min_exposure, min(exposure_time, self.Max_exposure))
        self.current_exposure_time = exposure_time

    @tool_func
    def get_exposure(self) -> float:
        print("Running function: get_exposure")
        return self.current_exposure_time

    @tool_func
    def set_brightness(self, brightness: int):
        print("Running function: set_brightness")
        if self.current_channel != '1-NONE':
            self.current_brightness = 0
        else:
            brightness = max(self.Min_brightness, min(brightness, self.Max_brightness))
            self.current_brightness = brightness
            self._user_brightness = brightness

    def remember_brightfield_brightness(self, brightness: int) -> None:
        brightness = max(self.Min_brightness, min(int(brightness), self.Max_brightness))
        self._user_brightness = brightness
        if self.current_channel == '1-NONE':
            self.current_brightness = brightness

    @tool_func
    def get_brightness(self) -> int:
        print("Running function: get_brightness")
        return self.current_brightness

    @tool_func
    def set_objective(self, objective_label: str):
        print("Running function: set_objective")
        if objective_label in objective_labels:
            self.current_objective = objective_label
            self.pixel_size = 1.6234 * 4 / objective_labels[self.current_objective]

    @tool_func
    def set_channel(self, channel: str):
        print("Running function: set_channel")
        if channel in dichroic_colors:
            if self.current_channel == '1-NONE':
                self._user_brightness = self.current_brightness
            self.current_channel = channel
            if channel == '1-NONE':
                self.current_brightness = self._user_brightness
            else:
                self.current_brightness = 0

    @tool_func
    def get_channel(self):
        print("Running function: get_channel")
        return self.current_channel

    @tool_func
    def get_objective(self):
        print("Running function: get_objective")
        return self.current_objective

    @tool_func
    def start_preview(self):
        print("Running function: start_preview")
        self.preview_running = True

    @tool_func
    def stop_preview(self):
        print("Running function: stop_preview")
        self.preview_running = False

    @tool_func
    def get_image(self, width_micro=None, height_micro=None):
        print("Running function: get_image")
        if width_micro and height_micro:
            return self._acquire_stitch_mosaic(width_micro, height_micro)
        else:
            return self._acquire_single_image()

    def _acquire_single_image(self) -> np.ndarray:
        with self._acquisition_guard():
            return self._deterministic_image(fill=int(self.current_Z_position) % 1024)

    def _acquire_stitch_mosaic(self, width_micro: float, height_micro: float, overlap=0.1) -> np.ndarray:
        with self._acquisition_guard():
            if overlap < 0.05 or overlap >= 0.5:
                raise ValueError("Overlap rate must be between 5% and 50%")

            fov_width = self.current_img_width * self.pixel_size
            fov_height = self.current_img_height * self.pixel_size
            step_x = fov_width * (1 - overlap)
            step_y = fov_height * (1 - overlap)
            cols = max(1, math.ceil(width_micro / step_x))
            rows = max(1, math.ceil(height_micro / step_y))

            mosaic_height = self.current_img_height * rows
            mosaic_width = self.current_img_width * cols
            mosaic = self._deterministic_image(mosaic_height, mosaic_width, fill=int(width_micro + height_micro))

            return mosaic

    @tool_func
    def add_acquisition_position(self, name: str, x: float, y: float, width: float, height: float) -> None:
        print("Running function: add_acquisition_position")
        for pos in self.acquisition_positions:
            if pos["name"] == name:
                return
        self.acquisition_positions.append({
            "name": name,
            "x": x,
            "y": y,
            'width': width,
            'height': height
        })

    @tool_func
    def add_channels(self, channel: str, exposure: float) -> None:
        print("Running function: add_channels")
        for existing in self.acquisition_channels:
            if existing["channel"] == channel:
                return
        self.acquisition_channels.append({
            "channel": channel,
            "exposure": exposure
        })

    @tool_func
    def set_z_stack(self, z_start: float, z_end: float, z_step: float) -> None:
        print("Running function: set_z_stack")
        if z_step <= 0:
            raise ValueError("Z-axis step size must be positive")
        if (z_end - z_start) * z_step < 0:
            raise ValueError("Z-axis step direction conflicts with start and end range")
        self.z_stack_params = {
            "z_start": z_start,
            "z_end": z_end,
            "z_step": z_step
        }

    @tool_func
    def set_time_series(self, num_frames: int, interval_sec: float) -> None:
        print("Running function: set_time_series")
        self.time_lapse_params = {
            "num_frames": num_frames,
            "interval_sec": interval_sec
        }

    @tool_func
    def run_acquisition(self) -> List[ImagingData]:
        print("Running function: run_acquisition")
        with self._acquisition_guard():
            if not self.acquisition_positions:
                raise ValueError("Please add at least one acquisition position")
            if not self.acquisition_channels:
                raise ValueError("Please configure at least one acquisition channel")

            if not self.time_lapse_params:
                self.time_lapse_params = {"num_frames": 1, "interval_sec": 0}

            if not self.z_stack_params:
                current_z = self.get_z_position()
                self.z_stack_params = {"z_start": current_z, "z_end": current_z, "z_step": 1}

            init_x, init_y = self.get_x_y_position()
            init_z = self.get_z_position()
            init_channel = self.get_channel()
            init_exposure = self.get_exposure()

            time_num_frames = self.time_lapse_params["num_frames"]
            tim_interval = self.time_lapse_params["interval_sec"]
            z_start = self.z_stack_params['z_start']
            z_end = self.z_stack_params['z_end']
            z_step = self.z_stack_params['z_step']

            channels_name = [ch["channel"] for ch in self.acquisition_channels]
            acquisition_results: List[ImagingData] = []

            try:
                position_data = []
                for pos in self.acquisition_positions:
                    try:
                        x_pos = pos["x"]
                        y_pos = pos["y"]
                        width = pos["width"]
                        height = pos["height"]
                        self.set_x_y_position(x_pos, y_pos)

                        metadata = self._create_ome_metadata(
                            channel_names=channels_name,
                            time_interval=tim_interval if time_num_frames > 1 else None,
                            microscope="olympus lx83",
                            objective=self.current_objective,
                            pixel_type=self.img_dtype
                        )

                        position_data.append({
                            "name": pos["name"],
                            "metadata": metadata,
                            "x": x_pos,
                            "y": y_pos,
                            "width": width,
                            "height": height
                        })
                    except Exception:
                        continue

                for pos in position_data:
                    os.makedirs(self.output_directory, exist_ok=True)
                    channel_colors = [dichroic_colors.get(ch, "Unknown") for ch in channels_name]
                    objective_magnification = objective_labels[self.current_objective]
                    description = (f'channel_names: {channel_colors}, '
                                   f'pixel_size: {self.pixel_size:.2f}, '
                                   f'objective_label: {self.current_objective}, '
                                   f'magnification: {objective_magnification}, '
                                   f'frames: {time_num_frames}')
                    output_filename = f"{pos['name']}.ome.tif"
                    self._write_mock_output_file(
                        output_filename,
                        "\n".join(
                            [
                                "mock microscope acquisition",
                                f"name={pos['name']}",
                                f"channel_names={channels_name}",
                                f"pixel_size={self.pixel_size:.4f}",
                                f"objective_label={self.current_objective}",
                                f"magnification={objective_magnification}",
                                f"frames={time_num_frames}",
                            ]
                        ),
                    )
                    self._storagemanger.register_file(output_filename, description, 'microscope', 'ome-tiff')

                    self.set_x_y_position(pos["x"], pos["y"])
                    self.set_z_position(init_z)
                    frame_height = self.current_img_height
                    frame_width = self.current_img_width
                    if pos["width"] and pos["height"]:
                        mock_image = self._acquire_stitch_mosaic(pos["width"], pos["height"])
                        frame_height, frame_width = mock_image.shape
                    else:
                        mock_image = self._acquire_single_image()

                    for frame_idx in range(time_num_frames):
                        if frame_idx == 0:
                            frame_image = mock_image
                        else:
                            frame_image = self._deterministic_image(frame_height, frame_width, fill=frame_idx * 17)

                        acquisition_results.append(
                            ImagingData(
                                image=frame_image,
                                center_x=pos["x"],
                                center_y=pos["y"],
                                center_z=init_z,
                                objective_magnification=self.current_objective,
                                pixel_size=self.pixel_size,
                                position_name=(
                                    pos["name"] if time_num_frames == 1 else f"{pos['name']}_frame_{frame_idx + 1}"
                                ),
                            )
                        )
            finally:
                self.acquisition_positions.clear()
                self.acquisition_channels.clear()
                self.z_stack_params = None
                self.time_lapse_params = None
                self.set_x_y_position(init_x, init_y)
                self.set_z_position(init_z)
                self.set_channel(init_channel)
                self.set_exposure(init_exposure)
            return acquisition_results

    def _create_ome_metadata(self, channel_names, time_interval, microscope, objective, pixel_type) -> Dict:
        channel_colors = [dichroic_colors.get(ch, (128, 128, 128)) for ch in channel_names]
        return {
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "time_interval": time_interval,
            "microscope": microscope,
            "objective": objective,
            "objective_label": objective,
            "objective_magnification": objective_labels.get(objective),
            "datetime": "2026-01-01T00:00:00",
            "pixel_type": pixel_type
        }

    def _get_focus_params_for_magnification(self, magnification: float) -> Tuple[float, float]:
        if magnification < 5:
            return 50.0, 10.0
        elif magnification < 10:
            return 30.0, 5.0
        elif magnification < 20:
            return 20.0, 2.0
        elif magnification < 50:
            return 10.0, 1.0
        else:
            return 5.0, 0.5

    @tool_func
    def perform_autofocus(self, tolerance=0.5, use_auto_params=True, search_range=500.0) -> float:
        print("Running function: perform_autofocus")
        del tolerance, use_auto_params, search_range
        current_z = self.get_z_position()
        best_z = current_z + 5.0
        best_z = max(self.Min_Z_position, min(best_z, self.Max_Z_position))
        best_z = float(best_z)  # Convert string input to float (core correction)
        self.set_z_position(best_z)
        return best_z

    @tool_func
    def perform_autobrightness(
        self,
        target_high_percentile: float = 0.82,
        high_percentile: float = 99.5,
        max_saturation_ratio: float = 0.002,
        min_median_ratio: float = 0.08,
        max_iterations: int = 8,
        settle_time_sec: float = 0.15,
    ) -> int:
        print("Running function: perform_autobrightness")
        del target_high_percentile, high_percentile, max_saturation_ratio, min_median_ratio, max_iterations, settle_time_sec
        best_brightness = int(self.current_brightness + 5)
        best_brightness = max(self.Min_brightness, min(best_brightness, self.Max_brightness))
        self.set_brightness(best_brightness)
        return best_brightness

    @tool_func
    def shutdown(self):
        print("Running function: shutdown")
        self.stop_preview()
        self.acquisition_abort = True
        self.laser_enabled = False
        self.set_brightness(self.Min_brightness)

    @tool_func
    def z_stack_range(self) -> Tuple[int, int]:
        print("Running function: z_stack_range")
        return (5000, 1000)

    @tool_func
    def load_target_locations(self, filename: str) -> List[Tuple[float, float, float, float]]:
        print("Running function: load_target_locations")
        del filename
        return [
            (54000.0, 33400.0, 500, 500),
            (54184.6, 33513.0, 500, 500),
            (54300.0, 33800.0, 500, 500),
            (54415.9, 34164.4, 500, 500),
            (54550.0, 34500.0, 500, 500),
        ]

    @tool_func
    def create_96_wells_positions(self, start_position=(0, 0)):
        print("Running function: create_96_wells_positions")
        rows = 8
        cols = 12
        spacing_x = 18
        spacing_y = 18
        start_x, start_y = start_position
        positions = []
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * spacing_x
                y = start_y + row * spacing_y
                positions.append((x, y))
        return positions

    @tool_func
    def create_24_wells_positions(self, start_position=(0, 0)):
        print("Running function: create_24_wells_positions")
        rows = 4
        cols = 6
        spacing_x = 18
        spacing_y = 18
        start_x, start_y = start_position
        positions = []
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * spacing_x
                y = start_y + row * spacing_y
                positions.append((x, y))
        return positions
    

    @tool_func
    def detect_targets_in_image(
            self,
            image: np.ndarray,
            target_class: str,
            pixel_size: float,
            confidence_threshold: float = 0.5,
            device: Optional[Any] = None
    ) -> List[Dict[str, float]]:
        print("Running function: detect_targets_in_image")
        del target_class
        del device

        image_2d = _coerce_detection_image_to_2d(image)
        if pixel_size <= 0:
            raise ValueError("pixel_size must be positive")

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

        coords = np.argwhere(mask)
        if coords.size == 0:
            return []

        h, w = image_2d.shape
        img_center_x_px = (w - 1) / 2.0
        img_center_y_px = (h - 1) / 2.0
        image_center_x_um, image_center_y_um = self.get_x_y_position()

        visited = np.zeros(mask.shape, dtype=bool)
        results: List[Dict[str, float]] = []
        max_regions = 8

        for seed_y, seed_x in coords:
            if visited[seed_y, seed_x]:
                continue

            stack = [(int(seed_y), int(seed_x))]
            component: List[tuple[int, int]] = []
            visited[seed_y, seed_x] = True

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
            offset_y_um = -(cy_px - img_center_y_px) * pixel_size
            center_x_um = image_center_x_um + offset_x_um
            center_y_um = image_center_y_um + offset_y_um

            results.append(
                {
                    "center_x_um": float(center_x_um),
                    "center_y_um": float(center_y_um),
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
    def get_live_preview_image(self) -> np.ndarray:
        return self._deterministic_image(fill=123)

    @tool_func
    def open_laser(self) -> None:
        self.laser_enabled = True

    @tool_func
    def close_laser(self) -> None:
        self.laser_enabled = False


class Cellpose2D(BaseTool):
    def __init__(self, storagemanger, output_path: str):
        self._storagemanger = storagemanger
        self.output_directory: str = output_path
        self.model = None

    def _resolve_input_path(self, file_path: str | Path) -> Path:
        candidate = Path(file_path).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = Path(self.output_directory, candidate).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Mock input file does not exist: {resolved}")
        return resolved

    def _save_mock_target_preview(self, output_json_path: Path, masks: np.ndarray) -> Path:
        preview_path = output_json_path.with_name(f"{output_json_path.stem}_annotated.png")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.touch(exist_ok=True)
        return preview_path

    @tool_func
    def cellpose_initialize(self, gpu: bool = False, model_type: str = "cpsam"):
        print("Running function: cellpose_initialize")
        self.model = "MOCK_MODEL"

    @tool_func
    def cellpose_read(self, file_path: str) -> np.ndarray:
        print("Running function: cellpose_read")
        self._resolve_input_path(file_path)
        shape = (3, 3, 3, 32, 32)
        return np.zeros(shape, dtype=np.float32)

    @tool_func
    def segment(
        self,
        image: np.ndarray,
        channels: Sequence[int] | None = None,
        diameter: float | None = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 15,
        denoise: bool = False,
    ) -> np.ndarray:
        print("Running function: segment")
        del image, channels, diameter, flow_threshold, cellprob_threshold, min_size, denoise
        return np.ones((32, 32), dtype=np.int32)

    @tool_func
    def analyze_masks(
        self,
        masks: np.ndarray,
        px_size: float = 1.0,
        unit: Literal["px", "μm2"] = "px",
        bins: int | np.ndarray = 20,
        plot: bool = False,
        **bar_kwargs
    ) -> pd.DataFrame:
        print("Running function: analyze_masks")
        return pd.DataFrame({
            "cell_id": np.arange(1, 101),
            "area": np.linspace(50, 500, 100, dtype=np.float64),
            "bin_idx": np.arange(100) % 10,
        })

    @tool_func
    def save_masks(self, masks: np.ndarray, filename: str | Path, description) -> Path:
        print("Running function: save_masks")
        del masks
        output_path = Path(self.output_directory, filename).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=True)
        self._storagemanger.register_file(output_path.name, str(description), 'cellpose', 'tiff')
        return output_path

    @tool_func
    def save_csv(self, df: pd.DataFrame, filename: str | Path) -> Path:
        print("Running function: save_csv")
        output_path = Path(self.output_directory, filename).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        df.to_csv(output_path, index=False)
        self._storagemanger.register_file(output_path.name, "Cellpose analysis CSV", 'cellpose', 'csv')
        return output_path

    @tool_func
    def save_target_locations(
        self,
        masks: np.ndarray,
        source_image_path: str | Path,
        filename: str | Path = "cellpose_target_locations.json",
        description: str = "Cellpose target locations for microscope reacquisition",
        min_area_px: int = 15,
        max_area_px: int | None = None,
        top_k: int | None = None,
    ) -> Path:
        print("Running function: save_target_locations")
        del source_image_path, min_area_px, max_area_px, top_k
        if np.asarray(masks).ndim < 2:
            raise ValueError("masks must be at least 2D")

        output_path = Path(self.output_directory, filename).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[[1000, 2000, 64, 64]]", encoding="utf-8")
        self._storagemanger.register_file(output_path.name, str(description), "cellpose", "json")
        self._save_mock_target_preview(output_path, np.asarray(masks))
        return output_path

    @tool_func
    def color_masks(self, masks: np.ndarray) -> np.ndarray:
        print("Running function: color_masks")
        if masks.ndim != 2:
            raise ValueError("masks must be a 2D array")

        colored = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
        unique_labels = [int(label) for label in np.unique(masks) if int(label) != 0]
        palette = [
            (220, 20, 60),
            (65, 105, 225),
            (50, 205, 50),
            (255, 165, 0),
            (138, 43, 226),
            (0, 206, 209),
        ]
        for index, label in enumerate(unique_labels):
            colored[masks == label] = palette[index % len(palette)]
        return colored

    @tool_func
    def export_results(self, masks: np.ndarray, base_filename: str, image: np.ndarray | None = None):
        print("Running function: export_results")
        self.save_masks(masks, f"{base_filename}_masks.tif", "Cellpose segmentation mask")
        colored_mask = self.color_masks(masks)

        color_path = Path(self.output_directory, f"{base_filename}_colored.png").expanduser().resolve()
        color_path.parent.mkdir(parents=True, exist_ok=True)
        color_path.touch(exist_ok=True)
        self._storagemanger.register_file(color_path.name, "Colored cellpose mask", "cellpose", "png")

        df = self.analyze_masks(masks)
        csv_path = self.save_csv(df, f"{base_filename}_analysis.csv")

        overlay_path: Path | None = None
        if image is not None:
            del image
            overlay_path = Path(self.output_directory, f"{base_filename}_overlay.png").expanduser().resolve()
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path.touch(exist_ok=True)
            self._storagemanger.register_file(overlay_path.name, "Cellpose overlay image", "cellpose", "png")

        return {
            "mask_path": str(Path(self.output_directory, f"{base_filename}_masks.tif").expanduser().resolve()),
            "colored_mask": colored_mask,
            "colored_mask_path": str(color_path),
            "analysis_csv_path": str(csv_path),
            "overlay_path": str(overlay_path) if overlay_path is not None else None,
        }

    @staticmethod
    def _unique(p: Path) -> Path:
        return p


class ImageJProcessor(BaseTool):
    def __init__(self, storagemanger, output_path: str):
        self._storagemanger = storagemanger
        self.output_directory: str = output_path
        self.ij = None
        self.initialized = False

    def _resolve_input_path(self, file_name: str | Path) -> Path:
        candidate = Path(file_name).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = Path(self.output_directory, candidate).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Mock input file does not exist: {resolved}")
        return resolved

    @tool_func
    def fiji_initialize(self, fiji_path=None):
        print("Running function: fiji_initialize")
        del fiji_path
        self.ij = "mock_imagej_instance"
        self.initialized = True
        return True

    @tool_func
    def load_image(self, file_name) -> ImageWithMetadata:
        print("Running function: load_image")
        resolved_path = self._resolve_input_path(file_name)
        return ImageWithMetadata(
            dataset=f"mock_dataset_{resolved_path.name}",
            center_x_um=0.0,
            center_y_um=0.0,
            center_z_um=0.0,
            pixel_size_x_um=1.0,
            pixel_size_y_um=1.0,
        )

    def _load_image_IMP(self, file_path):
        return f"mock_imp_{os.path.basename(file_path)}"

    @tool_func
    def dataset_to_imp(self, dataset):
        dataset_value = dataset.dataset if isinstance(dataset, ImageWithMetadata) else dataset
        return f"mock_imp_from_{dataset_value}"

    @tool_func
    def save_image(self, image_meta, filename, description):
        print("Running function: save_image")
        del image_meta
        output_path = Path(self.output_directory, filename).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=True)
        self._storagemanger.register_file(filename, description, 'analysis_platform', 'ome-tiff')
        return str(output_path)

    @tool_func
    def adjust_contrast(self, image_meta, saturated=5) -> ImageWithMetadata:
        print("Running function: adjust_contrast")
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        return self._clone_image_meta(source, dataset=f"mock_contrast_enhanced_{saturated}_{source.dataset}")

    @tool_func
    def dump_info(self, image):
        return {
            "image": str(image),
            "backend": "mock_imagej_instance",
            "output_directory": self.output_directory,
        }

    @tool_func
    def split_channels(self, image_meta) -> List[ImageWithMetadata]:
        print("Running function: split_channels")
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        return [
            self._clone_image_meta(source, dataset=f"mock_channel_{index}_{source.dataset}")
            for index in range(4)
        ]

    @tool_func
    def merge_channels(
        self,
        image_metas,
        colors=None,
        outpath='merge_output.ome.tif',
        preview_path=None,
        preview_seconds: float = 0.0,
    ) -> ImageWithMetadata:
        print("Running function: merge_channels")
        del preview_seconds
        resolved_colors = colors or ['Red', 'Green', 'Blue']
        output_path = Path(self.output_directory, outpath).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=True)
        self._storagemanger.register_file(output_path.name, f"Image after merging channels {resolved_colors}", 'analysis_platform', 'tiff', False)
        if preview_path:
            preview_output_path = Path(self.output_directory, preview_path).expanduser().resolve()
            preview_output_path.parent.mkdir(parents=True, exist_ok=True)
            preview_output_path.touch(exist_ok=True)
            self._storagemanger.register_file(
                preview_output_path.name,
                f"Preview image after merging channels {resolved_colors}",
                'analysis_platform',
                preview_output_path.suffix.lstrip(".") or "png",
                False,
            )
        source = self._coerce_first_image_meta(image_metas)
        return self._clone_image_meta(source, dataset=f"mock_merged_{'_'.join(resolved_colors)}")

    @tool_func
    def set_lut(self, image_meta, color_name) -> ImageWithMetadata:
        print("Running function: set_lut")
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        return self._clone_image_meta(source, dataset=f"mock_lut_{color_name}_{source.dataset}")

    def _temp_tiff(self, img):
        mock_path = f"/tmp/mock_{img}.tif"
        return mock_path

    @tool_func
    def richardson_lucy(self, image_meta, magnification: int, iterations: int = 50,
                        out_filename: str = "deconvolved_result",
                        out_dir: str = "E:/desk/LLM-MICRO") -> ImageWithMetadata:
        print("Running function: richardson_lucy")
        del magnification, iterations, out_dir
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        return self._clone_image_meta(source, dataset=f"mock_deconvolved_{out_filename}")

    @tool_func
    def denoise(self, image_meta, method="Gaussian", radius=2.0) -> ImageWithMetadata:
        print("Running function: denoise")
        del radius
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        return self._clone_image_meta(source, dataset=f"mock_denoised_{method}_{source.dataset}")

    @tool_func
    def fiji_shutdown(self):
        print("Running function: fiji_shutdown")
        self.ij = None
        self.initialized = False
        return True

    @tool_func
    def z_projection(self, image_meta, method="max") -> ImageWithMetadata:
        print("Running function: z_projection")
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        return self._clone_image_meta(source, dataset=f"mock_proj_{method}_{source.dataset}")

    @tool_func
    def quantify_fluorescence(self, image_meta) -> float:
        print("Running function: quantify_fluorescence")
        del image_meta
        return 123.45

    @tool_func
    def analysis_platform_find_target_positions(
        self,
        image_meta,
        target_type: str,
        description: str,
    ) -> List[Tuple[int, int, int, int]]:
        print(f"Running function: analysis_platform_find_target_positions for {target_type}")
        filename = f"{target_type}_locations_list.json"
        del image_meta
        target = str(target_type).lower()
        if target == "tumor":
            regions = [(10, 20, 30, 40), (60, 80, 25, 25)]
        elif target == "organoid":
            regions = [(15, 15, 20, 20), (45, 35, 18, 18)]
        elif target == "lesion":
            regions = [(20, 30, 40, 50)]
        elif target == "bacteria":
            regions = [(5, 5, 12, 12), (25, 18, 10, 10), (40, 33, 8, 8)]
        elif target == "bloodvessel":
            regions = [(30, 10, 60, 12), (22, 48, 44, 10)]
        else:
            regions = [(12, 12, 16, 16)]
        self._storagemanger.register_file(filename, description, 'analysis_platform', 'json')
        return regions

    @tool_func
    def analysis_platform_find_tumor_position(self, image_meta, description: str) -> List[
        Tuple[int, int, int, int]]:
        return self.analysis_platform_find_target_positions(image_meta, "tumor", description)

    @tool_func
    def analysis_platform_find_organoid_position(self, image_meta, description: str) -> List[
        Tuple[int, int, int, int]]:
        return self.analysis_platform_find_target_positions(image_meta, "organoid", description)

    @tool_func
    def analysis_platform_find_lesion_position(self, image_meta, description: str) -> List[
        Tuple[int, int, int, int]]:
        return self.analysis_platform_find_target_positions(image_meta, "lesion", description)

    @tool_func
    def analysis_platform_find_bacteria_position(self, image_meta, description: str) -> List[
        Tuple[int, int, int, int]]:
        return self.analysis_platform_find_target_positions(image_meta, "bacteria", description)

    @tool_func
    def analysis_platform_find_2Dcell_position(self, image_meta, description: str) -> List[
        Tuple[int, int, int, int]]:
        return self.analysis_platform_find_target_positions(image_meta, "2Dcell", description)

    @tool_func
    def analysis_platform_find_BloodVessel_position(self, image_meta, description: str) -> List[
        Tuple[int, int, int, int]]:
        return self.analysis_platform_find_target_positions(image_meta, "BloodVessel", description)

    @tool_func
    def convert_to_numpy(self, image_meta) -> np.ndarray:
        print("Running function: convert_to_numpy")
        del image_meta
        return np.full((256, 256), fill_value=7, dtype=np.uint8)

    def _coerce_image_meta(self, value: Any) -> ImageWithMetadata:
        if isinstance(value, ImageWithMetadata):
            return value
        return ImageWithMetadata(
            dataset=value,
            center_x_um=0.0,
            center_y_um=0.0,
            center_z_um=0.0,
            pixel_size_x_um=1.0,
            pixel_size_y_um=1.0,
        )

    def _coerce_first_image_meta(self, image_metas: Any) -> ImageWithMetadata:
        if isinstance(image_metas, ImageWithMetadata):
            return image_metas
        if isinstance(image_metas, Sequence) and image_metas:
            return self._coerce_image_meta(image_metas[0])
        return self._coerce_image_meta("mock_empty_dataset")

    def _clone_image_meta(self, image_meta: ImageWithMetadata, *, dataset: Any) -> ImageWithMetadata:
        return ImageWithMetadata(
            dataset=dataset,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            center_z_um=image_meta.center_z_um,
            pixel_size_x_um=image_meta.pixel_size_x_um,
            pixel_size_y_um=image_meta.pixel_size_y_um,
        )
    



