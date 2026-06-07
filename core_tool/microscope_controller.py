import logging
import math
from contextlib import contextmanager
import os
from pathlib import Path
import threading
import time
from datetime import datetime
from queue import Queue, Empty
from typing import Any, List, Dict, Optional, Tuple
import cv2
import numpy as np
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.writers import OmeTiffWriter
from ome_types.model import Plane
from pymmcore_plus import CMMCorePlus
from core_tool import tool_utils
import signal
import json

try:
    from mmdet.apis import init_detector, inference_detector
except Exception:
    init_detector = None
    inference_detector = None

try:
    import torch
except Exception:
    torch = None

from config.system_config import objective_labels, dichroic_colors
from config.system_config import (
    camera_device, 
    xy_stage_device,
    objective_device,
    transmittedIllumination,
    focus_drive,
    Dichroic,
    Max_X_position,
    Min_X_position,
    Max_Y_position,
    Min_Y_position,
    Max_Z_position,
    Min_Z_position,
    Max_brightness,
    Min_brightness,
    Max_exposure,
    Min_exposure
)

from config.system_config import get_detection_targets

logger = logging.getLogger(__name__)

TARGET_MODEL_MAP = {
    str(target_name): (
        str(spec.get("model_config", "")),
        str(spec.get("model_checkpoint", "")),
    )
    for target_name, spec in get_detection_targets().items()
}

global_controller = None


@contextmanager
def _silence_native_stdio():
    """Temporarily silence native stdout/stderr noise from MMCore calls."""
    try:
        devnull = open(os.devnull, "w")
    except OSError:
        yield
        return

    try:
        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
    except OSError:
        devnull.close()
        yield
        return

    try:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        try:
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
        finally:
            os.close(old_stdout)
            os.close(old_stderr)
            devnull.close()


def _configure_core_logging(core: Any) -> None:
    """Best-effort suppression of MMCore debug/stderr logging across versions."""
    actions = (
        ("enableDebugLog", (False,)),
        ("enableStderrLog", (False,)),
        ("setPrimaryLogFile", (os.devnull,)),
    )
    for method_name, args in actions:
        method = getattr(core, method_name, None)
        if method is None:
            continue
        try:
            method(*args)
        except Exception:
            logger.debug("Failed to configure MMCore logging via %s", method_name, exc_info=True)


def _coerce_detection_image_to_2d(image: np.ndarray) -> np.ndarray:
    """Accept a 2D image or a singleton multidimensional acquisition result."""
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return image_array

    squeezed = np.squeeze(image_array)
    if squeezed.ndim == 2:
        return squeezed

    raise ValueError("Only 2D grayscale image or singleton multidimensional image supported")


def _to_numpy_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _resolve_model_classes(model: Any) -> list[str]:
    dataset_meta = getattr(model, "dataset_meta", None)
    if isinstance(dataset_meta, dict):
        classes = dataset_meta.get("classes")
        if classes:
            return list(classes)

    classes = getattr(model, "CLASSES", None)
    if classes:
        return list(classes)

    raise RuntimeError("MMDetection model does not expose class metadata")


def _extract_class_detections(det_results: Any, class_idx: int) -> np.ndarray:
    if hasattr(det_results, "pred_instances"):
        pred = det_results.pred_instances
        bboxes = _to_numpy_array(getattr(pred, "bboxes", None))
        scores = _to_numpy_array(getattr(pred, "scores", None))
        labels = _to_numpy_array(getattr(pred, "labels", None)).astype(int, copy=False)

        if bboxes.size == 0 or scores.size == 0 or labels.size == 0:
            return np.empty((0, 5), dtype=np.float32)

        score_mask = labels == class_idx
        if not np.any(score_mask):
            return np.empty((0, 5), dtype=np.float32)

        filtered_boxes = np.asarray(bboxes[score_mask], dtype=np.float32)
        filtered_scores = np.asarray(scores[score_mask], dtype=np.float32).reshape(-1, 1)
        return np.concatenate([filtered_boxes, filtered_scores], axis=1)

    if isinstance(det_results, (list, tuple)):
        if class_idx >= len(det_results):
            return np.empty((0, 5), dtype=np.float32)
        class_dets = np.asarray(det_results[class_idx], dtype=np.float32)
        if class_dets.size == 0:
            return np.empty((0, 5), dtype=np.float32)
        return class_dets.reshape(-1, 5)

    raise RuntimeError(f"Unsupported MMDetection result type: {type(det_results).__name__}")


def _validate_loaded_devices(loaded_devices: Any, required_devices: list[str]) -> None:
    loaded = {str(device) for device in loaded_devices}
    missing = [device for device in required_devices if device not in loaded]
    if missing:
        raise RuntimeError(f"Core devices not loaded: {missing}")

def signal_handler(sig, frame):
    if global_controller:
        global_controller.shutdown_event.set()
        global_controller.shutdown()

def _coerce_brightness_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.size == 0:
        raise ValueError("Cannot evaluate brightness from an empty image")
    arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3].mean(axis=-1)
    return arr.astype(np.float64, copy=False)


def brightness_metrics(
    image: np.ndarray,
    *,
    intensity_max: Optional[float] = None,
    high_percentile: float = 99.5,
    dark_threshold: float = 0.05,
    saturation_threshold: float = 0.98,
) -> Dict[str, float]:
    arr = _coerce_brightness_image(image)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Cannot evaluate brightness from an image without finite pixels")

    if intensity_max is None or intensity_max <= 0:
        if np.issubdtype(np.asarray(image).dtype, np.integer):
            intensity_max = float(np.iinfo(np.asarray(image).dtype).max)
        else:
            intensity_max = max(float(np.max(finite)), 1.0)

    normalized = np.clip(finite / float(intensity_max), 0.0, 1.0)
    return {
        "p50": float(np.percentile(normalized, 50)),
        "p95": float(np.percentile(normalized, 95)),
        "p_high": float(np.percentile(normalized, high_percentile)),
        "dark_ratio": float(np.mean(normalized <= dark_threshold)),
        "saturation_ratio": float(np.mean(normalized >= saturation_threshold)),
    }


def _build_z_positions(z_start: float, z_end: float, z_step: float) -> np.ndarray:
    num_steps = 1 if z_start == z_end else int(round((z_end - z_start) / z_step)) + 1
    return np.linspace(z_start, z_end, num_steps)


def _generate_well_positions(
    rows: int,
    cols: int,
    x_spacing_um: float,
    y_spacing_um: float,
    *,
    origin_x_um: float = 0.0,
    origin_y_um: float = 0.0,
) -> List[Tuple[float, float]]:
    positions: List[Tuple[float, float]] = []
    for row_idx in range(rows):
        for col_idx in range(cols):
            current_x_um = origin_x_um + (col_idx * x_spacing_um)
            current_y_um = origin_y_um + (row_idx * y_spacing_um)
            positions.append((current_x_um, current_y_um))
    return positions

# ===================== ImagingData Class (No Modifications) =====================
class ImagingData:
    """
    Image data encapsulation class for storing formal acquisition images and corresponding metadata
    (center coordinates, objective magnification)
    """
    def __init__(self, image: np.ndarray, center_x: float, center_y: float, center_z: float, objective_magnification: str, pixel_size: Optional[float] = None):
        self.image = image
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.objective_magnification = objective_magnification
        self.pixel_size = pixel_size
        self.position_name = ""

    def __repr__(self):
        """
        Formatted output when printing the instance for easy debugging
        """
        return (f"ImagingData(position_name={self.position_name}, center_xyz=({self.center_x:.1f}, {self.center_y:.1f}, {self.center_z:.1f}) μm, "
                f"objective={self.objective_magnification}, image_shape={self.image.shape}, "
                f"image_dtype={self.image.dtype})")

# ===================== MicroscopeController Class (Key Modification: run_acquisition) =====================
from tool.base import BaseTool, tool_func

class MicroscopeController(BaseTool):
    def __init__(self, config_path: str, app_dir: str, output_path: str, storagemanger):
        self._storagemanger = storagemanger
        self.app_dir = app_dir
        self.config_path = config_path
        with _silence_native_stdio():
            self.core = CMMCorePlus()
        _configure_core_logging(self.core)
        self.device_lock = threading.RLock()
        self.camera_device = camera_device
        self.xy_stage_device = xy_stage_device
        self.objective_device = objective_device
        self.transmittedIllumination = transmittedIllumination
        self.focus_drive = focus_drive
        self.Dichroic = Dichroic

        # Axis ranges
        self.Max_X_position = Max_X_position
        self.Min_X_position = Min_X_position
        self.Max_Y_position = Max_Y_position
        self.Min_Y_position = Min_Y_position
        self.Max_Z_position = Max_Z_position
        self.Min_Z_position = Min_Z_position
        self.Max_brightness = Max_brightness
        self.Min_brightness = Min_brightness
        self.Max_exposure = Max_exposure
        self.Min_exposure = Min_exposure

        # Current state
        self.current_channel = ''
        self.current_objective = ''
        self.current_X_position = 0
        self.current_Y_position = 0
        self.current_Z_position = 0
        self.current_brightness = 0
        self.current_exposure_time = 0
        self._user_brightness = 0

        # Auto acquisition parameters
        self.acquisition_positions: List[Dict] = []
        self.acquisition_channels: List[Dict] = []
        self.z_stack_params: Optional[Dict] = None
        self.time_lapse_params: Optional[Dict] = None
        self.output_directory: str = output_path

        # Image parameters
        self.pixel_size = 0.0
        self.img_dtype = None
        self.current_img_height = 0
        self.current_img_width = 0
        self.is_16bit = False

        # Preview related
        self.preview_running = False
        self.preview_auto_restart_enabled = True
        self.acquisition_thread = None
        self.image_queue = Queue(maxsize=5)  # Only stores image arrays for display
        self.preview_window_name = "micro live"
        self.is_continuous = False
        self.shutdown_event = threading.Event()
        self.img_lock = threading.Lock()
        self.img = None
        self.latest_display_frame: Optional[np.ndarray] = None
        self.last_preview_frame_at: Optional[float] = None
        self.preview_started_at: Optional[float] = None
        self.last_preview_error: str = ""
        self._preview_auto_shutter_original: Optional[bool] = None
        self._preview_shutter_forced_open = False

        self.acquisition_running = False
        self.acquisition_abort = False

        # Auto contrast
        self.auto_contrast_enabled = True
        self.contrast_percentile = 0.1

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

    def _clear_image_queue(self) -> None:
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except Empty:
                break

    def _get_auto_shutter_state(self) -> Optional[bool]:
        getter = getattr(self.core, "getAutoShutter", None)
        if getter is None:
            return None
        try:
            return bool(getter())
        except Exception:
            return None

    def _set_auto_shutter_state(self, enabled: bool) -> None:
        setter = getattr(self.core, "setAutoShutter", None)
        if setter is None:
            return
        try:
            setter(bool(enabled))
        except Exception:
            logger.debug("Failed to set MMCore auto shutter=%s", enabled, exc_info=True)

    def _set_shutter_open_state(self, opened: bool) -> None:
        setter = getattr(self.core, "setShutterOpen", None)
        if setter is None:
            return
        try:
            setter(bool(opened))
        except Exception:
            logger.debug("Failed to set MMCore shutter open=%s", opened, exc_info=True)

    def _prepare_preview_shutter(self) -> None:
        if self._preview_auto_shutter_original is None:
            self._preview_auto_shutter_original = self._get_auto_shutter_state()
        if self._preview_auto_shutter_original is not None:
            self._set_auto_shutter_state(False)
        self._set_shutter_open_state(True)
        self._preview_shutter_forced_open = True

    def _restore_preview_shutter(self) -> None:
        if self._preview_shutter_forced_open:
            self._set_shutter_open_state(False)
            self._preview_shutter_forced_open = False
        if self._preview_auto_shutter_original is not None:
            self._set_auto_shutter_state(self._preview_auto_shutter_original)
            self._preview_auto_shutter_original = None

    def _capture_device_state(self) -> Dict[str, Any]:
        return self._capture_runtime_state(include_xy=True, include_preview=False)

    def _restore_device_state(self, state: Dict[str, Any]) -> None:
        self._restore_runtime_state(state, restore_xy=True, restore_preview=False)

    def _capture_runtime_state(
        self,
        *,
        include_xy: bool,
        include_preview: bool,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "z": self.get_z_position(),
            "channel": self.get_channel(),
            "exposure": self.get_exposure(),
            "transmitted_brightness": self.get_brightness(),
            "brightfield_memory": int(self._user_brightness),
        }
        if include_xy:
            x_pos, y_pos = self.get_x_y_position()
            state["x"] = x_pos
            state["y"] = y_pos
        if include_preview:
            state["preview_running"] = bool(self.preview_running)
        return state

    def _restore_runtime_state(
        self,
        state: Dict[str, Any],
        *,
        restore_xy: bool,
        restore_preview: bool,
    ) -> None:
        target_preview_running = bool(state.get("preview_running", False))
        if restore_preview and target_preview_running and not self.preview_running:
            self.start_preview()
        if restore_xy and "x" in state and "y" in state:
            self.set_x_y_position(state["x"], state["y"])
        self.set_z_position(state["z"])
        self._user_brightness = self._clamp_brightness(state.get("brightfield_memory", self._user_brightness))
        self.set_channel(state["channel"])
        self.set_exposure(state["exposure"])
        if state["channel"] == "1-NONE":
            self._set_transmitted_brightness(state["transmitted_brightness"])
        else:
            self._set_transmitted_brightness(0)
        if restore_preview and not target_preview_running and self.preview_running:
            self.stop_preview()

    def _reset_acquisition_plan(self) -> None:
        self.acquisition_positions.clear()
        self.acquisition_channels.clear()
        self.z_stack_params = None
        self.time_lapse_params = None

    def initialize(self):
        with _silence_native_stdio():
            self.core.reset()
            self.core.unloadAllDevices()
        time.sleep(1.0)

        if self.app_dir and self.app_dir not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + self.app_dir

        with self.device_lock:
            self.core.loadSystemConfiguration(self.config_path)

        loaded_devices = self.core.getLoadedDevices()
        required_devices = [
            self.camera_device,
            self.xy_stage_device,
            self.objective_device,
            self.focus_drive,
            self.transmittedIllumination,
            self.Dichroic,
        ]
        _validate_loaded_devices(loaded_devices, required_devices)

        self.core.setCameraDevice(self.camera_device)
        self.core.waitForSystem()
        time.sleep(4.0)
        self.core.waitForDevice(self.camera_device)

        with self.device_lock:
            self.core.startContinuousSequenceAcquisition(0)
            time.sleep(1.0)
            while self.core.getRemainingImageCount() > 0:
                self.core.getLastImage()
            self.core.stopSequenceAcquisition()

        # Test acquisition (using formal acquisition method to get ImagingData)
        test_imaging_data = None
        for _ in range(3):
            try:
                test_imaging_data = self._acquire_single_image()
                if test_imaging_data is not None and test_imaging_data.image.size > 0 and len(test_imaging_data.image.shape) == 2:
                    break
                time.sleep(1.0)
            except Exception as e:
                test_imaging_data = None

        if test_imaging_data is None:
            raise RuntimeError("Initialization acquisition failed! Please check camera connection and configuration")

        self.current_img_height, self.current_img_width = test_imaging_data.image.shape
        self.img_dtype = test_imaging_data.image.dtype
        self.is_16bit = (test_imaging_data.image.dtype == np.uint16)

        self.current_objective = self.get_objective()
        if self.current_objective not in objective_labels:
            raise RuntimeError(f"Objective not configured: {self.current_objective}")
        self.pixel_size = 1.6234 * 4 / objective_labels[self.current_objective]

        self.current_X_position, self.current_Y_position = self.get_x_y_position()
        self.current_Z_position = self.get_z_position()
        self.current_channel = self.get_channel()
        self.current_brightness = self.get_brightness()
        self._user_brightness = self.current_brightness

    # ====== Device Control (No Modifications) ======
    @tool_func
    def set_x_y_position(self, x: float, y: float):
        if not (self.Min_X_position - 10 <= x <= self.Max_X_position + 10 and
                self.Min_Y_position - 10 <= y <= self.Max_Y_position + 10):
            raise ValueError("XY position out of range")
        if abs(x - self.current_X_position) < 1 and abs(y - self.current_Y_position) < 1:
            return
        self.core.setXYStageDevice(self.xy_stage_device)
        self.core.setXYPosition(x, y)
        self.core.waitForDevice(self.xy_stage_device)
        with self.device_lock:
            self.current_X_position, self.current_Y_position = x, y
    @tool_func
    def get_x_y_position(self) -> Tuple[float, float]:
        x, y = self.core.getXYPosition()
        with self.device_lock:
            self.current_X_position, self.current_Y_position = x, y
        return x, y
    @tool_func
    def set_z_position(self, z: float):
        if not (self.Min_Z_position - 1 <= z <= self.Max_Z_position + 1):
            raise ValueError("Z position out of range")
        if abs(z - self.current_Z_position) < 0.5:
            return
        self.core.setFocusDevice(self.focus_drive)
        self.core.setPosition(z)
        self.core.waitForDevice(self.focus_drive)
        with self.device_lock:
            self.current_Z_position = z
    @tool_func
    def get_z_position(self) -> float:
        z = self.core.getPosition(self.focus_drive)
        with self.device_lock:
            self.current_Z_position = z
        return z
    @tool_func
    def set_exposure(self, exposure_time: float):
        if exposure_time == self.current_exposure_time:
            return
        with self.device_lock:
            was_continuous = self.is_continuous
            if was_continuous:
                self.core.stopSequenceAcquisition()
                self.is_continuous = False
            try:
                exposure_time = max(self.Min_exposure, min(exposure_time, self.Max_exposure))
                self.core.setProperty(self.camera_device, 'Exposure', exposure_time)
                self.core.waitForDevice(self.camera_device)
                self.current_exposure_time = exposure_time
            finally:
                if was_continuous and self.preview_running:
                    self.core.startContinuousSequenceAcquisition(0)
                    self.is_continuous = True
    @tool_func
    def get_exposure(self) -> float:
        exp = self.core.getProperty(self.camera_device, "Exposure")
        with self.device_lock:
            self.current_exposure_time = float(exp)
        return float(exp)

    def _clamp_brightness(self, brightness: int) -> int:
        return int(max(self.Min_brightness, min(int(brightness), self.Max_brightness)))

    def _set_transmitted_brightness(self, brightness: int) -> int:
        brightness = self._clamp_brightness(brightness)
        self.core.setProperty(self.transmittedIllumination, 'Brightness', brightness)
        self.core.waitForDevice(self.transmittedIllumination)
        actual_brightness = int(self.core.getProperty(self.transmittedIllumination, 'Brightness'))
        with self.device_lock:
            self.current_brightness = actual_brightness
        return actual_brightness

    def remember_brightfield_brightness(self, brightness: int) -> None:
        self._user_brightness = self._clamp_brightness(brightness)
        if self.get_channel() == '1-NONE':
            self._set_transmitted_brightness(self._user_brightness)

    @tool_func
    def set_brightness(self, brightness: int):
        current_channel = self.get_channel()
        is_brightfield = (current_channel == '1-NONE')
        if is_brightfield:
            brightness = self._clamp_brightness(brightness)
            self._user_brightness = brightness
        else:
            brightness = 0
        self._set_transmitted_brightness(brightness)
    @tool_func
    def get_brightness(self) -> int:
        bright = self.core.getProperty(self.transmittedIllumination, 'Brightness')
        with self.device_lock:
            self.current_brightness = int(bright)
        return int(bright)
    @tool_func
    def set_objective(self, objective_label: str):
        if objective_label not in self.core.getStateLabels(self.objective_device):
            raise ValueError(f"Unsupported objective: {objective_label}")
        self.core.setStateLabel(self.objective_device, objective_label)
        self.core.waitForDevice(self.objective_device)
        with self.device_lock:
            self.current_objective = objective_label
            self.pixel_size = 1.6234 * 4 / objective_labels[self.current_objective]
    @tool_func
    def get_objective(self) -> str:
        with self.device_lock:
            self.current_objective = self.core.getStateLabel(self.objective_device)
        return self.current_objective
    @tool_func
    def set_channel(self, channel: str):
        supported = self.core.getStateLabels(self.Dichroic)
        if channel not in supported:
            raise ValueError(f"Unsupported channel: {channel}, Available options: {supported}")
        previous_channel = self.get_channel()
        if previous_channel == '1-NONE':
            self._user_brightness = self._clamp_brightness(self.get_brightness())
        self.core.setStateLabel(self.Dichroic, channel)
        self.core.waitForDevice(self.Dichroic)
        with self.device_lock:
            self.current_channel = channel
        if channel == '1-NONE':
            self._set_transmitted_brightness(self._user_brightness)
        else:
            self._set_transmitted_brightness(0)
    @tool_func
    def get_channel(self) -> str:
        with self.device_lock:
            self.current_channel = self.core.getStateLabel(self.Dichroic)
        return self.current_channel

    # ====== Real-time Preview ======
    def start_preview(self):
        if self.preview_running and self.acquisition_thread and self.acquisition_thread.is_alive():
            return
        if self.preview_running:
            logger.warning("Preview flag was still enabled, but the acquisition thread was not alive. Restarting preview.")

        self.preview_running = True
        self.shutdown_event.clear()
        with self.img_lock:
            self.latest_display_frame = None
            self.last_preview_frame_at = None
        self.last_preview_error = ""
        self.preview_started_at = time.monotonic()

        with self.device_lock:
            self._prepare_preview_shutter()

        self.acquisition_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.acquisition_thread.start()

        print("Preview acquisition started")

    @tool_func
    def stop_preview(self):
        """Stop preview safely by shutting down worker threads and cleaning up resources."""
        if not self.preview_running:
            return

        self.preview_running = False
        self.shutdown_event.set()

        with self.device_lock:
            if self.is_continuous:
                self.core.stopSequenceAcquisition()
                self.is_continuous = False
            self._restore_preview_shutter()

        if (
            self.acquisition_thread
            and self.acquisition_thread.is_alive()
            and threading.current_thread() is not self.acquisition_thread
        ):
            self.acquisition_thread.join(timeout=1.0)

        self._clear_image_queue()
        with self.img_lock:
            self.latest_display_frame = None
            self.last_preview_frame_at = None

        print("Preview acquisition stopped")

    def _acquisition_loop(self):
        unexpected_exit = False
        try:
            with self.device_lock:
                self.core.startContinuousSequenceAcquisition(0)
                self.is_continuous = True
            while self.preview_running and not self.shutdown_event.is_set():
                with self.device_lock:
                    if self.core.getRemainingImageCount() == 0:
                        time.sleep(0.01)
                        continue
                    img = self.core.getLastImage()
                    if img is None:
                        continue
                with self.img_lock:
                    self.img = img.copy()
                self._publish_preview_frame(img)
                time.sleep(0.01)
        except Exception as exc:
            unexpected_exit = True
            self.last_preview_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Preview acquisition loop failed")
        finally:
            if unexpected_exit:
                self.preview_running = False
                self.shutdown_event.set()
                with self.img_lock:
                    self.latest_display_frame = None
            with self.device_lock:
                if self.is_continuous:
                    self.core.stopSequenceAcquisition()
                    self.is_continuous = False
                if not self.preview_running:
                    self._restore_preview_shutter()
            if unexpected_exit:
                logger.warning("Preview acquisition loop exited unexpectedly: %s", self.last_preview_error)

    def _process_image_for_display(self, img):
        try:
            if self.auto_contrast_enabled:
                low, high = np.percentile(img, [self.contrast_percentile, 100 - self.contrast_percentile])
                img = np.clip(img, low, high)
                img = (img - low) / (high - low + 1e-8)
            else:
                img = img / np.max(img) if np.max(img) > 0 else img

            if self.is_16bit:
                display_img = (img * 255).astype(np.uint8)
            else:
                display_img = (img * 255).astype(np.uint8) if img.dtype != np.uint8 else img

            color = dichroic_colors.get(self.current_channel, (128, 128, 128))
            if color != (128, 128, 128):
                r = (color[0] * (display_img / 255)).astype(np.uint8)
                g = (color[1] * (display_img / 255)).astype(np.uint8)
                b = (color[2] * (display_img / 255)).astype(np.uint8)
                display_img = cv2.merge([b, g, r])
            else:
                display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            display_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        return display_img

    def _publish_preview_frame(self, img: np.ndarray) -> None:
        """Refresh the preview cache from an arbitrary raw frame."""
        processed_img = self._process_image_for_display(img.copy())
        with self.img_lock:
            self.latest_display_frame = processed_img.copy()
            self.last_preview_frame_at = time.monotonic()
        self.last_preview_error = ""
        if self.image_queue.full():
            try:
                self.image_queue.get_nowait()
            except Empty:
                pass
        try:
            self.image_queue.put(processed_img, timeout=0.01)
        except Exception:
            logger.debug("Preview image queue is full; dropping an old frame", exc_info=True)

    def get_live_preview_image(self) -> Optional[np.ndarray]:
        """Only returns image array for preview display, no metadata"""
        if not self.preview_running:
            return None
        try:
            with self.img_lock:
                if self.latest_display_frame is not None:
                    return self.latest_display_frame.copy()
        except Exception:
            return None
        return None

    # ====== Image Acquisition (No Modifications, Formal Acquisition Returns ImagingData) ======
    def _get_image(self, width_micro=None, height_micro=None) -> ImagingData:
        if width_micro and height_micro:
            return self._acquire_stitch_mosaic(width_micro, height_micro)
        else:
            return self._acquire_single_image()

    def _snap_raw_image(self) -> np.ndarray:
        with self._acquisition_guard():
            with self.device_lock:
                was_continuous = self.is_continuous
                if was_continuous:
                    self.core.stopSequenceAcquisition()
                    self.is_continuous = False
                try:
                    self.core.snapImage()
                    img = self.core.getImage()
                    if img is None:
                        raise RuntimeError("Acquisition failed")
                    return img.copy()
                finally:
                    if was_continuous:
                        self.core.startContinuousSequenceAcquisition(0)
                        self.is_continuous = True

    def _acquire_single_image(self) -> ImagingData:
        """Formal single image acquisition: returns a synchronized raw image with metadata."""
        img = self._snap_raw_image()
        current_x, current_y = self.get_x_y_position()
        current_z = self.get_z_position()
        current_obj = self.get_objective()
        return ImagingData(
            image=img,
            center_x=current_x,
            center_y=current_y,
            center_z=current_z,
            objective_magnification=current_obj,
            pixel_size=self.pixel_size,
        )

    def _snap_image_preserving_preview(self) -> np.ndarray:
        """Capture a synchronized raw image for feedback workflows while preserving preview state."""
        img = self._snap_raw_image()
        if self.preview_running:
            self._publish_preview_frame(img)
        return img

    def _get_image_intensity_max(self, image: np.ndarray) -> float:
        try:
            bit_depth = int(self.core.getImageBitDepth())
            if bit_depth > 0:
                return float((1 << bit_depth) - 1)
        except Exception:
            pass
        image_array = np.asarray(image)
        if np.issubdtype(image_array.dtype, np.integer):
            return float(np.iinfo(image_array.dtype).max)
        return max(float(np.nanmax(image_array)), 1.0)

    def _acquire_stitch_mosaic(self, width_micro: float, height_micro: float, overlap=0) -> ImagingData:
        """Formal stitched acquisition: returns ImagingData with metadata"""
        with self._acquisition_guard():
            initial_x, initial_y = self.get_x_y_position()
            initial_z = self.get_z_position()
            current_obj = self.get_objective()
            fov_width = self.current_img_width * self.pixel_size
            fov_height = self.current_img_height * self.pixel_size

            step_x = fov_width * (1 - overlap)
            step_y = fov_height * (1 - overlap)

            min_cols = max(1, math.ceil(width_micro / step_x))
            min_rows = max(1, math.ceil(height_micro / step_y))
            cols = min_cols + 1 if min_cols % 2 == 0 else min_cols
            rows = min_rows + 1 if min_rows % 2 == 0 else min_rows

            center_col = cols // 2
            center_row = rows // 2
            start_x = initial_x - center_col * step_x - fov_width / 2
            start_y = initial_y - center_row * step_y - fov_height / 2

            if (start_x < self.Min_X_position or start_y < self.Min_Y_position or
                start_x + (cols - 1) * step_x + fov_width > self.Max_X_position or
                start_y + (rows - 1) * step_y + fov_height > self.Max_Y_position):
                raise ValueError("Stitching area out of range")

            mosaic = np.zeros((self.current_img_height * rows, self.current_img_width * cols), dtype=self.img_dtype)

            for y_idx in range(rows):
                y_pos = start_y + y_idx * step_y + fov_height / 2
                x_indices = range(cols) if y_idx % 2 == 0 else reversed(range(cols))
                for x_idx in x_indices:
                    x_pos = start_x + x_idx * step_x + fov_width / 2
                    self.set_x_y_position(x_pos, y_pos)
                    # Call formal acquisition method to get image with metadata
                    imaging_data = self._acquire_single_image()
                    img = imaging_data.image
                    y_start = y_idx * self.current_img_height
                    x_start = x_idx * self.current_img_width
                    mosaic[y_start:y_start + self.current_img_height, x_start:x_start + self.current_img_width] = img
            self.set_x_y_position(initial_x, initial_y)
            # Return ImagingData of stitched image
            return ImagingData(
                image=mosaic,
                center_x=initial_x,
                center_y=initial_y,
                center_z=initial_z,
                objective_magnification=current_obj,
                pixel_size=self.pixel_size
            )

    def _get_effective_time_lapse_params(self) -> Dict[str, float]:
        if self.time_lapse_params:
            return dict(self.time_lapse_params)
        return {"num_frames": 1, "interval_sec": 0}

    def _get_effective_z_stack_params(self) -> Dict[str, float]:
        if self.z_stack_params:
            return dict(self.z_stack_params)
        current_z = self.get_z_position()
        return {"z_start": current_z, "z_end": current_z, "z_step": 1}

    def _get_acquisition_settle_time(self) -> float:
        return 0.10

    def _ensure_acquisition_image_spec(self) -> None:
        if self.current_img_height > 0 and self.current_img_width > 0 and self.img_dtype is not None:
            return
        sample = self._acquire_single_image()
        image = np.asarray(sample.image)
        if image.ndim != 2:
            raise RuntimeError("Formal acquisition image must be a 2D grayscale array")
        self.current_img_height, self.current_img_width = image.shape
        self.img_dtype = image.dtype
        if sample.pixel_size is not None:
            self.pixel_size = sample.pixel_size

    def _calculate_stitch_grid(
        self,
        width_micro: float,
        height_micro: float,
        overlap: float = 0,
    ) -> Dict[str, float]:
        fov_width = self.current_img_width * self.pixel_size
        fov_height = self.current_img_height * self.pixel_size
        if fov_width <= 0 or fov_height <= 0:
            raise RuntimeError("Current image size and pixel size must be initialized before stitch planning")

        step_x = fov_width * (1 - overlap)
        step_y = fov_height * (1 - overlap)

        min_cols = max(1, math.ceil(width_micro / step_x))
        min_rows = max(1, math.ceil(height_micro / step_y))
        cols = min_cols + 1 if min_cols % 2 == 0 else min_cols
        rows = min_rows + 1 if min_rows % 2 == 0 else min_rows
        return {
            "cols": cols,
            "rows": rows,
            "step_x": step_x,
            "step_y": step_y,
            "fov_width": fov_width,
            "fov_height": fov_height,
        }

    def _resolve_position_output_shape(self, position: Dict[str, Any]) -> Tuple[int, int]:
        if position["width"] and position["height"]:
            grid = self._calculate_stitch_grid(position["width"], position["height"])
            return (
                int(self.current_img_height * int(grid["rows"])),
                int(self.current_img_width * int(grid["cols"])),
            )
        return int(self.current_img_height), int(self.current_img_width)

    def _build_position_acquisition_metadata(
        self,
        position: Dict[str, Any],
        *,
        channel_names: List[str],
        time_interval: float,
        z_positions: np.ndarray,
    ) -> Dict[str, Any]:
        reference_z = float(z_positions[0]) if len(z_positions) else float(self.get_z_position())
        return self._create_ome_metadata(
            channel_names=channel_names,
            time_interval=time_interval,
            microscope="olympus lx83",
            objective=self.current_objective,
            pixel_type=self.img_dtype,
            center_x=float(position["x"]),
            center_y=float(position["y"]),
            center_z=reference_z,
        )

    def _build_position_acquisition_record(
        self,
        position: Dict[str, Any],
        *,
        channel_names: List[str],
        time_interval: float,
        num_frames: int,
        z_positions: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        image_height, image_width = self._resolve_position_output_shape(position)
        data = np.zeros(
            (num_frames, len(channel_names), len(z_positions), image_height, image_width),
            dtype=self.img_dtype,
        )
        metadata = self._build_position_acquisition_metadata(
            position,
            channel_names=channel_names,
            time_interval=time_interval,
            z_positions=z_positions,
        )
        return {
            "name": position["name"],
            "metadata": metadata,
            "data": data,
            "x": position["x"],
            "y": position["y"],
            "width": position["width"],
            "height": position["height"],
            "z_positions": np.asarray(z_positions, dtype=float),
            "objective_magnification": self.current_objective,
            "pixel_size": self.pixel_size,
        }

    def _prepare_acquisition_records(
        self,
        *,
        channel_names: List[str],
        time_interval: float,
        num_frames: int,
        z_positions: np.ndarray,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for position in self.acquisition_positions:
            record = self._build_position_acquisition_record(
                position,
                channel_names=channel_names,
                time_interval=time_interval,
                num_frames=num_frames,
                z_positions=z_positions,
            )
            if record is not None:
                records.append(record)
        return records

    def _move_to_acquisition_position(self, position_record: Dict[str, Any]) -> None:
        self.set_x_y_position(position_record["x"], position_record["y"])
        time.sleep(self._get_acquisition_settle_time())

    def _configure_acquisition_channel(self, channel_config: Dict[str, Any]) -> None:
        self.set_channel(channel_config["channel"])
        self.set_exposure(channel_config["exposure"])

    def _capture_acquisition_plane(
        self,
        position_record: Dict[str, Any],
        *,
        z_position: float,
    ) -> ImagingData:
        self.set_z_position(float(z_position))
        time.sleep(self._get_acquisition_settle_time())
        return self._get_image(position_record["width"], position_record["height"])

    def _capture_position_timepoint(
        self,
        position_record: Dict[str, Any],
        *,
        time_index: int,
        z_positions: np.ndarray,
    ) -> None:
        self._move_to_acquisition_position(position_record)
        for channel_index, channel_config in enumerate(self.acquisition_channels):
            self._configure_acquisition_channel(channel_config)
            for z_index, z_position in enumerate(z_positions):
                imaging_data = self._capture_acquisition_plane(
                    position_record,
                    z_position=float(z_position),
                )
                position_record["data"][time_index, channel_index, z_index] = imaging_data.image

    def _save_position_acquisition_result(
        self,
        position_record: Dict[str, Any],
        *,
        pixel_sizes: PhysicalPixelSizes,
        channel_names: List[str],
        num_frames_captured: Optional[int] = None,
    ) -> ImagingData:
        save_path = os.path.join(self.output_directory, f"{position_record['name']}.ome.tif")
        os.makedirs(self.output_directory, exist_ok=True)
        captured_frames = int(num_frames_captured) if num_frames_captured is not None else int(position_record["data"].shape[0])
        if captured_frames < 1:
            raise ValueError("No time-series frames were captured for this acquisition position")
        captured_data = position_record["data"][:captured_frames]
        self._save_ome_tiff(
            captured_data,
            save_path,
            pixel_sizes,
            position_record["metadata"],
            z_positions=position_record.get("z_positions"),
        )

        channel_colors = [dichroic_colors.get(channel, "Unknown") for channel in channel_names]
        objective_magnification = objective_labels.get(self.current_objective)
        desc = (
            f'"channel_names": {channel_colors}, '
            f'pixel_size: {self.pixel_size}, '
            f'"objective_label": {self.current_objective}, '
            f'"magnification": {objective_magnification}'
        )
        self._storagemanger.register_file(
            f"{position_record['name']}.ome.tif",
            desc,
            'microscope',
            'ome-tiff',
        )

        imaging_data = ImagingData(
            image=captured_data,
            center_x=position_record["metadata"]["center_x"],
            center_y=position_record["metadata"]["center_y"],
            center_z=position_record["metadata"]["center_z"],
            objective_magnification=position_record["objective_magnification"],
            pixel_size=position_record.get("pixel_size"),
        )
        imaging_data.position_name = position_record["name"]
        return imaging_data


    # ====== Auto Acquisition (Key Modification: Returns List[ImagingData]) ======
    @tool_func
    def add_acquisition_position(self, name: str, x: float, y: float, width: float, height: float) -> None:
        """Add a stage position to the automatic acquisition queue."""
        self.acquisition_positions.append({
            "name": name,
            "x": x,
            "y": y,
            'width': width,
            'height': height
        })
    @tool_func
    def add_channels(self, channel: str, exposure: float) -> None:
        """Add a channel configuration to the automatic acquisition queue."""
        self.acquisition_channels.append({
            "channel": channel,
            "exposure": exposure
        })
    @tool_func
    def set_z_stack(self, z_start: float, z_end: float, z_step: float) -> None:
        """Configure Z-stack acquisition parameters."""
        if z_step <= 0:
            raise ValueError("Z-stack step size must be positive.")
        if (z_end - z_start) * z_step < 0:
            raise ValueError("Z-stack step direction conflicts with the start/end range.")
        self.z_stack_params = {
            "z_start": z_start,
            "z_end": z_end,
            "z_step": z_step
        }
    @tool_func
    def set_time_series(self, num_frames: int, interval_sec: float) -> None:
        """Configure time-series acquisition parameters."""
        if int(num_frames) < 1:
            raise ValueError("num_frames must be at least 1.")
        if float(interval_sec) < 0:
            raise ValueError("interval_sec must be non-negative.")
        self.time_lapse_params = {
            "num_frames": int(num_frames),
            "interval_sec": float(interval_sec)
        }
    @tool_func
    def run_acquisition(self) -> List[ImagingData]:
        """
        Perform automatic acquisition and return a list of ImagingData containing
        images and metadata for all acquisition positions.
        Returns:
            List[ImagingData]: Each element corresponds to the final image (including
            time series/Z-stack/channel information) and metadata of one acquisition position.
        """
        with self._acquisition_guard():
            acquisition_imaging_data_list = []
            if not self.acquisition_positions:
                raise ValueError("Please add acquisition positions")
            if not self.acquisition_channels:
                raise ValueError("Please configure channels")

            time_lapse_params = self._get_effective_time_lapse_params()
            z_stack_params = self._get_effective_z_stack_params()
            initial_state = self._capture_runtime_state(include_xy=True, include_preview=True)

            time_num_frames = int(time_lapse_params["num_frames"])
            tim_interval = float(time_lapse_params["interval_sec"])
            z_positions = _build_z_positions(
                z_stack_params["z_start"],
                z_stack_params["z_end"],
                z_stack_params["z_step"],
            )
            channels_name = [ch["channel"] for ch in self.acquisition_channels]
            self._ensure_acquisition_image_spec()

            position_data = self._prepare_acquisition_records(
                channel_names=channels_name,
                time_interval=tim_interval,
                num_frames=time_num_frames,
                z_positions=z_positions,
            )
            completed_timepoints = 0

            try:
                for t_idx in range(time_num_frames):
                    if self.shutdown_event.is_set():
                        break
                    start_time = time.time()
                    for pos_item in position_data:
                        self._capture_position_timepoint(
                            pos_item,
                            time_index=t_idx,
                            z_positions=z_positions,
                        )
                    completed_timepoints = t_idx + 1

                    if time_num_frames > 1 and t_idx < time_num_frames - 1:
                        elapsed = time.time() - start_time
                        if elapsed > tim_interval:
                            logger.warning(
                                "Time-series acquisition overran the requested interval: elapsed=%.3fs, requested_interval=%.3fs. "
                                "The next frame will start immediately.",
                                elapsed,
                                tim_interval,
                            )
                        wait_time = max(0, tim_interval - elapsed)
                        time.sleep(wait_time)

                if completed_timepoints < 1:
                    return acquisition_imaging_data_list

                pixel_sizes = PhysicalPixelSizes(
                    Z=z_stack_params["z_step"],
                    Y=self.pixel_size,
                    X=self.pixel_size
                )
                for pos_item in position_data:
                    acquisition_imaging_data_list.append(
                        self._save_position_acquisition_result(
                            pos_item,
                            pixel_sizes=pixel_sizes,
                            channel_names=channels_name,
                            num_frames_captured=completed_timepoints,
                        )
                    )

            except Exception as exc:
                logger.exception("Microscope acquisition failed during run_acquisition")
                raise RuntimeError(f"Microscope acquisition failed: {exc}") from exc
            finally:
                self._reset_acquisition_plan()
                self._restore_runtime_state(initial_state, restore_xy=True, restore_preview=True)

            return acquisition_imaging_data_list

    def _save_ome_tiff(self, data, save_path, pixel_sizes, metadata, *, z_positions=None):
        # Extract center position (if exists)
        center_x = metadata.get("center_x")
        center_y = metadata.get("center_y")
        center_z = metadata.get("center_z")

        # Calculate Position for each plane (assuming all planes share the same XY position, Z can vary with stack)
        # data shape: (T, C, Z, Y, X)
        t, c, z = data.shape[:3]

        # Default: all planes use the same XY center, Z increases from center_z by step
        if z_positions is not None:
            z_positions = [float(value) for value in np.asarray(z_positions).reshape(-1).tolist()]
            if len(z_positions) != z:
                raise ValueError(
                    f"Expected {z} z positions for saved stack metadata, got {len(z_positions)}"
                )
        elif hasattr(pixel_sizes, 'Z') and pixel_sizes.Z not in (None, 0):
            z_positions = [center_z + i * pixel_sizes.Z for i in range(z)] if center_z is not None else [0.0] * z
        else:
            z_positions = [center_z] * z if center_z is not None else [0.0] * z

        # Build plane positions list (order: T, C, Z)
        plane_position_x = []
        plane_position_y = []
        plane_position_z = []
        for ti in range(t):
            for ci in range(c):
                for zi in range(z):
                    plane_position_x.append(center_x if center_x is not None else 0.0)
                    plane_position_y.append(center_y if center_y is not None else 0.0)
                    plane_position_z.append(z_positions[zi])

        ome_xml = OmeTiffWriter.build_ome(
            [data.shape],
            [data.dtype],
            dimension_order=["TCZYX"],
            channel_names=[metadata["channel_names"]],
            physical_pixel_sizes=[pixel_sizes],
            channel_colors=[metadata["channel_colors"]],
            image_name=[Path(save_path).stem],
        )
        planes = []
        for ti in range(t):
            for ci in range(c):
                for zi in range(z):
                    plane_index = (ti * c * z) + (ci * z) + zi
                    planes.append(
                        Plane(
                            the_t=ti,
                            the_c=ci,
                            the_z=zi,
                            position_x=plane_position_x[plane_index],
                            position_y=plane_position_y[plane_index],
                            position_z=plane_position_z[plane_index],
                        )
                    )
        ome_xml.images[0].pixels.planes = planes

        OmeTiffWriter.save(
            data,
            save_path,
            dim_order="TCZYX",
            ome_xml=ome_xml,
        )
    
    def _create_ome_metadata(
        self,
        channel_names: List[str],
        time_interval: float,
        microscope: str,
        objective: str,
        pixel_type: np.dtype,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        center_z: Optional[float] = None,
    ) -> Dict:
        """
        Create OME metadata dictionary, optionally including image center physical position.
        """
        channel_colors = [dichroic_colors.get(ch, (128, 128, 128)) for ch in channel_names]
        metadata = {
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "time_interval": time_interval,
            "microscope": microscope,
            "objective": objective,
            "objective_label": objective,
            "objective_magnification": objective_labels.get(objective),
            "datetime": datetime.now().isoformat(),
            "pixel_type": pixel_type.name
        }

        # If center position is provided, add to metadata (for _save_ome_tiff usage)
        if center_x is not None and center_y is not None:
            metadata["center_x"] = center_x
            metadata["center_y"] = center_y
            metadata["center_z"] = center_z if center_z is not None else 0.0

        return metadata
    
    def _get_autofocus_params_for_magnification(
        self,
        magnification: float,
        is_fluorescence: bool,
    ) -> Dict[str, float]:
        if magnification < 5:
            params = {"search_range": 400.0, "coarse_step": 40.0, "tolerance": 2.0, "max_search_range": 800.0}
        elif magnification < 15:
            params = {"search_range": 220.0, "coarse_step": 25.0, "tolerance": 1.0, "max_search_range": 600.0}
        elif magnification < 30:
            params = {"search_range": 180.0, "coarse_step": 12.0, "tolerance": 0.5, "max_search_range": 500.0}
        elif magnification < 50:
            params = {"search_range": 90.0, "coarse_step": 8.0, "tolerance": 0.5, "max_search_range": 300.0}
        else:
            params = {"search_range": 60.0, "coarse_step": 5.0, "tolerance": 0.5, "max_search_range": 180.0}

        if not is_fluorescence and magnification < 15:
            params["center_roi_size"] = 768.0
        else:
            params["center_roi_size"] = 1024.0
        params["settle_time_sec"] = 0.10 if is_fluorescence else 0.05
        return params

    # ====== Auto Focus / Brightness ======
    @tool_func
    def perform_autofocus(self, tolerance=0.5, use_auto_params=False, search_range=600.0) -> float:
        state = self._capture_runtime_state(include_xy=False, include_preview=True)
        base_center_z = float(state["z"])
        current_channel = self.get_channel()
        current_objective = self.get_objective()
        magnification = float(objective_labels.get(current_objective, 10.0))
        is_fluorescence = current_channel != "1-NONE"
        auto_params = self._get_autofocus_params_for_magnification(magnification, is_fluorescence)
        tolerance = max(float(tolerance), 0.5)
        requested_search_range = max(float(search_range), tolerance)

        if use_auto_params:
            tolerance = max(tolerance, auto_params["tolerance"])
            search_range = max(tolerance, min(requested_search_range, auto_params["search_range"]))
            coarse_step = min(auto_params["coarse_step"], search_range)
            expansion_cap = max(requested_search_range, auto_params["max_search_range"], search_range)
        else:
            search_range = requested_search_range
            coarse_step = max(tolerance * 4.0, min(50.0, search_range))
            expansion_cap = search_range

        center_roi_size = int(auto_params["center_roi_size"])
        settle_time_sec = float(auto_params["settle_time_sec"])
        scores: Dict[float, float] = {}
        autofocus_completed = False

        def score_at(z_position: float, lower_z: float, upper_z: float) -> float:
            z_position = float(max(lower_z, min(z_position, upper_z)))
            cache_key = round(z_position, 4)
            if cache_key in scores:
                return scores[cache_key]
            self.set_z_position(z_position)
            if settle_time_sec > 0:
                time.sleep(settle_time_sec)
            image = self._snap_image_preserving_preview()
            score = float(
                tool_utils.tenengrad_calculate_sharpness(
                    image,
                    center_roi_size=center_roi_size,
                )
            )
            scores[cache_key] = score
            return score

        def search_once(
            search_center_z: float,
            active_search_range: float,
            active_coarse_step: float,
        ) -> Tuple[float, float, float, float]:
            lower_z = max(float(self.Min_Z_position), search_center_z - active_search_range)
            upper_z = min(float(self.Max_Z_position), search_center_z + active_search_range)
            coarse_positions = np.arange(
                lower_z,
                upper_z + active_coarse_step * 0.5,
                active_coarse_step,
                dtype=float,
            )
            if coarse_positions.size == 0:
                coarse_positions = np.array([search_center_z], dtype=float)

            best_z = float(coarse_positions[0])
            best_score = score_at(best_z, lower_z, upper_z)
            for z_position in coarse_positions[1:]:
                score = score_at(float(z_position), lower_z, upper_z)
                if score > best_score:
                    best_score = score
                    best_z = float(z_position)

            step = active_coarse_step / 2.0
            iterations = 0
            while step >= tolerance and iterations < 50:
                improved = False
                for candidate_z in (best_z - step, best_z + step):
                    if not (lower_z <= candidate_z <= upper_z):
                        continue
                    score = score_at(candidate_z, lower_z, upper_z)
                    if score > best_score:
                        best_score = score
                        best_z = float(candidate_z)
                        improved = True
                if not improved:
                    step /= 2.0
                iterations += 1
            return best_z, best_score, lower_z, upper_z

        def is_near_search_boundary(best_z: float, lower_z: float, upper_z: float, active_coarse_step: float) -> bool:
            boundary_margin = max(tolerance, active_coarse_step * 0.5)
            lower_available = lower_z > float(self.Min_Z_position) + tolerance
            upper_available = upper_z < float(self.Max_Z_position) - tolerance
            return (
                lower_available and (best_z - lower_z) <= boundary_margin
            ) or (
                upper_available and (upper_z - best_z) <= boundary_margin
            )

        try:
            active_center_z = base_center_z
            active_search_range = search_range
            active_coarse_step = coarse_step
            expansion_round = 0
            max_expansion_rounds = 4

            best_z, best_score, lower_z, upper_z = search_once(
                active_center_z,
                active_search_range,
                active_coarse_step,
            )
            while (
                use_auto_params
                and active_search_range < expansion_cap
                and expansion_round < max_expansion_rounds
                and is_near_search_boundary(best_z, lower_z, upper_z, active_coarse_step)
            ):
                expanded_range = min(expansion_cap, max(active_search_range * 2.0, active_search_range + active_coarse_step))
                expanded_coarse_step = min(active_coarse_step, expanded_range)
                logger.warning(
                    "Autofocus best Z %.3f is near search boundary [%.3f, %.3f]; "
                    "expanding search range from %.3f to %.3f um around %.3f um",
                    best_z,
                    lower_z,
                    upper_z,
                    active_search_range,
                    expanded_range,
                    best_z,
                )
                active_center_z = best_z
                active_search_range = expanded_range
                active_coarse_step = expanded_coarse_step
                best_z, best_score, lower_z, upper_z = search_once(
                    active_center_z,
                    active_search_range,
                    active_coarse_step,
                )
                expansion_round += 1

            if is_near_search_boundary(best_z, lower_z, upper_z, active_coarse_step):
                logger.warning(
                    "Autofocus best Z %.3f remains near search boundary [%.3f, %.3f]; "
                    "focus may be outside the searched range",
                    best_z,
                    lower_z,
                    upper_z,
                )

            self.set_z_position(best_z)
            autofocus_completed = True
            return float(best_z)
        finally:
            if not autofocus_completed:
                try:
                    self._restore_runtime_state(state, restore_xy=False, restore_preview=True)
                except Exception:
                    logger.exception("Failed to restore microscope state after autofocus failure")

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
        del tolerance  # Kept for compatibility with older prompt signatures.
        if self.get_channel() != '1-NONE':
            self.set_brightness(0)
            return 0

        min_br = int(self.Min_brightness)
        max_br = int(self.Max_brightness)
        original_brightness = int(max(min_br, min(self.get_brightness(), max_br)))
        samples: Dict[int, Dict[str, float]] = {}

        def capture_metrics(brightness: int) -> Dict[str, float]:
            br = int(max(min_br, min(brightness, max_br)))
            if br in samples:
                return samples[br]
            self.set_brightness(br)
            if settle_time_sec > 0:
                time.sleep(settle_time_sec)
            img = self._snap_image_preserving_preview()
            metrics = brightness_metrics(
                img,
                intensity_max=self._get_image_intensity_max(img),
                high_percentile=high_percentile,
            )
            samples[br] = metrics
            logger.info(
                "Autobrightness sample brightness=%s p50=%.3f p95=%.3f p%s=%.3f saturation=%.4f dark=%.4f",
                br,
                metrics["p50"],
                metrics["p95"],
                high_percentile,
                metrics["p_high"],
                metrics["saturation_ratio"],
                metrics["dark_ratio"],
            )
            return metrics

        def candidate_key(item: Tuple[int, Dict[str, float]]) -> Tuple[int, float, float]:
            br, metrics = item
            is_overexposed = (
                metrics["saturation_ratio"] > max_saturation_ratio
                or metrics["p_high"] >= 0.98
            )
            if is_overexposed:
                return (
                    1,
                    metrics["saturation_ratio"] + abs(metrics["p_high"] - target_high_percentile),
                    abs(br - original_brightness),
                )
            dark_penalty = max(0.0, min_median_ratio - metrics["p50"]) * 0.25
            return (
                0,
                abs(metrics["p_high"] - target_high_percentile) + dark_penalty,
                abs(br - original_brightness),
            )

        original_metrics = capture_metrics(original_brightness)
        if (
            original_metrics["saturation_ratio"] > max_saturation_ratio
            or original_metrics["p_high"] > target_high_percentile
        ):
            low, high = min_br, original_brightness
        else:
            low, high = original_brightness, max_br

        capture_metrics(low)
        capture_metrics(high)

        for _ in range(max_iterations):
            if high - low <= 1:
                break
            mid = int(round((low + high) / 2))
            if mid in samples:
                break
            metrics = capture_metrics(mid)
            if metrics["saturation_ratio"] > max_saturation_ratio or metrics["p_high"] > target_high_percentile:
                high = mid
            else:
                low = mid

        best_brightness, best_metrics = min(samples.items(), key=candidate_key)
        logger.info(
            "Autobrightness selected brightness=%s p%s=%.3f saturation=%.4f",
            best_brightness,
            high_percentile,
            best_metrics["p_high"],
            best_metrics["saturation_ratio"],
        )
        self.set_brightness(best_brightness)
        return int(best_brightness)

    # ====== System Control (No Modifications) ======

    @tool_func
    def shutdown(self):
        self.shutdown_event.set()
        if self.preview_running:
            print("Microscope shutdown: stopping preview...")
            self.stop_preview()
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            print("Microscope shutdown: waiting for acquisition thread...")
            self.acquisition_thread.join(timeout=5.0)
        try:
            with self.device_lock:
                print("Microscope shutdown: resetting hardware core...")
                self._set_transmitted_brightness(self.Min_brightness)
                with _silence_native_stdio():
                    self.core.stopSequenceAcquisition()
                    self.core.reset()
                    self.core.unloadAllDevices()
                print("Microscope shutdown: hardware core reset complete.")
        except Exception as e:
            pass
    @tool_func
    def load_target_locations(self, filename: str) -> List[Tuple[float, float, float, float]]:
        filepath = os.path.join(self.output_directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Target location file not found: {filepath}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Target location file is not valid JSON: {filepath}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load target locations from {filepath}: {exc}") from exc

        regions = []
        for item in loaded_data:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                x, y, width, height = map(float, item)
                regions.append((x, y, width, height))

        if not regions:
            raise ValueError(f"No valid target locations were found in {filepath}")
        return regions
    @tool_func
    def create_96_wells_positions(self) -> List[Tuple[float, float]] :
        """Generates positions for each well in a 96-well plate.
        Follows the standard 96-well plate layout (8 columns × 12 rows, A-H rows, 1-12 columns).
        Coordinates are based on standard well spacing (center-to-center) of 96-well plates,
        with (0, 0) as the reference origin (adjustable via internal parameters).

        Returns:
            positions: Positions (micrometer) of each well in the 96-well plate
                    List order: A1 → A12 → B1 → B12 → ... → H1 → H12
                    Each tuple is (X_coordinate_um, Y_coordinate_um)
        """
        return _generate_well_positions(
            rows=8,
            cols=12,
            x_spacing_um=9000.0,
            y_spacing_um=9000.0,
        )
    @tool_func
    def create_24_wells_positions(self) -> List[Tuple[float, float]]:
        """Generates positions for each well in a standard 24-well plate.
        Follows the standard 24-well plate layout (4 rows × 6 columns, A-D rows, 1-6 columns).
        Coordinates are based on standard well center-to-center spacing, with (0, 0) as the reference origin (A1 well center).

        Returns:
            positions: Positions (micrometer) of each well in the 24-well plate
                    List order: A1 → A6 → B1 → B6 → ... → D1 → D6
                    Each tuple is (X_coordinate_um, Y_coordinate_um)
        """
        return _generate_well_positions(
            rows=4,
            cols=6,
            x_spacing_um=12700.0,
            y_spacing_um=12700.0,
        )

    @tool_func
    def z_stack_range(self) -> Tuple[float, float]:
        """
        Calculates recommended Z-stack scanning range (μm) from a sharpness-vs-Z curve.

        Strategy:
            - Keep preview available for visual feedback during the scan.
            - Discretely sample raw images around current Z using objective-aware spacing.
            - Identify a continuous high-sharpness plateau as the recommended stack range.

        Returns:
            (z_max, z_min): Recommended maximum and minimum Z positions for stacking (μm)
        """
        def z_stack_scan_params(magnification: float, is_fluorescence: bool) -> Dict[str, float]:
            if magnification < 5:
                params = {"range": 250.0, "step": 30.0, "min_width": 40.0, "max_width": 350.0}
            elif magnification < 15:
                params = {"range": 200.0, "step": 25.0, "min_width": 30.0, "max_width": 280.0}
            elif magnification < 30:
                params = {"range": 150.0, "step": 20.0, "min_width": 20.0, "max_width": 220.0}
            elif magnification < 50:
                params = {"range": 80.0, "step": 10.0, "min_width": 10.0, "max_width": 120.0}
            else:
                params = {"range": 50.0, "step": 6.0, "min_width": 6.0, "max_width": 70.0}
            params["settle_time"] = 0.15 if is_fluorescence else 0.10
            params["roi_size"] = 1024.0
            params["threshold_ratio"] = 0.60
            params["margin_steps"] = 2.0
            return params

        def clamp_z(z_position: float) -> float:
            return float(max(self.Min_Z_position, min(float(z_position), self.Max_Z_position)))

        def fallback_range(center_z: float, half_width: float) -> Tuple[float, float]:
            z_min = clamp_z(center_z - half_width)
            z_max = clamp_z(center_z + half_width)
            return (z_max, z_min)

        def smooth_scores(scores: np.ndarray) -> np.ndarray:
            if scores.size < 5:
                return scores
            try:
                from scipy.signal import savgol_filter
                window_length = min(11, scores.size if scores.size % 2 else scores.size - 1)
                if window_length >= 5:
                    return savgol_filter(scores, window_length=window_length, polyorder=2)
            except Exception:
                logger.debug("Failed to smooth Z-stack sharpness scores", exc_info=True)
            return scores

        def contiguous_true_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
            regions: List[Tuple[int, int]] = []
            start_idx: Optional[int] = None
            for idx, is_selected in enumerate(mask):
                if is_selected and start_idx is None:
                    start_idx = idx
                elif not is_selected and start_idx is not None:
                    regions.append((start_idx, idx - 1))
                    start_idx = None
            if start_idx is not None:
                regions.append((start_idx, len(mask) - 1))
            return regions

        state = self._capture_runtime_state(include_xy=False, include_preview=True)
        orig_z = float(state["z"])
        orig_channel = state["channel"]
        orig_objective = self.get_objective()
        magnification = float(objective_labels.get(orig_objective, 10.0))
        params = z_stack_scan_params(magnification, orig_channel != '1-NONE')

        was_preview_running = bool(state["preview_running"])
        if not was_preview_running:
            self.start_preview()
            time.sleep(0.5)

        z_start = clamp_z(orig_z - params["range"])
        z_end = clamp_z(orig_z + params["range"])
        z_step = float(params["step"])
        z_positions = _build_z_positions(z_start, z_end, z_step)
        if z_positions.size == 0:
            return fallback_range(orig_z, params["min_width"] / 2.0)

        sharpness_samples: List[Tuple[float, float]] = []

        try:
            for z_position in z_positions:
                if self.shutdown_event.is_set():
                    break
                self.set_z_position(float(z_position))
                time.sleep(float(params["settle_time"]))
                image = self._snap_image_preserving_preview()
                score = float(
                    tool_utils.tenengrad_calculate_sharpness(
                        image,
                        center_roi_size=int(params["roi_size"]),
                    )
                )
                sharpness_samples.append((float(z_position), score))
        finally:
            self._restore_runtime_state(state, restore_xy=False, restore_preview=True)

        if len(sharpness_samples) < 5:
            logger.warning("Z-stack range scan collected too few samples: %s", len(sharpness_samples))
            return fallback_range(orig_z, params["min_width"] / 2.0)

        z_vals, scores = zip(*sharpness_samples)
        z_vals = np.array(z_vals)
        scores = np.array(scores)
        scores_smooth = smooth_scores(scores)
        peak_idx = int(np.argmax(scores_smooth))
        peak_score = float(scores_smooth[peak_idx])
        baseline = float(np.percentile(scores_smooth, 10))
        score_span = peak_score - baseline

        if score_span <= max(abs(peak_score), 1.0) * 0.02:
            logger.warning(
                "Z-stack sharpness curve is flat; peak=%.3f baseline=%.3f",
                peak_score,
                baseline,
            )
            return fallback_range(float(z_vals[peak_idx]), params["min_width"] / 2.0)

        threshold = baseline + float(params["threshold_ratio"]) * score_span
        above = scores_smooth >= threshold
        regions = contiguous_true_regions(above)
        if not regions:
            logger.warning("Z-stack range scan found no high-sharpness plateau")
            return fallback_range(float(z_vals[peak_idx]), params["min_width"] / 2.0)

        peak_regions = [region for region in regions if region[0] <= peak_idx <= region[1]]
        if peak_regions:
            region_start, region_end = peak_regions[0]
        else:
            region_start, region_end = max(
                regions,
                key=lambda region: float(np.sum(scores_smooth[region[0]:region[1] + 1])),
            )

        z_min = float(z_vals[region_start])
        z_max = float(z_vals[region_end])
        margin = float(params["margin_steps"]) * z_step
        z_min = clamp_z(z_min - margin)
        z_max = clamp_z(z_max + margin)

        width = z_max - z_min
        min_width = float(params["min_width"])
        max_width = float(params["max_width"])
        peak_z = float(z_vals[peak_idx])
        if width < min_width:
            z_min = clamp_z(peak_z - min_width / 2.0)
            z_max = clamp_z(peak_z + min_width / 2.0)
        elif width > max_width:
            z_min = clamp_z(peak_z - max_width / 2.0)
            z_max = clamp_z(peak_z + max_width / 2.0)

        if region_start == 0 or region_end == len(z_vals) - 1:
            logger.warning(
                "Z-stack high-sharpness plateau touches scan boundary [%.3f, %.3f]; "
                "returned range may be truncated",
                z_start,
                z_end,
            )

        return (z_max, z_min)

    @tool_func
    def detect_targets_in_image(
            self,
            image_data: ImagingData,
            target_class: str,
            confidence_threshold: float = 0.5,
            device: Optional[Any] = None
    ) -> List[Dict[str, float]]:
        if not isinstance(image_data, ImagingData):
            raise TypeError("image_data must be an ImagingData instance")

        if image_data.pixel_size is None or float(image_data.pixel_size) <= 0:
            raise ValueError("image_data.pixel_size must be a positive number")

        image_2d = _coerce_detection_image_to_2d(image_data.image)
        pixel_size = float(image_data.pixel_size)
        if torch is None or init_detector is None or inference_detector is None:
            raise RuntimeError(
                "MMDetection dependencies are unavailable. Please install a compatible "
                "mmdet/mmcv/torch stack before using detect_targets_in_image."
            )

        h, w = image_2d.shape
        img_center_x_px = (w - 1) / 2.0
        img_center_y_px = (h - 1) / 2.0
        image_center_x_um = float(image_data.center_x)
        image_center_y_um = float(image_data.center_y)

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_cache_key = f"_mmdet_model_{target_class}"

        try:
            if hasattr(self, model_cache_key) and getattr(self, model_cache_key) is not None:
                model = getattr(self, model_cache_key)
            else:
                if target_class not in TARGET_MODEL_MAP:
                    raise ValueError(f"Target class '{target_class}' not in TARGET_MODEL_MAP")
                config_path, ckpt_path = TARGET_MODEL_MAP[target_class]
                if not config_path or not ckpt_path:
                    raise RuntimeError(f"MMDetection model paths are not configured for target '{target_class}'")
                model = init_detector(config_path, ckpt_path, device=device)
                setattr(self, model_cache_key, model)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize MMDetection model for '{target_class}': {exc}") from exc

        if self.auto_contrast_enabled:
            low, high = np.percentile(image_2d, [self.contrast_percentile, 100 - self.contrast_percentile])
            img_norm = np.clip(image_2d, low, high)
            img_uint8 = ((img_norm - low) / (high - low + 1e-8) * 255).astype(np.uint8)
        else:
            img_max = float(np.max(image_2d))
            img_uint8 = (image_2d / (img_max + 1e-8) * 255).astype(np.uint8) if img_max > 0 else image_2d.astype(np.uint8)

        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

        try:
            det_results = inference_detector(model, img_rgb)
        except Exception as exc:
            raise RuntimeError(f"Failed to run MMDetection inference for '{target_class}': {exc}") from exc

        classes = _resolve_model_classes(model)
        if target_class not in classes:
            return []

        class_idx = classes.index(target_class)
        class_dets = _extract_class_detections(det_results, class_idx)

        if class_dets.size == 0:
            return []

        valid_dets = class_dets[class_dets[:, 4] >= confidence_threshold]
        if valid_dets.size == 0:
            return []

        results = []
        for x1, y1, x2, y2, score in valid_dets:
            cx_px = (x1 + x2) / 2.0
            cy_px = (y1 + y2) / 2.0

            offset_x_um = (cx_px - img_center_x_px) * pixel_size
            offset_y_um = -(cy_px - img_center_y_px) * pixel_size
            center_x_um = image_center_x_um + offset_x_um
            center_y_um = image_center_y_um + offset_y_um

            results.append({
                "center_x_um": float(center_x_um),
                "center_y_um": float(center_y_um),
                "offset_x_um": float(offset_x_um),
                "offset_y_um": float(offset_y_um),
                "confidence": float(score),
            })
        return results




