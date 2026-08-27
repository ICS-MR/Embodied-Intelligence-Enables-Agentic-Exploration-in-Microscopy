import logging
import math
from contextlib import contextmanager
import os
from pathlib import Path
import threading
import time
from datetime import datetime
from queue import Queue, Empty
from typing import Any, Callable, List, Dict, Optional, Tuple
import cv2
import numpy as np
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.writers import OmeTiffWriter
from ome_types.model import Plane
# Prevent pymmcore_plus from creating a file handler at import time.
# Its configure_logging() crashes with PermissionError when the log file
# is locked by another process.  Setting PYTEST_RUNNING disables the
# file handler entirely (the only side-effect is skipping file logging).
import os as _os
_os.environ.setdefault("PYTEST_RUNNING", "1")
del _os

from pymmcore_plus import CMMCorePlus
from core_tool import tool_utils
import json

# mmdet imports are deferred to detect_targets_in_image() to avoid
# hanging at module load time when mmdet initialization is slow.

try:
    import torch
except Exception:
    torch = None

from bootstrap.microscope_semantics import objective_pixel_size_um
logger = logging.getLogger(__name__)

# Wall-clock caps for autofocus/autobrightness. Defaults only; overridden by the
# runtime config fields autofocus_timeout_seconds / autobrightness_timeout_seconds.
AUTOFOCUS_TIMEOUT_SEC = 300.0
AUTOBRIGHTNESS_TIMEOUT_SEC = 20.0


# ====== MMCore console noise suppression ======
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


# ====== Detection helpers ======
def _coerce_detection_image_to_2d(image: np.ndarray) -> np.ndarray:
    """Accept a 2D image or an unambiguous acquisition plane for detection."""
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return image_array

    squeezed = np.squeeze(image_array)
    if squeezed.ndim == 2:
        return squeezed

    if image_array.ndim == 5:
        time_count, channel_count, z_count, _height, _width = image_array.shape
        if time_count == 1 and z_count == 1 and channel_count >= 1:
            return image_array[0, 0, 0, :, :]
        raise ValueError(
            "Detection requires a single 2D image plane. "
            f"Received acquisition image with shape {image_array.shape}, interpreted as (T, C, Z, H, W). "
            "Only single-timepoint, single-Z acquisition tensors can be reduced automatically; "
            "select the desired timepoint/channel/Z plane before detection."
        )

    raise ValueError(
        "Only 2D grayscale image, singleton multidimensional image, or single-timepoint "
        f"single-Z acquisition tensor supported; got shape {image_array.shape}."
    )


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


def _resolve_detection_target_spec(
    target_class: str,
    target_specs: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Resolve a detection target spec by registered target key (case-insensitive)."""
    target_specs = {
        str(key): dict(value)
        for key, value in target_specs.items()
        if isinstance(value, dict)
    }
    normalized = str(target_class or "").strip()
    resolved_spec: Optional[Dict[str, Any]] = None
    if normalized in target_specs:
        resolved_spec = dict(target_specs[normalized])
    if resolved_spec is None:
        lowered = normalized.lower()
        for key, spec in target_specs.items():
            if key.lower() == lowered:
                resolved_spec = dict(spec)
                break
    if resolved_spec is None:
        raise ValueError(
            f"Unsupported target class: {target_class}. "
            f"Available targets: {', '.join(target_specs.keys())}"
        )
    return resolved_spec


def _resolve_detection_class_index(classes: List[str], class_name: Optional[str]) -> int:
    """Select the model class index by name, falling back to the only class."""
    if class_name:
        for index, candidate in enumerate(classes):
            if str(candidate) == class_name:
                return index
    if len(classes) == 1:
        return 0
    raise RuntimeError(
        f"Unable to resolve target class from model classes {list(classes)}; "
        f"requested name={class_name!r}"
    )


def _validate_loaded_devices(loaded_devices: Any, required_devices: list[str]) -> None:
    loaded = {str(device) for device in loaded_devices}
    missing = [device for device in required_devices if device and device not in loaded]
    if missing:
        raise RuntimeError(f"Core devices not loaded: {missing}")


def _required_float_config(config: Any, field_name: str) -> float:
    if not hasattr(config, field_name):
        raise ValueError(f"Missing required microscope system config field: {field_name}")
    raw_value = getattr(config, field_name)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Microscope system config field {field_name} must be numeric, got {raw_value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Microscope system config field {field_name} must be finite, got {value!r}")
    return value


def _required_int_config(config: Any, field_name: str) -> int:
    value = _required_float_config(config, field_name)
    if not value.is_integer():
        raise ValueError(f"Microscope system config field {field_name} must be an integer, got {value!r}")
    return int(value)


def _optional_float_config(config: Any, field_name: str, default: float) -> float:
    if not hasattr(config, field_name):
        return float(default)
    raw_value = getattr(config, field_name)
    if raw_value in (None, ""):
        return float(default)
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Microscope system config field {field_name} must be numeric, got {raw_value!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"Microscope system config field {field_name} must be finite, got {value!r}")
    return value


def _configured_numeric_range(config: Any, min_field: str, max_field: str) -> Tuple[float, float]:
    minimum = _required_float_config(config, min_field)
    maximum = _required_float_config(config, max_field)
    if minimum > maximum:
        raise ValueError(
            f"Microscope system config range is invalid: {min_field}={minimum!r} > {max_field}={maximum!r}"
        )
    return minimum, maximum


def _configured_int_range(config: Any, min_field: str, max_field: str) -> Tuple[int, int]:
    minimum = _required_int_config(config, min_field)
    maximum = _required_int_config(config, max_field)
    if minimum > maximum:
        raise ValueError(
            f"Microscope system config range is invalid: {min_field}={minimum!r} > {max_field}={maximum!r}"
        )
    return minimum, maximum


def _configured_brightness_range(system_config: Any, transmitted_light: Dict[str, Any]) -> Tuple[int, int]:
    system_min, system_max = _configured_int_range(system_config, "Min_brightness", "Max_brightness")
    raw_min = transmitted_light.get("min")
    raw_max = transmitted_light.get("max")
    if raw_min in (None, "") and raw_max in (None, ""):
        return system_min, system_max
    if raw_min in (None, "") or raw_max in (None, ""):
        raise ValueError("transmitted_light min/max must be configured together when overriding system brightness limits")
    try:
        minimum = int(float(raw_min))
        maximum = int(float(raw_max))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"transmitted_light min/max must be integer-compatible values, got min={raw_min!r}, max={raw_max!r}"
        ) from exc
    if minimum > maximum:
        raise ValueError(f"transmitted_light brightness range is invalid: min={minimum!r} > max={maximum!r}")
    return minimum, maximum


# ====== Brightness analysis helpers ======
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


# ====== Acquisition geometry helpers ======
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
    def __init__(
        self,
        config_path: str,
        app_dir: str,
        output_path: str,
        storagemanger,
        *,
        system_config: Any = None,
        detection_targets: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        if system_config is None:
            raise ValueError(
                "MicroscopeController requires an explicit system_config. "
                "Load runtime settings in the runtime assembly layer and pass settings.system."
            )

        detection_targets = {
            str(target_name): dict(spec)
            for target_name, spec in (detection_targets or {}).items()
        }
        self.target_model_map = {
            target_name: (
                str(spec.get("model_config", "")),
                str(spec.get("model_checkpoint", "")),
            )
            for target_name, spec in detection_targets.items()
        }
        self.detection_targets = detection_targets
        self.system_config = system_config
        self.objective_labels = dict(getattr(system_config, "objective_labels", {}))
        self.dichroic_colors = dict(getattr(system_config, "dichroic_colors", {}))
        self.objectives = dict(getattr(system_config, "objectives", {}))
        self.channels = dict(getattr(system_config, "channels", {}))
        self._storagemanger = storagemanger
        self.app_dir = app_dir
        self.config_path = config_path
        with _silence_native_stdio():
            self.core = CMMCorePlus(mm_path=app_dir or None, adapter_paths=[app_dir] if app_dir else ())
        _configure_core_logging(self.core)
        self.device_lock = threading.RLock()
        self.camera_device = getattr(system_config, "camera_device", "")
        self.xy_stage_device = getattr(system_config, "xy_stage_device", "")
        self.objective_device = getattr(system_config, "objective_device", "")
        self.focus_drive = getattr(system_config, "focus_drive", "")
        self.Dichroic = getattr(system_config, "Dichroic", "")
        transmitted_light = dict(getattr(system_config, "transmitted_light", {}) or {})
        self.brightness_device = str(transmitted_light.get("device") or "").strip()
        self.brightness_property = str(transmitted_light.get("intensity_property") or "").strip()
        self._configured_brightness_property = self.brightness_property
        self.brightness_property_source = "configured" if self.brightness_property else "unavailable"
        self.brightness_property_candidates: List[str] = []
        self.brightness_property_discovery_reason = ""
        self.brightness_control_kind = str(transmitted_light.get("control_kind") or "").strip()
        self.brightness_surrogate_min_property_value = float(
            transmitted_light.get("surrogate_min_property_value", 0.5)
        )
        self.brightness_surrogate_scale = float(transmitted_light.get("surrogate_scale", 100.0))

        # Hardware limits must come from validated config; controller does not guess them.
        self.Min_X_position, self.Max_X_position = _configured_numeric_range(
            system_config, "Min_X_position", "Max_X_position"
        )
        self.Min_Y_position, self.Max_Y_position = _configured_numeric_range(
            system_config, "Min_Y_position", "Max_Y_position"
        )
        self.Min_Z_position, self.Max_Z_position = _configured_numeric_range(
            system_config, "Min_Z_position", "Max_Z_position"
        )
        self.Min_brightness, self.Max_brightness = _configured_brightness_range(system_config, transmitted_light)
        self.Min_exposure, self.Max_exposure = _configured_numeric_range(
            system_config, "Min_exposure", "Max_exposure"
        )
        self.acquisition_timeout_base_seconds = _optional_float_config(
            system_config, "acquisition_timeout_base_seconds", 0.0
        )
        self.acquisition_timeout_per_position_seconds = _optional_float_config(
            system_config, "acquisition_timeout_per_position_seconds", 0.0
        )
        self.autofocus_timeout_seconds = _optional_float_config(
            system_config, "autofocus_timeout_seconds", AUTOFOCUS_TIMEOUT_SEC
        )
        self.autobrightness_timeout_seconds = _optional_float_config(
            system_config, "autobrightness_timeout_seconds", AUTOBRIGHTNESS_TIMEOUT_SEC
        )
        if self.autofocus_timeout_seconds <= 0:
            logger.warning(
                "autofocus_timeout_seconds must be positive; falling back to default %.1fs",
                AUTOFOCUS_TIMEOUT_SEC,
            )
            self.autofocus_timeout_seconds = AUTOFOCUS_TIMEOUT_SEC
        if self.autobrightness_timeout_seconds <= 0:
            logger.warning(
                "autobrightness_timeout_seconds must be positive; falling back to default %.1fs",
                AUTOBRIGHTNESS_TIMEOUT_SEC,
            )
            self.autobrightness_timeout_seconds = AUTOBRIGHTNESS_TIMEOUT_SEC

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
        self.acquisition_thread = None
        self.image_queue = Queue(maxsize=5)  # Only stores image arrays for display
        self.is_continuous = False
        self.shutdown_event = threading.Event()
        self.preview_stop_event = threading.Event()
        self._hardware_shutdown_complete = False
        self.img_lock = threading.Lock()
        self.latest_display_frame: Optional[np.ndarray] = None
        self.last_preview_frame_at: Optional[float] = None
        self.preview_started_at: Optional[float] = None
        self.last_preview_error: str = ""
        self._preview_auto_shutter_original: Optional[bool] = None
        self._preview_shutter_forced_open = False

        self.acquisition_running = False
        self._task_progress_listener: Optional[Callable[[dict[str, Any]], None]] = None
        self._task_progress_lock = threading.Lock()
        self._last_task_progress: Optional[Dict[str, Any]] = None

        # Auto contrast
        self.auto_contrast_enabled = True
        self.contrast_percentile = 0.1
        self._detection_models: Dict[str, Any] = {}
        self._stage_image_measured_pixel_sizes: Dict[str, float] = {}

    def set_task_progress_listener(
        self,
        listener: Optional[Callable[[dict[str, Any]], None]],
    ) -> None:
        self._task_progress_listener = listener

    def get_last_task_progress(self) -> Optional[Dict[str, Any]]:
        if not hasattr(self, "_task_progress_lock"):
            self._task_progress_lock = threading.Lock()
        if not hasattr(self, "_last_task_progress"):
            self._last_task_progress = None
        with self._task_progress_lock:
            if self._last_task_progress is None:
                return None
            return dict(self._last_task_progress)

    def _emit_task_progress(
        self,
        *,
        task_kind: str,
        status: str,
        title: str,
        detail: str,
        stage_key: str,
        stage_label: str,
        progress_current: int = 0,
        progress_total: int = 0,
    ) -> None:
        if not hasattr(self, "_task_progress_lock"):
            self._task_progress_lock = threading.Lock()
        if not hasattr(self, "_task_progress_listener"):
            self._task_progress_listener = None
        if not hasattr(self, "_last_task_progress"):
            self._last_task_progress = None
        progress_percent = 0
        if progress_total > 0:
            progress_percent = int(max(0.0, min(100.0, (float(progress_current) / float(progress_total)) * 100.0)))
        payload = {
            "task_kind": str(task_kind),
            "status": str(status),
            "title": str(title),
            "detail": str(detail),
            "progress_current": int(progress_current),
            "progress_total": int(progress_total),
            "progress_percent": int(progress_percent),
            "stage_key": str(stage_key),
            "stage_label": str(stage_label),
            "timestamp": datetime.now().isoformat(),
        }
        with self._task_progress_lock:
            self._last_task_progress = dict(payload)
        if self._task_progress_listener is None:
            return
        try:
            self._task_progress_listener(dict(payload))
        except Exception:
            logger.exception("Failed to emit microscope task progress event")

    def _raise_if_long_task_timed_out(
        self,
        *,
        deadline: float,
        task_kind: str,
        stage_label: str,
        detail: str,
    ) -> None:
        if time.monotonic() <= deadline:
            return
        raise TimeoutError(f"{task_kind} timed out during {stage_label}: {detail}")

    @contextmanager
    def _acquisition_guard(self):
        previous_running = self.acquisition_running
        self.acquisition_running = True
        try:
            yield
        finally:
            self.acquisition_running = previous_running

    def _warm_up_camera_for_initialization(self) -> None:
        with self.device_lock:
            self.core.startContinuousSequenceAcquisition(0)
            if self.shutdown_event.wait(timeout=1.0):
                self.core.stopSequenceAcquisition()
                raise RuntimeError("microscope initialization cancelled")
            while self.core.getRemainingImageCount() > 0:
                if self.shutdown_event.is_set():
                    self.core.stopSequenceAcquisition()
                    raise RuntimeError("microscope initialization cancelled")
                try:
                    self.core.getLastImage()
                except IndexError:
                    break
            self.core.stopSequenceAcquisition()

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
            logger.debug("Failed to read MMCore auto shutter state", exc_info=True)
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

    def _capture_runtime_state(
        self,
        *,
        include_xy: bool,
        include_preview: bool,
    ) -> Dict[str, Any]:
        channel = self.get_channel()
        transmitted_brightness = self.get_brightness()
        state: Dict[str, Any] = {
            "z": self.get_z_position(),
            "channel": channel,
            "exposure": self.get_exposure(),
            "brightfield_memory": (
                int(self._clamp_brightness(transmitted_brightness))
                if self._is_brightfield_channel(channel)
                else int(self._user_brightness)
            ),
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
        if restore_preview and not target_preview_running and self.preview_running:
            self.stop_preview()

    def capture_hardware_owner_state(self) -> Dict[str, Any]:
        state = self._capture_runtime_state(include_xy=True, include_preview=True)
        state["objective"] = self.get_objective()
        return state

    def release_hardware_owner(self) -> None:
        if self._hardware_shutdown_complete:
            return
        if self.preview_running:
            print("Microscope hardware release: stopping preview...")
            self.stop_preview()
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            print("Microscope hardware release: waiting for acquisition thread...")
            self.acquisition_thread.join(timeout=5.0)
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            raise RuntimeError("microscope acquisition thread did not stop within 5 seconds")
        with self.device_lock:
            print("Microscope hardware release: resetting hardware core...")
            self._write_transmitted_brightness(self.Min_brightness)
            with _silence_native_stdio():
                self.core.stopSequenceAcquisition()
                self.core.reset()
                self.core.unloadAllDevices()
            print("Microscope hardware release: hardware core reset complete.")
        self._hardware_shutdown_complete = True

    def restore_hardware_owner(self, state: Dict[str, Any]) -> None:
        if self._hardware_shutdown_complete:
            self.initialize()
        if state.get("objective"):
            self.set_objective(state["objective"])
        self._restore_runtime_state(state, restore_xy=True, restore_preview=False)
        if bool(state.get("preview_running", False)) and not self.preview_running:
            self.start_preview()

    def _reset_acquisition_plan(self) -> None:
        self.acquisition_positions.clear()
        self.acquisition_channels.clear()
        self.z_stack_params = None
        self.time_lapse_params = None

    def _is_writable_numeric_runtime_property(self, device: str, property_name: str) -> bool:
        try:
            if self.core.isPropertyReadOnly(device, property_name):
                return False
        except Exception:
            logger.debug(
                "MMCore could not report whether %s.%s is read-only",
                device,
                property_name,
                exc_info=True,
            )
        try:
            if self.core.isPropertyPreInit(device, property_name):
                return False
        except Exception:
            logger.debug(
                "MMCore could not report whether %s.%s is PreInit-only",
                device,
                property_name,
                exc_info=True,
            )

        metadata_methods_available = any(
            callable(getattr(self.core, method_name, None))
            for method_name in (
                "getPropertyType",
                "hasPropertyLimits",
                "getAllowedPropertyValues",
            )
        )
        if not metadata_methods_available:
            return True

        property_type = ""
        try:
            property_type = str(self.core.getPropertyType(device, property_name) or "").lower()
        except Exception:
            logger.debug(
                "MMCore could not report the type of %s.%s",
                device,
                property_name,
                exc_info=True,
            )
        if any(token in property_type for token in ("integer", "float", "double")):
            return True

        try:
            if self.core.hasPropertyLimits(device, property_name):
                return True
        except Exception:
            logger.debug(
                "MMCore could not report limits for %s.%s",
                device,
                property_name,
                exc_info=True,
            )

        try:
            allowed_values = list(self.core.getAllowedPropertyValues(device, property_name))
            if allowed_values:
                for value in allowed_values:
                    float(str(value))
                return True
        except (TypeError, ValueError):
            return False
        except Exception:
            logger.debug(
                "MMCore could not report allowed values for %s.%s",
                device,
                property_name,
                exc_info=True,
            )
        return False

    def _resolve_transmitted_light_property_from_core(self) -> None:
        self.brightness_property_candidates = []
        configured_property = str(self._configured_brightness_property or "").strip()
        if not self.brightness_device and not configured_property:
            self.brightness_property_source = "unavailable"
            self.brightness_property_discovery_reason = (
                "No transmitted-light intensity-control device is configured."
            )
            return
        if not self.brightness_device:
            raise RuntimeError(
                "transmitted_light.intensity_property is configured, but "
                "transmitted_light.device is missing. Configure both fields before initialization."
            )
        if not configured_property:
            self.brightness_property = ""
            self.brightness_property_source = "unavailable"
            self.brightness_property_discovery_reason = (
                "transmitted_light.device is configured, but transmitted_light.intensity_property is missing."
            )
            raise RuntimeError(
                f"Transmitted-light device '{self.brightness_device}' is configured, but "
                "transmitted_light.intensity_property is missing. Configure the exact MMCore "
                "property in the configuration or complete the startup self-check before initialization."
            )

        try:
            property_exists = bool(self.core.hasProperty(self.brightness_device, configured_property))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to verify configured transmitted-light property "
                f"'{self.brightness_device}.{configured_property}' against the loaded "
                f"Micro-Manager device adapter: {exc}"
            ) from exc
        if not property_exists:
            raise RuntimeError(
                f"Configured transmitted-light property "
                f"'{self.brightness_device}.{configured_property}' is not exposed by the "
                "loaded Micro-Manager device adapter."
            )
        if not self._is_writable_numeric_runtime_property(
            self.brightness_device,
            configured_property,
        ):
            raise RuntimeError(
                f"Configured transmitted-light property "
                f"'{self.brightness_device}.{configured_property}' is not a writable, "
                "runtime-settable numeric control."
            )
        self.brightness_property = configured_property
        self.brightness_property_source = "configured"
        self.brightness_property_discovery_reason = (
            "The configured property was verified against the loaded Micro-Manager device adapter."
        )

    def get_transmitted_light_runtime_info(self) -> Dict[str, Any]:
        return {
            "available": self._supports_transmitted_brightness(),
            "device": self.brightness_device,
            "configured_property": str(self._configured_brightness_property or ""),
            "selected_property": self.brightness_property,
            "source": self.brightness_property_source,
            "candidates": list(self.brightness_property_candidates),
            "reason": self.brightness_property_discovery_reason,
            "min": self.Min_brightness,
            "max": self.Max_brightness,
        }

    def initialize(self):
        self._hardware_shutdown_complete = False
        if self.shutdown_event.is_set():
            raise RuntimeError("microscope initialization cancelled")
        with _silence_native_stdio():
            self.core.reset()
            self.core.unloadAllDevices()
        if self.shutdown_event.wait(timeout=1.0):
            raise RuntimeError("microscope initialization cancelled")

        if self.app_dir and self.app_dir not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + self.app_dir

        with self.device_lock:
            self.core.loadSystemConfiguration(self.config_path)
        if self.shutdown_event.is_set():
            raise RuntimeError("microscope initialization cancelled")

        self._resolve_transmitted_light_property_from_core()

        loaded_devices = self.core.getLoadedDevices()
        required_devices = [
            self.camera_device,
            self.xy_stage_device,
            self.objective_device,
            self.focus_drive,
            self.Dichroic,
        ]
        if self.brightness_device:
            required_devices.append(self.brightness_device)
        _validate_loaded_devices(loaded_devices, required_devices)

        self.core.setCameraDevice(self.camera_device)
        self.core.waitForSystem()
        if self.shutdown_event.wait(timeout=4.0):
            raise RuntimeError("microscope initialization cancelled")
        self.core.waitForDevice(self.camera_device)
        if self.shutdown_event.is_set():
            raise RuntimeError("microscope initialization cancelled")

        self.current_X_position, self.current_Y_position = self.get_x_y_position()
        self.current_Z_position = self.get_z_position()
        self.current_channel = self.get_channel()
        self.current_objective = self.get_objective()
        self._warm_up_camera_for_initialization()

        # Test acquisition (using formal acquisition method to get ImagingData)
        test_imaging_data = None
        last_test_acquisition_error: Optional[Exception] = None
        for attempt in range(1, 4):
            if self.shutdown_event.is_set():
                raise RuntimeError("microscope initialization cancelled")
            try:
                test_imaging_data = self._acquire_single_image()
                if test_imaging_data is not None and test_imaging_data.image.size > 0 and len(test_imaging_data.image.shape) == 2:
                    break
                if self.shutdown_event.wait(timeout=1.0):
                    raise RuntimeError("microscope initialization cancelled")
            except Exception as exc:
                if self.shutdown_event.is_set():
                    raise RuntimeError("microscope initialization cancelled")
                last_test_acquisition_error = exc
                logger.warning(
                    "Initialization test acquisition failed. attempt=%s/3 error=%s",
                    attempt,
                    exc,
                    exc_info=True,
                )
                test_imaging_data = None

        if test_imaging_data is None:
            detail = ""
            if last_test_acquisition_error is not None:
                detail = f" Last error: {type(last_test_acquisition_error).__name__}: {last_test_acquisition_error}"
            raise RuntimeError(
                "Initialization acquisition failed after 3 attempts. "
                f"Please check camera connection and configuration.{detail}"
            )

        self.current_img_height, self.current_img_width = test_imaging_data.image.shape
        self.img_dtype = test_imaging_data.image.dtype
        self.is_16bit = (test_imaging_data.image.dtype == np.uint16)

        self.current_objective = self.get_objective()
        if self.current_objective not in self.objective_labels:
            raise RuntimeError(f"Objective not configured: {self.current_objective}")
        self.pixel_size = self._current_pixel_size_um(self.current_objective)

        self.current_X_position, self.current_Y_position = self.get_x_y_position()
        self.current_Z_position = self.get_z_position()
        self.current_channel = self.get_channel()
        self.current_brightness = self.get_brightness()
        self._user_brightness = self.current_brightness

    # ====== Core device control ======
    @tool_func
    def set_x_y_position(self, x: float, y: float):
        if not (self.Min_X_position - 10 <= x <= self.Max_X_position + 10 and
                self.Min_Y_position - 10 <= y <= self.Max_Y_position + 10):
            raise ValueError(
                f"XY position ({float(x):.3f}, {float(y):.3f}) out of effective range "
                f"X=[{float(self.Min_X_position):.3f}, {float(self.Max_X_position):.3f}], "
                f"Y=[{float(self.Min_Y_position):.3f}, {float(self.Max_Y_position):.3f}]"
            )
        if abs(x - self.current_X_position) < 1 and abs(y - self.current_Y_position) < 1:
            return
        self.core.setXYStageDevice(self.xy_stage_device)
        try:
            self.core.setXYPosition(x, y)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to set XY position to ({float(x):.3f}, {float(y):.3f}); effective configured/device range is "
                f"X=[{float(self.Min_X_position):.3f}, {float(self.Max_X_position):.3f}], "
                f"Y=[{float(self.Min_Y_position):.3f}, {float(self.Max_Y_position):.3f}]. "
                f"Original MMCore error: {exc}"
            ) from exc
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
            raise ValueError(
                f"Z position {float(z):.3f} out of effective range "
                f"[{float(self.Min_Z_position):.3f}, {float(self.Max_Z_position):.3f}]"
            )
        if abs(z - self.current_Z_position) < 0.5:
            return
        self.core.setFocusDevice(self.focus_drive)
        try:
            self.core.setPosition(z)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to set Z position to {float(z):.3f}; effective configured/device range is "
                f"[{float(self.Min_Z_position):.3f}, {float(self.Max_Z_position):.3f}]. "
                f"Original MMCore error: {exc}"
            ) from exc
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
        exposure_time = float(exposure_time)
        if not math.isfinite(exposure_time):
            raise ValueError(f"Exposure time must be finite, got {exposure_time!r}")
        if not (float(self.Min_exposure) <= exposure_time <= float(self.Max_exposure)):
            raise ValueError(
                f"Exposure time {exposure_time:.3f} out of configured range "
                f"[{float(self.Min_exposure):.3f}, {float(self.Max_exposure):.3f}]"
            )
        if exposure_time == self.current_exposure_time:
            return
        with self.device_lock:
            was_continuous = self.is_continuous
            if was_continuous:
                self.core.stopSequenceAcquisition()
                self.is_continuous = False
            try:
                self.core.setExposure(float(exposure_time))
                self.core.waitForDevice(self.camera_device)
                self.current_exposure_time = exposure_time
            finally:
                if was_continuous and self.preview_running:
                    self.core.startContinuousSequenceAcquisition(0)
                    self.is_continuous = True
    @tool_func
    def get_exposure(self) -> float:
        exp = self.core.getExposure()
        with self.device_lock:
            self.current_exposure_time = float(exp)
        return float(exp)

    # ====== Config-backed label helpers ======
    def _configured_brightfield_label(self) -> str:
        brightfield_entry = self.channels.get("brightfield", {})
        return str(brightfield_entry.get("label") or "").strip()

    def _is_brightfield_channel(self, channel: str) -> bool:
        brightfield_label = self._configured_brightfield_label()
        return bool(brightfield_label) and str(channel or "").strip() == brightfield_label

    # ====== Transmitted-light brightness control ======
    def _clamp_brightness(self, brightness: int) -> int:
        return int(max(self.Min_brightness, min(int(brightness), self.Max_brightness)))

    def _supports_transmitted_brightness(self) -> bool:
        return bool(self.brightness_device and self.brightness_property)

    def _uses_scaled_brightness_property(self) -> bool:
        return self.brightness_control_kind == "scaled_property"

    def _brightness_to_scaled_property_value(self, brightness: int) -> float:
        brightness = self._clamp_brightness(brightness)
        if brightness <= self.Min_brightness:
            return self.brightness_surrogate_min_property_value
        return max(
            self.brightness_surrogate_min_property_value,
            float(brightness) / self.brightness_surrogate_scale,
        )

    def _clamp_brightness_property_value(
        self,
        property_value: int | float,
        *,
        allow_zero_off: bool = False,
    ) -> int | float:
        del allow_zero_off
        clamped = float(property_value)
        if self._uses_scaled_brightness_property():
            return clamped
        return int(round(clamped))

    def _scaled_property_value_to_brightness(self, property_value: str) -> int:
        try:
            raw_value = float(property_value)
        except (TypeError, ValueError):
            return int(self.current_brightness)
        if raw_value <= self.brightness_surrogate_min_property_value and int(self.current_brightness) == self.Min_brightness:
            return int(self.Min_brightness)
        return self._clamp_brightness(round(raw_value * self.brightness_surrogate_scale))

    def _brightness_unavailable_message(self) -> str:
        return (
            "Transmitted-light brightness control is not configured. "
            "Set system.transmitted_light.device and system.transmitted_light.intensity_property "
            "in config/runtime_config.json if this microscope exposes a device-specific intensity property."
        )

    def _read_transmitted_brightness(self) -> int:
        # Unlike exposure, MMCore does not provide a generic core API for transmitted-light
        # intensity. When a microscope exposes lamp intensity, it must be read from the
        # configured device-specific property.
        if not self._supports_transmitted_brightness():
            return int(self.current_brightness)
        property_value = self.core.getProperty(self.brightness_device, self.brightness_property)
        if self._uses_scaled_brightness_property():
            bright = self._scaled_property_value_to_brightness(property_value)
        else:
            bright = int(float(property_value))
        with self.device_lock:
            self.current_brightness = bright
        return bright

    def _write_transmitted_brightness(self, brightness: int, *, require_supported: bool = False) -> int:
        if not self._supports_transmitted_brightness():
            if require_supported:
                raise RuntimeError(self._brightness_unavailable_message())
            with self.device_lock:
                self.current_brightness = 0
            return 0
        requested_brightness = int(brightness)
        brightness = 0 if requested_brightness <= 0 else self._clamp_brightness(requested_brightness)
        property_value: int | float
        if self._uses_scaled_brightness_property():
            property_value = self._brightness_to_scaled_property_value(brightness)
            property_value = self._clamp_brightness_property_value(property_value)
        else:
            property_value = brightness
            property_value = self._clamp_brightness_property_value(
                property_value,
                allow_zero_off=brightness == 0,
            )
        try:
            self.core.setProperty(self.brightness_device, self.brightness_property, property_value)
        except RuntimeError as exc:
            raise RuntimeError(
                f"Failed to set transmitted-light brightness {brightness} via "
                f"{self.brightness_device}.{self.brightness_property}={property_value}. "
                f"Original MMCore error: {exc}"
            ) from exc
        self.core.waitForDevice(self.brightness_device)
        if self._uses_scaled_brightness_property():
            with self.device_lock:
                self.current_brightness = brightness
            return brightness
        return self._read_transmitted_brightness()

    @tool_func
    def set_brightness(self, brightness: int):
        current_channel = self.get_channel()
        is_brightfield = self._is_brightfield_channel(current_channel)
        if is_brightfield:
            if not self._supports_transmitted_brightness():
                raise RuntimeError(self._brightness_unavailable_message())
            brightness = self._clamp_brightness(brightness)
            self._user_brightness = brightness
        else:
            brightness = 0
        self._write_transmitted_brightness(brightness, require_supported=True)
    @tool_func
    def get_brightness(self) -> int:
        return self._read_transmitted_brightness()

    def _validate_mmcore_pixel_size_affine(self) -> None:
        method = getattr(self.core, "getPixelSizeAffine", None)
        if not callable(method):
            return
        try:
            affine = [float(value) for value in method()]
        except Exception as exc:
            logger.warning("MMCore getPixelSizeAffine() failed; using scalar getPixelSizeUm() only: %s", exc)
            return
        if len(affine) < 6:
            raise RuntimeError(f"MMCore getPixelSizeAffine() returned {len(affine)} values; expected 6.")
        a, b, _c, d, e, _f = affine[:6]
        if a <= 0 or e <= 0:
            raise RuntimeError(f"MMCore pixel size affine must have positive X/Y scale, got {affine[:6]!r}.")
        if not math.isclose(a, e, rel_tol=1e-6, abs_tol=1e-9):
            raise RuntimeError(
                f"MMCore pixel size affine is not square-pixel compatible: X={a!r}, Y={e!r}. "
                "EIMS currently requires scalar square-pixel calibration."
            )
        if abs(b) > 1e-9 or abs(d) > 1e-9:
            raise RuntimeError(
                f"MMCore pixel size affine contains rotation/shear terms: {affine[:6]!r}. "
                "EIMS currently requires a scalar no-shear pixel calibration."
            )

    def _mmcore_current_pixel_size_um(self) -> float:
        method = getattr(self.core, "getPixelSizeUm", None)
        if not callable(method):
            raise RuntimeError("MMCore getPixelSizeUm() API is unavailable.")
        try:
            pixel_size_um = float(method())
        except Exception as exc:
            raise RuntimeError("MMCore could not resolve the current pixel size calibration.") from exc
        if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
            current_config = ""
            try:
                current_config = str(self.core.getCurrentPixelSizeConfig() or "")
            except Exception as exc:
                logger.debug("Failed to read MMCore current pixel-size config after invalid pixel-size value: %s", exc)
                current_config = ""
            detail = f" Current pixel size config: {current_config!r}." if current_config else ""
            raise RuntimeError(f"MMCore returned invalid current pixel size {pixel_size_um!r}.{detail}")
        self._validate_mmcore_pixel_size_affine()
        return pixel_size_um

    def _normalized_registration_image(self, image: np.ndarray) -> np.ndarray:
        image_array = _coerce_detection_image_to_2d(np.asarray(image)).astype(np.float32, copy=False)
        if image_array.size == 0:
            raise RuntimeError("Stage/image pixel-size calibration captured an empty image.")
        finite = np.isfinite(image_array)
        if not np.any(finite):
            raise RuntimeError("Stage/image pixel-size calibration image has no finite pixels.")
        fill_value = float(np.mean(image_array[finite]))
        image_array = np.nan_to_num(image_array, nan=fill_value, posinf=fill_value, neginf=fill_value)
        image_array = image_array - float(np.mean(image_array))
        std = float(np.std(image_array))
        if std <= 1e-6:
            raise RuntimeError("Stage/image pixel-size calibration image has insufficient texture for registration.")
        return image_array / std

    def _phase_correlation_shift(self, reference: np.ndarray, moved: np.ndarray) -> Tuple[float, float, float]:
        reference_image = self._normalized_registration_image(reference)
        moved_image = self._normalized_registration_image(moved)
        if reference_image.shape != moved_image.shape:
            raise RuntimeError(
                f"Stage/image pixel-size calibration images have different shapes: "
                f"{reference_image.shape} vs {moved_image.shape}."
            )
        height, width = reference_image.shape
        window = cv2.createHanningWindow((width, height), cv2.CV_32F)
        (shift_x, shift_y), response = cv2.phaseCorrelate(reference_image, moved_image, window)
        values = (float(shift_x), float(shift_y), float(response))
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Stage/image pixel-size calibration returned non-finite registration values: {values!r}.")
        return values

    def _probe_axis_target(self, *, axis: str, start_x: float, start_y: float, delta_um: float) -> Optional[Tuple[float, float, float]]:
        if axis == "x":
            if start_x + delta_um <= float(self.Max_X_position):
                return start_x + delta_um, start_y, delta_um
            if start_x - delta_um >= float(self.Min_X_position):
                return start_x - delta_um, start_y, -delta_um
            return None
        if axis == "y":
            if start_y + delta_um <= float(self.Max_Y_position):
                return start_x, start_y + delta_um, delta_um
            if start_y - delta_um >= float(self.Min_Y_position):
                return start_x, start_y - delta_um, -delta_um
            return None
        raise ValueError(f"Unsupported calibration axis: {axis!r}")

    def _measure_stage_image_axis_pixel_size_um(
        self,
        *,
        axis: str,
        start_x: float,
        start_y: float,
        reference_image: np.ndarray,
    ) -> Dict[str, float]:
        min_shift_px = 3.0
        max_shift_px = max(8.0, 0.45 * min(reference_image.shape[-2:]))
        rejected: List[str] = []
        for delta_um in (2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0):
            target = self._probe_axis_target(axis=axis, start_x=start_x, start_y=start_y, delta_um=delta_um)
            if target is None:
                rejected.append(f"{delta_um:g} um is outside the configured stage range")
                continue
            target_x, target_y, signed_delta = target
            try:
                self.set_x_y_position(target_x, target_y)
                moved_image = self._snap_raw_image()
                shift_x, shift_y, response = self._phase_correlation_shift(reference_image, moved_image)
            finally:
                self.set_x_y_position(start_x, start_y)
            shift_px = math.hypot(float(shift_x), float(shift_y))
            if response <= 0.02:
                rejected.append(f"{delta_um:g} um response too low ({response:.4f})")
                continue
            if shift_px < min_shift_px:
                rejected.append(f"{delta_um:g} um shift too small ({shift_px:.3f} px)")
                continue
            if shift_px > max_shift_px:
                rejected.append(f"{delta_um:g} um shift too large ({shift_px:.3f} px)")
                continue
            return {
                "axis": axis,
                "stage_delta_um": abs(float(signed_delta)),
                "shift_x_px": float(shift_x),
                "shift_y_px": float(shift_y),
                "shift_px": float(shift_px),
                "response": float(response),
                "pixel_size_um": abs(float(signed_delta)) / float(shift_px),
            }
        raise RuntimeError(
            f"Stage/image pixel-size calibration could not measure a valid {axis.upper()}-axis stage/image shift. "
            f"Rejected probes: {'; '.join(rejected) if rejected else 'none'}"
        )

    def _measure_stage_image_pixel_size_um(self, objective_label: str) -> float:
        objective_label = str(objective_label or "").strip()
        cached = self._stage_image_measured_pixel_sizes.get(objective_label)
        if cached is not None:
            return cached

        start_x, start_y = self.get_x_y_position()
        reference_image = self._snap_raw_image()
        measurements: List[Dict[str, float]] = []
        errors: List[str] = []
        try:
            for axis in ("x", "y"):
                try:
                    measurements.append(
                        self._measure_stage_image_axis_pixel_size_um(
                            axis=axis,
                            start_x=start_x,
                            start_y=start_y,
                            reference_image=reference_image,
                        )
                    )
                except Exception as exc:
                    errors.append(f"{axis.upper()} axis: {exc}")
                    logger.warning("Stage/image pixel-size calibration %s-axis probe failed: %s", axis.upper(), exc)
        finally:
            self.set_x_y_position(start_x, start_y)

        if not measurements:
            raise RuntimeError(
                "Stage/image pixel-size calibration failed; no valid stage/image displacement was measured. "
                + " ".join(errors)
            )
        pixel_sizes = [float(item["pixel_size_um"]) for item in measurements]
        if len(pixel_sizes) >= 2:
            min_pixel_size = min(pixel_sizes)
            max_pixel_size = max(pixel_sizes)
            mean_pixel_size = sum(pixel_sizes) / len(pixel_sizes)
            if (max_pixel_size - min_pixel_size) / mean_pixel_size > 0.20:
                raise RuntimeError(
                    "Stage/image pixel-size calibration produced inconsistent X/Y scale estimates: "
                    + ", ".join(
                        f"{item['axis'].upper()}={item['pixel_size_um']:.6g} um/px "
                        f"({item['stage_delta_um']:.3g} um / {item['shift_px']:.3g} px, "
                        f"response={item['response']:.3g})"
                        for item in measurements
                    )
                )
            pixel_size_um = mean_pixel_size
        else:
            pixel_size_um = pixel_sizes[0]
            logger.warning(
                "Stage/image pixel-size calibration used a single-axis estimate for objective '%s': %s",
                objective_label,
                "; ".join(errors) if errors else "only one axis measured",
            )
        if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
            raise RuntimeError(f"Stage/image pixel-size calibration returned invalid pixel size: {pixel_size_um!r}.")
        self._stage_image_measured_pixel_sizes[objective_label] = float(pixel_size_um)
        logger.info(
            "Measured stage/image pixel_size_um for objective '%s': %.9g from %s",
            objective_label,
            pixel_size_um,
            "; ".join(
                f"{item['axis'].upper()} {item['stage_delta_um']:.3g} um / {item['shift_px']:.3g} px "
                f"(response={item['response']:.3g})"
                for item in measurements
            ),
        )
        return float(pixel_size_um)

    def _objective_key_for_label(self, objective_label: str) -> str:
        target_label = str(objective_label or "").strip()
        for key, entry in self.objectives.items():
            if isinstance(entry, dict) and str(entry.get("label") or "").strip() == target_label:
                return str(key)
        raise RuntimeError(
            f"Cannot use measured pixel_size_um because objective label '{target_label}' "
            "is not present in the configured objectives."
        )

    def _update_objective_pixel_size_in_memory(self, objective_label: str, pixel_size_um: float) -> str:
        objective_key = self._objective_key_for_label(objective_label)
        updated_objectives = dict(self.objectives)
        entry = dict(updated_objectives.get(objective_key) or {})
        entry["pixel_size_um"] = float(pixel_size_um)
        updated_objectives[objective_key] = entry
        self.objectives = updated_objectives
        if hasattr(self.system_config, "objectives"):
            self.system_config.objectives = json.loads(json.dumps(updated_objectives))
        return objective_key

    def _current_pixel_size_um(self, objective_label: str) -> float:
        try:
            return self._mmcore_current_pixel_size_um()
        except Exception as mmcore_exc:
            try:
                configured_pixel_size = objective_pixel_size_um(objective_label, self.system_config)
            except ValueError as config_exc:
                try:
                    measured_pixel_size = self._measure_stage_image_pixel_size_um(objective_label)
                except Exception as measurement_exc:
                    raise RuntimeError(
                        f"No valid pixel size calibration is available for objective '{objective_label}'. "
                        "MMCore did not provide an official Pixel Size Calibration, no saved positive "
                        "pixel_size_um fallback was available, and the stage/image measurement probe failed. "
                        f"MMCore error: {mmcore_exc}; configured calibration error: {config_exc}; "
                        f"measurement error: {measurement_exc}"
                    ) from measurement_exc
                objective_key = self._update_objective_pixel_size_in_memory(objective_label, measured_pixel_size)
                logger.info(
                    "Using measured pixel_size_um for objective '%s' (%s) in the current runtime only: %.9g",
                    objective_label,
                    objective_key,
                    measured_pixel_size,
                )
                return measured_pixel_size
            logger.warning(
                "Falling back to configured pixel_size_um for objective '%s' because MMCore did not provide "
                "a current pixel size: %s",
                objective_label,
                mmcore_exc,
            )
            return configured_pixel_size

    # ====== State-device control ======
    @tool_func
    def set_objective(self, objective_label: str):
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
        supported_labels = set(self.core.getStateLabels(self.objective_device))
        if target_label not in supported_labels:
            raise ValueError(
                f"Unsupported objective label: {target_label}, Available options: {sorted(supported_labels)}"
            )
        self.core.setStateLabel(self.objective_device, target_label)
        self.core.waitForDevice(self.objective_device)
        with self.device_lock:
            self.current_objective = target_label
            self.pixel_size = self._current_pixel_size_um(self.current_objective)
    @tool_func
    def get_objective(self) -> str:
        with self.device_lock:
            self.current_objective = self.core.getStateLabel(self.objective_device)
        return self.current_objective
    @tool_func
    def set_channel(self, channel: str):
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
        previous_channel = self.get_channel()
        if self._is_brightfield_channel(previous_channel) and self._supports_transmitted_brightness():
            self._user_brightness = self._clamp_brightness(self.get_brightness())
        supported_labels = set(self.core.getStateLabels(self.Dichroic))
        if target_label not in supported_labels:
            raise ValueError(
                f"Unsupported channel label: {target_label}, Available options: {sorted(supported_labels)}"
            )
        target_is_brightfield = self._is_brightfield_channel(target_label)
        if not target_is_brightfield:
            self._write_transmitted_brightness(0, require_supported=True)
        if target_label == previous_channel:
            with self.device_lock:
                self.current_channel = target_label
            if target_is_brightfield:
                self._write_transmitted_brightness(self._user_brightness)
            return
        self.core.setStateLabel(self.Dichroic, target_label)
        self.core.waitForDevice(self.Dichroic)
        with self.device_lock:
            self.current_channel = target_label
        if target_is_brightfield:
            self._write_transmitted_brightness(self._user_brightness)
        else:
            self._write_transmitted_brightness(0, require_supported=True)
    @tool_func
    def get_channel(self) -> str:
        with self.device_lock:
            self.current_channel = self.core.getStateLabel(self.Dichroic)
        return self.current_channel

    # ====== Real-time preview ======
    def start_preview(self):
        if self.shutdown_event.is_set():
            raise RuntimeError("microscope is shutting down")
        if self.preview_running and self.acquisition_thread and self.acquisition_thread.is_alive():
            return
        if self.preview_running:
            logger.warning("Preview flag was still enabled, but the acquisition thread was not alive. Restarting preview.")

        self.preview_running = True
        self.preview_stop_event.clear()
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

        acquisition_thread = self.acquisition_thread
        self.preview_running = False
        self.preview_stop_event.set()

        if (
            acquisition_thread
            and acquisition_thread.is_alive()
            and threading.current_thread() is not acquisition_thread
        ):
            acquisition_thread.join(timeout=1.0)
        thread_alive = bool(acquisition_thread and acquisition_thread.is_alive())
        if (
            not acquisition_thread
            or not thread_alive
            or threading.current_thread() is acquisition_thread
        ):
            with self.device_lock:
                if self.is_continuous:
                    self.core.stopSequenceAcquisition()
                    self.is_continuous = False
                self._restore_preview_shutter()

        self._clear_image_queue()
        with self.img_lock:
            self.latest_display_frame = None
            self.last_preview_frame_at = None
        if not thread_alive or threading.current_thread() is acquisition_thread:
            self.acquisition_thread = None
        else:
            self.acquisition_thread = acquisition_thread
            logger.warning("Preview stop requested, but acquisition thread is still alive.")

        print("Preview acquisition stopped")

    def _acquisition_loop(self):
        unexpected_exit = False
        try:
            with self.device_lock:
                self.core.startContinuousSequenceAcquisition(0)
                self.is_continuous = True
            while self.preview_running and not self.preview_stop_event.is_set() and not self.shutdown_event.is_set():
                img = None
                with self.device_lock:
                    if self.core.getRemainingImageCount() > 0:
                        img = self.core.getLastImage()
                if img is None:
                    time.sleep(0.01)
                    continue
                self._publish_preview_frame(img)
                time.sleep(0.01)
        except Exception as exc:
            unexpected_exit = True
            self.last_preview_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Preview acquisition loop failed")
        finally:
            if unexpected_exit:
                self.preview_running = False
                self.preview_stop_event.set()
                with self.img_lock:
                    self.latest_display_frame = None
                    self.last_preview_frame_at = None
            with self.device_lock:
                if self.is_continuous:
                    self.core.stopSequenceAcquisition()
                    self.is_continuous = False
                if not self.preview_running:
                    self._restore_preview_shutter()
            if threading.current_thread() is self.acquisition_thread:
                self.acquisition_thread = None
            if unexpected_exit:
                logger.warning("Preview acquisition loop exited unexpectedly: %s", self.last_preview_error)

    def _process_image_for_display(self, img):
        image_array = np.asarray(img)
        if image_array.ndim != 2 or image_array.size == 0:
            raise ValueError(f"Preview display expects a non-empty 2D image, got shape={image_array.shape}")

        try:
            if self.auto_contrast_enabled:
                low, high = np.percentile(image_array, [self.contrast_percentile, 100 - self.contrast_percentile])
                normalized = np.clip(image_array, low, high)
                normalized = (normalized - low) / (high - low + 1e-8)
            else:
                if np.issubdtype(image_array.dtype, np.integer):
                    full_scale = float(np.iinfo(image_array.dtype).max)
                else:
                    full_scale = 1.0
                normalized = image_array.astype(np.float32) / full_scale

            if self.is_16bit or normalized.dtype != np.uint8:
                display_img = (normalized * 255).astype(np.uint8)
            else:
                display_img = normalized

            color = self.dichroic_colors.get(self.current_channel, (128, 128, 128))
            if color != (128, 128, 128):
                display_float = display_img.astype(np.float32) / 255.0
                r = (color[0] * display_float).astype(np.uint8)
                g = (color[1] * display_float).astype(np.uint8)
                b = (color[2] * display_float).astype(np.uint8)
                return cv2.merge([b, g, r])
            return cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        except Exception:
            logger.exception(
                "Failed to process preview frame for display. shape=%s dtype=%s",
                getattr(image_array, "shape", None),
                getattr(image_array, "dtype", None),
            )
            return np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    def _publish_preview_frame(self, img: np.ndarray) -> None:
        """Refresh the preview cache from an arbitrary raw frame."""
        processed_img = self._process_image_for_display(img)
        with self.img_lock:
            self.latest_display_frame = processed_img.copy()
            self.last_preview_frame_at = time.monotonic()
        self.last_preview_error = ""
        if self.image_queue.full():
            try:
                self.image_queue.get_nowait()
            except Empty:
                logger.debug("Preview image queue became empty while dropping stale frame", exc_info=True)
        try:
            self.image_queue.put(processed_img, timeout=0.01)
        except Exception:
            logger.debug("Preview image queue is full; dropping an old frame", exc_info=True)

    def get_live_preview_image(self) -> Optional[np.ndarray]:
        """Only returns image array for preview display, no metadata"""
        if not self.preview_running:
            return None
        with self.img_lock:
            if self.latest_display_frame is not None:
                return self.latest_display_frame.copy()
        return None

    # ====== Image acquisition ======
    def _get_image(self, width_micro=None, height_micro=None) -> ImagingData:
        if width_micro and height_micro:
            return self._acquire_stitch_mosaic(width_micro, height_micro)
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
        except Exception as exc:
            logger.debug("Failed to read MMCore image bit depth; falling back to image dtype/range: %s", exc, exc_info=True)
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
            grid = self._calculate_stitch_grid(width_micro, height_micro, overlap=overlap)
            fov_width = float(grid["fov_width"])
            fov_height = float(grid["fov_height"])
            step_x = float(grid["step_x"])
            step_y = float(grid["step_y"])
            cols = int(grid["cols"])
            rows = int(grid["rows"])

            center_col = cols // 2
            center_row = rows // 2
            start_x = initial_x - center_col * step_x - fov_width / 2
            start_y = initial_y - center_row * step_y - fov_height / 2

            if (start_x < self.Min_X_position or start_y < self.Min_Y_position or
                start_x + (cols - 1) * step_x + fov_width > self.Max_X_position or
                start_y + (rows - 1) * step_y + fov_height > self.Max_Y_position):
                raise ValueError("Stitching area out of range")

            mosaic = np.zeros((self.current_img_height * rows, self.current_img_width * cols), dtype=self.img_dtype)
            try:
                for y_idx in range(rows):
                    y_pos = start_y + y_idx * step_y + fov_height / 2
                    x_indices = range(cols) if y_idx % 2 == 0 else reversed(range(cols))
                    for x_idx in x_indices:
                        x_pos = start_x + x_idx * step_x + fov_width / 2
                        self.set_x_y_position(x_pos, y_pos)
                        img = self._snap_raw_image()
                        y_start = y_idx * self.current_img_height
                        x_start = x_idx * self.current_img_width
                        mosaic[y_start:y_start + self.current_img_height, x_start:x_start + self.current_img_width] = img
            finally:
                self.set_x_y_position(initial_x, initial_y)

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
        width_micro = float(width_micro)
        height_micro = float(height_micro)
        overlap = float(overlap)
        if width_micro <= 0 or height_micro <= 0:
            raise ValueError("Stitch width and height must be positive")
        if not (0 <= overlap < 1):
            raise ValueError("Stitch overlap must be in the range [0, 1)")

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

    def _prepare_acquisition_records(
        self,
        *,
        channel_names: List[str],
        time_interval: float,
        num_frames: int,
        z_positions: np.ndarray,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        reference_z = float(z_positions[0]) if len(z_positions) else float(self.get_z_position())
        for position in self.acquisition_positions:
            image_height, image_width = self._resolve_position_output_shape(position)
            data = np.zeros(
                (num_frames, len(channel_names), len(z_positions), image_height, image_width),
                dtype=self.img_dtype,
            )
            metadata = self._create_ome_metadata(
                channel_names=channel_names,
                time_interval=time_interval,
                microscope="olympus lx83",
                objective=self.current_objective,
                pixel_type=self.img_dtype,
                center_x=float(position["x"]),
                center_y=float(position["y"]),
                center_z=reference_z,
            )
            records.append({
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
            })
        return records

    def _capture_position_timepoint(
        self,
        position_record: Dict[str, Any],
        *,
        time_index: int,
        z_positions: np.ndarray,
    ) -> None:
        settle_time = self._get_acquisition_settle_time()
        self.set_x_y_position(position_record["x"], position_record["y"])
        time.sleep(settle_time)
        for channel_index, channel_config in enumerate(self.acquisition_channels):
            self.set_channel(channel_config["channel"])
            self.set_exposure(channel_config["exposure"])
            for z_index, z_position in enumerate(z_positions):
                self.set_z_position(float(z_position))
                time.sleep(settle_time)
                imaging_data = self._get_image(position_record["width"], position_record["height"])
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

        channel_colors = [self.dichroic_colors.get(channel, "Unknown") for channel in channel_names]
        objective_label = str(position_record["objective_magnification"])
        objective_magnification = self.objective_labels.get(objective_label)
        desc = (
            f'"channel_names": {channel_colors}, '
            f'pixel_size: {position_record.get("pixel_size")}, '
            f'"objective_label": {objective_label}, '
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


    # ====== Auto acquisition planning and execution ======
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
        channel_label = str(channel or "").strip()
        if not channel_label:
            raise ValueError("Channel label cannot be empty")
        configured_labels = {
            str(item.get("label") or "").strip()
            for item in self.channels.values()
            if isinstance(item, dict) and str(item.get("label") or "").strip()
        }
        if configured_labels and channel_label not in configured_labels:
            raise ValueError(
                f"Channel label '{channel_label}' is not present in the confirmed configuration. "
                f"Configured labels: {sorted(configured_labels)}"
            )
        self.acquisition_channels.append({
            "channel": channel_label,
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
            time_interval = float(time_lapse_params["interval_sec"])
            z_positions = _build_z_positions(
                z_stack_params["z_start"],
                z_stack_params["z_end"],
                z_stack_params["z_step"],
            )
            channel_names = [ch["channel"] for ch in self.acquisition_channels]
            self._ensure_acquisition_image_spec()

            position_data = self._prepare_acquisition_records(
                channel_names=channel_names,
                time_interval=time_interval,
                num_frames=time_num_frames,
                z_positions=z_positions,
            )
            completed_timepoints = 0
            total_positions = max(len(position_data) * max(time_num_frames, 1), 1)
            acquisition_timeout_base = float(self.acquisition_timeout_base_seconds)
            acquisition_timeout_per_position = float(self.acquisition_timeout_per_position_seconds)
            acquisition_deadline: Optional[float]
            if acquisition_timeout_base > 0 or acquisition_timeout_per_position > 0:
                acquisition_deadline = time.monotonic() + max(
                    acquisition_timeout_base,
                    float(len(position_data)) * acquisition_timeout_per_position,
                )
            else:
                acquisition_deadline = None
            self._emit_task_progress(
                task_kind="acquisition",
                status="started",
                title="Acquisition",
                detail=f"Preparing acquisition for {len(position_data)} position(s)",
                stage_key="start",
                stage_label="Initialize acquisition",
                progress_current=0,
                progress_total=total_positions,
            )

            try:
                for t_idx in range(time_num_frames):
                    if self.shutdown_event.is_set():
                        break
                    start_time = time.time()
                    for pos_index, pos_item in enumerate(position_data, start=1):
                        if acquisition_deadline is not None:
                            self._raise_if_long_task_timed_out(
                                deadline=acquisition_deadline,
                                task_kind="acquisition",
                                stage_label="position capture",
                                detail=f"position={pos_item['name']} frame={t_idx + 1}",
                            )
                        progress_index = (t_idx * len(position_data)) + pos_index
                        self._emit_task_progress(
                            task_kind="acquisition",
                            status="running",
                            title="Acquisition",
                            detail=f"Capturing position {pos_item['name']} ({t_idx + 1}/{time_num_frames})",
                            stage_key="capture_position",
                            stage_label="Capture position",
                            progress_current=progress_index,
                            progress_total=total_positions,
                        )
                        self._capture_position_timepoint(
                            pos_item,
                            time_index=t_idx,
                            z_positions=z_positions,
                        )
                        self._emit_task_progress(
                            task_kind="acquisition",
                            status="running",
                            title="Acquisition",
                            detail=f"Completed position {pos_item['name']} ({t_idx + 1}/{time_num_frames})",
                            stage_key="position_completed",
                            stage_label="Position complete",
                            progress_current=progress_index,
                            progress_total=total_positions,
                        )
                    completed_timepoints = t_idx + 1

                    if time_num_frames > 1 and t_idx < time_num_frames - 1:
                        elapsed = time.time() - start_time
                        if elapsed > time_interval:
                            logger.warning(
                                "Time-series acquisition overran the requested interval: elapsed=%.3fs, requested_interval=%.3fs. "
                                "The next frame will start immediately.",
                                elapsed,
                                time_interval,
                            )
                        wait_time = max(0, time_interval - elapsed)
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
                            channel_names=channel_names,
                            num_frames_captured=completed_timepoints,
                        )
                    )
                self._emit_task_progress(
                    task_kind="acquisition",
                    status="completed",
                    title="Acquisition",
                    detail=f"Saved acquisition for {len(position_data)} position(s)",
                    stage_key="completed",
                    stage_label="Acquisition complete",
                    progress_current=total_positions,
                    progress_total=total_positions,
                )

            except TimeoutError as exc:
                self._emit_task_progress(
                    task_kind="acquisition",
                    status="timeout",
                    title="Acquisition",
                    detail=str(exc),
                    stage_key="timeout",
                    stage_label="Acquisition timed out",
                    progress_current=min(completed_timepoints * len(position_data), total_positions),
                    progress_total=total_positions,
                )
                raise
            except Exception as exc:
                self._emit_task_progress(
                    task_kind="acquisition",
                    status="failed",
                    title="Acquisition",
                    detail=str(exc) or type(exc).__name__,
                    stage_key="failed",
                    stage_label="Acquisition failed",
                    progress_current=min(completed_timepoints * len(position_data), total_positions),
                    progress_total=total_positions,
                )
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
        channel_colors = [self.dichroic_colors.get(ch, (128, 128, 128)) for ch in channel_names]
        metadata = {
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "time_interval": time_interval,
            "microscope": microscope,
            "objective": objective,
            "objective_label": objective,
            "objective_magnification": self.objective_labels.get(objective),
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

    # ====== Auto focus and auto brightness ======
    @tool_func
    def perform_autofocus(
        self,
        tolerance=0.5,
        use_auto_params=False,
        search_range=600.0,
        min_z: Optional[float] = None,
        max_z: Optional[float] = None,
    ) -> float:
        state = self._capture_runtime_state(include_xy=False, include_preview=True)
        captured_center_z = float(state["z"])
        configured_min_z = float(self.Min_Z_position)
        configured_max_z = float(self.Max_Z_position)
        if configured_min_z > configured_max_z:
            raise RuntimeError(
                f"Invalid configured autofocus Z range [{configured_min_z:.3f}, {configured_max_z:.3f}]"
            )
        requested_min_z = configured_min_z if min_z is None else float(min_z)
        requested_max_z = configured_max_z if max_z is None else float(max_z)
        if requested_min_z > requested_max_z:
            raise ValueError(
                f"Invalid requested autofocus Z range [{requested_min_z:.3f}, {requested_max_z:.3f}]"
            )
        effective_min_z = max(configured_min_z, requested_min_z)
        effective_max_z = min(configured_max_z, requested_max_z)
        if effective_min_z > effective_max_z:
            raise ValueError(
                "Requested autofocus Z range does not overlap the configured safe range: "
                f"requested=[{requested_min_z:.3f}, {requested_max_z:.3f}], "
                f"configured=[{configured_min_z:.3f}, {configured_max_z:.3f}]"
            )
        base_center_z = float(max(effective_min_z, min(captured_center_z, effective_max_z)))
        if abs(base_center_z - captured_center_z) >= 1e-6:
            logger.warning(
                "Autofocus center Z %.3f is outside effective range [%.3f, %.3f]; using %.3f as search center",
                captured_center_z,
                effective_min_z,
                effective_max_z,
                base_center_z,
            )
        current_channel = self.get_channel()
        current_objective = self.get_objective()
        magnification = float(self.objective_labels.get(current_objective, 10.0))
        is_fluorescence = not self._is_brightfield_channel(current_channel)
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
        autofocus_deadline = time.monotonic() + self.autofocus_timeout_seconds
        lower_bound = max(effective_min_z, base_center_z - search_range)
        upper_bound = min(effective_max_z, base_center_z + search_range)
        coarse_positions_preview = np.arange(
            lower_bound,
            upper_bound + coarse_step * 0.5,
            coarse_step,
            dtype=float,
        )
        coarse_total = max(int(coarse_positions_preview.size), 1)
        logger.info(
            "Autofocus started. objective=%s channel=%s base_z=%.3f search_range=%.3f coarse_step=%.3f tolerance=%.3f "
            "estimated_candidates=%s",
            current_objective,
            current_channel,
            base_center_z,
            search_range,
            coarse_step,
            tolerance,
            coarse_total,
        )
        self._emit_task_progress(
            task_kind="autofocus",
            status="started",
            title="Autofocus",
            detail=f"Starting autofocus around Z={base_center_z:.2f} um",
            stage_key="start",
            stage_label="Initialize autofocus",
            progress_current=0,
            progress_total=coarse_total,
        )

        def score_at(z_position: float, lower_z: float, upper_z: float) -> float:
            self._raise_if_long_task_timed_out(
                deadline=autofocus_deadline,
                task_kind="autofocus",
                stage_label="candidate evaluation",
                detail=f"candidate z={z_position:.3f}",
            )
            z_position = float(max(lower_z, min(z_position, upper_z)))
            cache_key = round(z_position, 4)
            if cache_key in scores:
                return scores[cache_key]
            candidate_index = len(scores) + 1
            logger.info(
                "Autofocus candidate start. index=%s z=%.3f range=[%.3f, %.3f]",
                candidate_index,
                z_position,
                lower_z,
                upper_z,
            )
            self._emit_task_progress(
                task_kind="autofocus",
                status="running",
                title="Autofocus",
                detail=f"Moving focus to Z={z_position:.2f} um",
                stage_key="move_z",
                stage_label="Move focus",
                progress_current=candidate_index,
                progress_total=max(coarse_total, candidate_index),
            )
            try:
                logger.info("Autofocus calling set_z_position for z=%.3f", z_position)
                self.set_z_position(z_position)
                logger.info("Autofocus set_z_position completed for z=%.3f", z_position)
            except (RuntimeError, ValueError) as exc:
                logger.warning(
                    "Autofocus skipped rejected Z candidate %.3f within requested range [%.3f, %.3f]: %s",
                    z_position,
                    lower_z,
                    upper_z,
                    exc,
                    exc_info=True,
                )
                scores[cache_key] = float("-inf")
                return scores[cache_key]
            if settle_time_sec > 0:
                time.sleep(settle_time_sec)
            self._raise_if_long_task_timed_out(
                deadline=autofocus_deadline,
                task_kind="autofocus",
                stage_label="image capture",
                detail=f"candidate z={z_position:.3f}",
            )
            logger.info("Autofocus snap start for z=%.3f", z_position)
            self._emit_task_progress(
                task_kind="autofocus",
                status="running",
                title="Autofocus",
                detail=f"Capturing image at Z={z_position:.2f} um",
                stage_key="snap_image",
                stage_label="Capture frame",
                progress_current=candidate_index,
                progress_total=max(coarse_total, candidate_index),
            )
            image = self._snap_image_preserving_preview()
            logger.info("Autofocus snap completed for z=%.3f", z_position)
            self._raise_if_long_task_timed_out(
                deadline=autofocus_deadline,
                task_kind="autofocus",
                stage_label="sharpness scoring",
                detail=f"candidate z={z_position:.3f}",
            )
            score = float(
                tool_utils.tenengrad_calculate_sharpness(
                    image,
                    center_roi_size=center_roi_size,
                )
            )
            logger.info("Autofocus score computed. z=%.3f score=%.6f", z_position, score)
            self._emit_task_progress(
                task_kind="autofocus",
                status="running",
                title="Autofocus",
                detail=f"Scored Z={z_position:.2f} um with sharpness {score:.3f}",
                stage_key="score_candidate",
                stage_label="Score sharpness",
                progress_current=candidate_index,
                progress_total=max(coarse_total, candidate_index),
            )
            scores[cache_key] = score
            return score

        def search_once(
            search_center_z: float,
            active_search_range: float,
            active_coarse_step: float,
        ) -> Tuple[float, float, float]:
            search_center_z = float(max(effective_min_z, min(search_center_z, effective_max_z)))
            lower_z = max(effective_min_z, search_center_z - active_search_range)
            upper_z = min(effective_max_z, search_center_z + active_search_range)
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

            if not math.isfinite(best_score):
                raise RuntimeError(
                    "Autofocus could not sample any valid Z position in requested range "
                    f"[{lower_z:.3f}, {upper_z:.3f}]. The MMCore focus device rejected all candidates."
                )

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
            return best_z, lower_z, upper_z

        def is_near_search_boundary(best_z: float, lower_z: float, upper_z: float, active_coarse_step: float) -> bool:
            boundary_margin = max(tolerance, active_coarse_step * 0.5)
            lower_available = lower_z > effective_min_z + tolerance
            upper_available = upper_z < effective_max_z - tolerance
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

            best_z, lower_z, upper_z = search_once(
                active_center_z,
                active_search_range,
                active_coarse_step,
            )
            near_boundary = is_near_search_boundary(best_z, lower_z, upper_z, active_coarse_step)
            while (
                use_auto_params
                and active_search_range < expansion_cap
                and expansion_round < 4
                and near_boundary
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
                best_z, lower_z, upper_z = search_once(
                    active_center_z,
                    active_search_range,
                    active_coarse_step,
                )
                expansion_round += 1
                near_boundary = is_near_search_boundary(best_z, lower_z, upper_z, active_coarse_step)

            if near_boundary:
                logger.warning(
                    "Autofocus best Z %.3f remains near search boundary [%.3f, %.3f]; "
                    "focus may be outside the searched range",
                    best_z,
                    lower_z,
                    upper_z,
                )

            self._raise_if_long_task_timed_out(
                deadline=autofocus_deadline,
                task_kind="autofocus",
                stage_label="finalize focus",
                detail=f"best z={best_z:.3f}",
            )
            self.set_z_position(best_z)
            logger.info(
                "Autofocus completed. best_z=%.3f searched_range=[%.3f, %.3f] near_boundary=%s",
                best_z,
                lower_z,
                upper_z,
                near_boundary,
            )
            self._emit_task_progress(
                task_kind="autofocus",
                status="completed",
                title="Autofocus",
                detail=f"Autofocus completed at Z={best_z:.2f} um",
                stage_key="completed",
                stage_label="Autofocus complete",
                progress_current=max(len(scores), coarse_total),
                progress_total=max(len(scores), coarse_total),
            )
            autofocus_completed = True
            return float(best_z)
        except TimeoutError as exc:
            logger.warning("Autofocus timed out: %s", exc)
            failure_progress_total = max(len(scores), coarse_total)
            self._emit_task_progress(
                task_kind="autofocus",
                status="timeout",
                title="Autofocus",
                detail=str(exc),
                stage_key="timeout",
                stage_label="Autofocus timed out",
                progress_current=min(len(scores), failure_progress_total),
                progress_total=failure_progress_total,
            )
            raise
        except Exception as exc:
            failure_progress_total = max(len(scores), coarse_total)
            self._emit_task_progress(
                task_kind="autofocus",
                status="failed",
                title="Autofocus",
                detail=str(exc) or type(exc).__name__,
                stage_key="failed",
                stage_label="Autofocus failed",
                progress_current=min(len(scores), failure_progress_total),
                progress_total=failure_progress_total,
            )
            raise
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
        autobrightness_deadline = time.monotonic() + self.autobrightness_timeout_seconds
        if not self._supports_transmitted_brightness():
            logger.warning(
                "Autobrightness skipped: transmitted-light brightness control is unavailable "
                "(device=%r, property=%r).",
                self.brightness_device,
                self.brightness_property,
            )
            return 0
        current_channel = self.get_channel()
        if not self._is_brightfield_channel(current_channel):
            logger.info(
                "Autobrightness skipped brightness search because current channel %r is not brightfield; "
                "forcing transmitted-light brightness to 0.",
                current_channel,
            )
            self.set_brightness(0)
            return 0

        min_br = int(self.Min_brightness)
        max_br = int(self.Max_brightness)
        original_brightness = int(max(min_br, min(self.get_brightness(), max_br)))
        samples: Dict[int, Dict[str, float]] = {}
        self._emit_task_progress(
            task_kind="autobrightness",
            status="started",
            title="Auto Brightness",
            detail=f"Searching brightness between {min_br} and {max_br}",
            stage_key="start",
            stage_label="Initialize brightness search",
            progress_current=0,
            progress_total=max_iterations,
        )

        def capture_metrics(brightness: int) -> Dict[str, float]:
            br = int(max(min_br, min(brightness, max_br)))
            if br in samples:
                return samples[br]
            self._raise_if_long_task_timed_out(
                deadline=autobrightness_deadline,
                task_kind="autobrightness",
                stage_label="brightness sampling",
                detail=f"brightness={br}",
            )
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

        try:
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
            sample_count = len(samples)
            self._emit_task_progress(
                task_kind="autobrightness",
                status="running",
                title="Auto Brightness",
                detail=f"Collected {sample_count} brightness samples",
                stage_key="initial_samples",
                stage_label="Capture initial samples",
                progress_current=min(sample_count, max_iterations),
                progress_total=max_iterations,
            )

            for _ in range(max_iterations):
                self._raise_if_long_task_timed_out(
                    deadline=autobrightness_deadline,
                    task_kind="autobrightness",
                    stage_label="brightness search loop",
                    detail=f"range={low}-{high}",
                )
                if high - low <= 1:
                    break
                mid = int(round((low + high) / 2))
                if mid in samples:
                    break
                self._emit_task_progress(
                    task_kind="autobrightness",
                    status="running",
                    title="Auto Brightness",
                    detail=f"Sampling brightness {mid}",
                    stage_key="sample",
                    stage_label="Sample brightness",
                    progress_current=min(len(samples) + 1, max_iterations),
                    progress_total=max_iterations,
                )
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
            self._emit_task_progress(
                task_kind="autobrightness",
                status="completed",
                title="Auto Brightness",
                detail=f"Selected brightness {best_brightness}",
                stage_key="completed",
                stage_label="Brightness search complete",
                progress_current=max(len(samples), 1),
                progress_total=max_iterations,
            )
            self.set_brightness(best_brightness)
            return int(best_brightness)
        except TimeoutError as exc:
            self._emit_task_progress(
                task_kind="autobrightness",
                status="timeout",
                title="Auto Brightness",
                detail=str(exc),
                stage_key="timeout",
                stage_label="Brightness search timed out",
                progress_current=min(len(samples), max_iterations),
                progress_total=max_iterations,
            )
            raise
        except Exception as exc:
            self._emit_task_progress(
                task_kind="autobrightness",
                status="failed",
                title="Auto Brightness",
                detail=str(exc) or type(exc).__name__,
                stage_key="failed",
                stage_label="Brightness search failed",
                progress_current=min(len(samples), max_iterations),
                progress_total=max_iterations,
            )
            raise

    # ====== System control ======

    @tool_func
    def shutdown(self):
        if self._hardware_shutdown_complete:
            return
        self.shutdown_event.set()
        if self.preview_running:
            print("Microscope shutdown: stopping preview...")
            self.stop_preview()
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            print("Microscope shutdown: waiting for acquisition thread...")
            self.acquisition_thread.join(timeout=5.0)
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            raise RuntimeError("microscope acquisition thread did not stop within 5 seconds")
        with self.device_lock:
            print("Microscope shutdown: resetting hardware core...")
            self._write_transmitted_brightness(self.Min_brightness)
            with _silence_native_stdio():
                self.core.stopSequenceAcquisition()
                self.core.reset()
                self.core.unloadAllDevices()
            print("Microscope shutdown: hardware core reset complete.")
        self._hardware_shutdown_complete = True
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
        if not isinstance(loaded_data, list):
            raise ValueError(f"Target location file must contain a JSON array: {filepath}")

        regions = []
        for item in loaded_data:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                x, y, width, height = map(float, item)
                regions.append((x, y, width, height))

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
        current_objective = self.get_objective()
        magnification = float(self.objective_labels.get(current_objective, 10.0))
        params = z_stack_scan_params(magnification, not self._is_brightfield_channel(orig_channel))
        fallback_half_width = float(params["min_width"]) / 2.0

        was_preview_running = bool(state["preview_running"])
        if not was_preview_running:
            self.start_preview()
            time.sleep(0.5)

        z_start = clamp_z(orig_z - params["range"])
        z_end = clamp_z(orig_z + params["range"])
        z_step = float(params["step"])
        z_positions = _build_z_positions(z_start, z_end, z_step)
        if z_positions.size == 0:
            return (
                clamp_z(orig_z + fallback_half_width),
                clamp_z(orig_z - fallback_half_width),
            )

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
            return (
                clamp_z(orig_z + fallback_half_width),
                clamp_z(orig_z - fallback_half_width),
            )

        z_vals, scores = zip(*sharpness_samples)
        z_vals = np.asarray(z_vals, dtype=float)
        scores = np.asarray(scores, dtype=float)
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
            peak_z = float(z_vals[peak_idx])
            return (
                clamp_z(peak_z + fallback_half_width),
                clamp_z(peak_z - fallback_half_width),
            )

        threshold = baseline + float(params["threshold_ratio"]) * score_span
        above = scores_smooth >= threshold
        regions = contiguous_true_regions(above)
        if not regions:
            logger.warning("Z-stack range scan found no high-sharpness plateau")
            peak_z = float(z_vals[peak_idx])
            return (
                clamp_z(peak_z + fallback_half_width),
                clamp_z(peak_z - fallback_half_width),
            )

        peak_region = next(
            (region for region in regions if region[0] <= peak_idx <= region[1]),
            None,
        )
        if peak_region is None:
            region_start, region_end = max(
                regions,
                key=lambda region: float(np.sum(scores_smooth[region[0]:region[1] + 1])),
            )
        else:
            region_start, region_end = peak_region

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

    # ====== Detection execution ======
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
        target_name = str(target_class or "").strip()
        if not target_name:
            raise ValueError("target_class cannot be empty")

        if image_data.pixel_size is None or float(image_data.pixel_size) <= 0:
            raise ValueError("image_data.pixel_size must be a positive number")

        image_2d = _coerce_detection_image_to_2d(image_data.image)
        pixel_size = float(image_data.pixel_size)
        # Lazy import to avoid hanging at module load time
        try:
            from mmdet.apis import init_detector, inference_detector
        except Exception as exc:
            raise RuntimeError(
                "MMDetection dependencies are unavailable. Please install a compatible "
                "mmdet/mmcv/torch stack before using detect_targets_in_image."
            ) from exc

        if torch is None:
            raise RuntimeError(
                "PyTorch is unavailable. Please install torch before using detect_targets_in_image."
            )

        h, w = image_2d.shape
        img_center_x_px = (w - 1) / 2.0
        img_center_y_px = (h - 1) / 2.0
        image_center_x_um = float(image_data.center_x)
        image_center_y_um = float(image_data.center_y)

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        spec = _resolve_detection_target_spec(target_name, self.detection_targets)
        config_path = str(spec.get("model_config") or "")
        ckpt_path = str(spec.get("model_checkpoint") or "")
        target_class_name = str(spec.get("target_class_name") or target_name)
        if not config_path or not ckpt_path:
            raise ValueError(f"Target class '{target_name}' is not configured")

        try:
            model = self._detection_models.get(target_name)
            if model is None:
                model = init_detector(config_path, ckpt_path, device=device)
                self._detection_models[target_name] = model
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize MMDetection model for '{target_name}': {exc}") from exc

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
            raise RuntimeError(f"Failed to run MMDetection inference for '{target_name}': {exc}") from exc

        classes = _resolve_model_classes(model)
        class_idx = _resolve_detection_class_index(classes, target_class_name)
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
            # Match the OME, Cellpose, Fiji, and MP285 image-to-stage convention.
            offset_y_um = (cy_px - img_center_y_px) * pixel_size
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
