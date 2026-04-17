import os
import tempfile
import time
import warnings
import logging
import importlib
from pathlib import Path

import imagej
import numpy as np
import scyjava as sj
import scyjava.config as sjconf
import tifffile
from scyjava import jimport
import json
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Optional
from bootstrap.config import load_detection_targets
from config.system_config import (
    # PSF and FIJI paths
    PSF_40X,
    PSF_60X,
    PSF_100X,
    FIJI_PATH,
    MAVEN_BIN,
)

try:
    from aicsimageio.types import PhysicalPixelSizes
except ImportError:  # pragma: no cover - optional dependency for OME writing paths
    @dataclass
    class PhysicalPixelSizes:
        Z: Optional[float] = None
        Y: Optional[float] = None
        X: Optional[float] = None
from core_tool.spatial_metadata import load_ome_spatial_metadata

logger = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[1]

_TORCH_MODULE = None
_CV2_MODULE = None
_MMDET_APIS = None


DEFAULT_DETECTION_TILE_SIZE = 2048
DEFAULT_DETECTION_TILE_OVERLAP = 512
DEFAULT_GLOBAL_TILE_IOU_THRESHOLD = 0.2
DEFAULT_TILE_EDGE_MARGIN = 32.0



def _iter_detection_tiles(image_shape: tuple[int, int], tile_size: int, tile_overlap: int = 0):
    height, width = image_shape
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if tile_overlap < 0 or tile_overlap >= tile_size:
        raise ValueError("tile_overlap must be >= 0 and smaller than tile_size")
    stride = tile_size - tile_overlap
    seen: set[tuple[int, int, int, int]] = set()
    for y0 in range(0, max(height, 1), stride):
        for x0 in range(0, max(width, 1), stride):
            y1 = min(height, y0 + tile_size)
            x1 = min(width, x0 + tile_size)
            if y1 <= y0 or x1 <= x0:
                continue
            key = (y0, y1, x0, x1)
            if key in seen:
                continue
            seen.add(key)
            yield key


def _box_iou_xywh_tuple(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax1, ay1, ax2, ay2 = ax - aw / 2.0, ay - ah / 2.0, ax + aw / 2.0, ay + ah / 2.0
    bx1, by1, bx2, by2 = bx - bw / 2.0, by - bh / 2.0, bx + bw / 2.0, by + bh / 2.0
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _deduplicate_pixel_regions(
    regions: List[Tuple[float, float, float, float]],
    *,
    iou_threshold: float,
) -> List[Tuple[float, float, float, float]]:
    deduped: List[Tuple[float, float, float, float]] = []
    for candidate in sorted(regions, key=lambda item: float(item[2]) * float(item[3]), reverse=True):
        if any(_box_iou_xywh_tuple(candidate, existing) >= iou_threshold for existing in deduped):
            continue
        deduped.append(candidate)
    return deduped


def _touches_internal_tile_edge(
    *,
    center_x_px: float,
    center_y_px: float,
    width_px: float,
    height_px: float,
    tile_width: int,
    tile_height: int,
    tile_x0: int,
    tile_y0: int,
    image_width: int,
    image_height: int,
    edge_margin: float,
) -> bool:
    x1 = float(center_x_px) - float(width_px) / 2.0
    y1 = float(center_y_px) - float(height_px) / 2.0
    x2 = float(center_x_px) + float(width_px) / 2.0
    y2 = float(center_y_px) + float(height_px) / 2.0
    margin = max(0.0, float(edge_margin))

    touches_left = tile_x0 > 0 and x1 <= margin
    touches_top = tile_y0 > 0 and y1 <= margin
    touches_right = (tile_x0 + tile_width) < image_width and x2 >= float(tile_width) - margin
    touches_bottom = (tile_y0 + tile_height) < image_height and y2 >= float(tile_height) - margin
    return touches_left or touches_top or touches_right or touches_bottom


def _prepare_java_cache_dirs() -> None:
    """
    Use workspace-local caches for cjdk/jgo/Maven to avoid failures caused by
    user-profile cache directory permissions on Windows.
    """
    runtime_root = ROOT_DIR / ".runtime"
    cjdk_cache_dir = runtime_root / "cjdk_cache"
    jgo_cache_dir = runtime_root / "jgo"
    m2_repo_dir = Path.home() / ".m2" / "repository"

    for path in (runtime_root, cjdk_cache_dir, jgo_cache_dir, m2_repo_dir):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["CJDK_CACHE_DIR"] = str(cjdk_cache_dir)
    sjconf.set_java_constraints(fetch="never")
    sjconf.set_cache_dir(jgo_cache_dir)
    sjconf.set_m2_repo(m2_repo_dir)


def _resolve_project_path(path_value: str) -> str:
    """
    Resolve a config path relative to the project root so subprocess workdirs do
    not break resource lookups.
    """
    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((ROOT_DIR / candidate).resolve())


def _ensure_maven_on_path() -> None:
    """
    Prepend the configured Maven bin directory to PATH so jgo can find `mvn`
    consistently regardless of how the app was launched.
    """
    if not MAVEN_BIN:
        return
    if not os.path.isdir(MAVEN_BIN):
        logger.warning("Configured MAVEN_BIN does not exist: %s", MAVEN_BIN)
        return

    current_path = os.environ.get("PATH", "")
    entries = current_path.split(os.pathsep) if current_path else []
    normalized_target = os.path.normcase(os.path.normpath(MAVEN_BIN))
    normalized_entries = {
        os.path.normcase(os.path.normpath(entry))
        for entry in entries
        if entry.strip()
    }
    if normalized_target not in normalized_entries:
        os.environ["PATH"] = MAVEN_BIN + os.pathsep + current_path if current_path else MAVEN_BIN


def _rescale_contrast_per_plane(
    data: np.ndarray,
    *,
    saturated=0.2,
    low_percentile=None,
    high_percentile=None,
    low_value=None,
    high_value=None,
) -> np.ndarray:
    """Rescale intensities independently for each 2D plane across the last two axes."""
    array = np.asarray(data)
    if array.size == 0:
        raise ValueError("Cannot adjust contrast for an empty image.")

    valid_mask = np.isfinite(array)
    if not np.any(valid_mask):
        raise ValueError("Cannot adjust contrast because the image contains no finite pixels.")

    if low_percentile is not None and low_value is not None:
        raise ValueError("Specify either low_percentile or low_value, not both.")
    if high_percentile is not None and high_value is not None:
        raise ValueError("Specify either high_percentile or high_value, not both.")

    working = array.astype(np.float32, copy=False)
    if np.issubdtype(array.dtype, np.integer):
        dtype_info = np.iinfo(array.dtype)
        out_min = float(dtype_info.min)
        out_max = float(dtype_info.max)
    else:
        out_min = 0.0
        out_max = 1.0

    if working.ndim < 2:
        plane_count = 1
        planes = working.reshape(1, -1)
        plane_valid = valid_mask.reshape(1, -1)
    else:
        plane_count = int(np.prod(working.shape[:-2], dtype=np.int64)) or 1
        planes = working.reshape(plane_count, -1)
        plane_valid = valid_mask.reshape(plane_count, -1)

    adjusted_planes = np.empty_like(planes, dtype=np.float32)
    use_auto_bounds = (
        low_percentile is None
        and high_percentile is None
        and low_value is None
        and high_value is None
    )

    if use_auto_bounds:
        saturated = float(saturated)
        if not 0 <= saturated < 100:
            raise ValueError("saturated must be within [0, 100).")
        tail_percent = saturated / 2.0
    else:
        if low_percentile is not None:
            low_percentile = float(low_percentile)
            if not 0 <= low_percentile <= 100:
                raise ValueError("low_percentile must be within [0, 100].")
        if high_percentile is not None:
            high_percentile = float(high_percentile)
            if not 0 <= high_percentile <= 100:
                raise ValueError("high_percentile must be within [0, 100].")

    for idx in range(plane_count):
        current_plane = planes[idx]
        current_valid = plane_valid[idx]
        if not np.any(current_valid):
            adjusted_planes[idx] = 0.0
            continue

        valid_values = current_plane[current_valid]
        if use_auto_bounds:
            if saturated > 0:
                low, high = np.percentile(valid_values, [tail_percent, 100.0 - tail_percent])
            else:
                low = float(valid_values.min())
                high = float(valid_values.max())
        else:
            if low_percentile is not None:
                low = float(np.percentile(valid_values, low_percentile))
            elif low_value is not None:
                low = float(low_value)
            else:
                low = float(valid_values.min())

            if high_percentile is not None:
                high = float(np.percentile(valid_values, high_percentile))
            elif high_value is not None:
                high = float(high_value)
            else:
                high = float(valid_values.max())

        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("Computed contrast bounds are not finite.")

        if high <= low:
            logger.warning(
                "Contrast adjustment skipped for plane %s because it has no dynamic range (low=%s, high=%s).",
                idx,
                low,
                high,
            )
            adjusted_planes[idx] = np.where(current_valid, current_plane, 0.0)
            continue

        scaled = np.clip((current_plane - low) / (high - low), 0.0, 1.0)
        scaled = scaled * (out_max - out_min) + out_min
        adjusted_planes[idx] = np.where(current_valid, scaled, 0.0)

    adjusted = adjusted_planes.reshape(array.shape)
    if np.issubdtype(array.dtype, np.integer):
        return np.rint(adjusted).astype(array.dtype, copy=False)
    if np.issubdtype(array.dtype, np.floating):
        return adjusted.astype(array.dtype, copy=False)
    return adjusted.astype(np.float32, copy=False)


def _default_inference_device() -> str:
    torch = _get_torch()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _safe_empty_cuda_cache() -> None:
    torch = _get_torch()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_torch():
    global _TORCH_MODULE
    if _TORCH_MODULE is None:
        _TORCH_MODULE = importlib.import_module("torch")
    return _TORCH_MODULE


def _get_cv2():
    global _CV2_MODULE
    if _CV2_MODULE is None:
        _CV2_MODULE = importlib.import_module("cv2")
    return _CV2_MODULE


def _get_mmdet_apis():
    global _MMDET_APIS
    if _MMDET_APIS is None:
        apis = importlib.import_module("mmdet.apis")
        _MMDET_APIS = (apis.init_detector, apis.inference_detector)
    return _MMDET_APIS


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


def _boxes_xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f"Expected boxes with shape (N, 4), got {boxes.shape}")

    xywh = boxes.copy()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]
    return xywh


def _normalize_nms_indices(indices: Any) -> list[int]:
    if indices is None:
        return []
    indices_array = np.asarray(indices)
    if indices_array.size == 0:
        return []
    return [int(index) for index in indices_array.reshape(-1).tolist()]


def _format_imagej_macro_path(path: str) -> str:
    normalized = str(path).replace("\\", "/")
    return "[" + normalized.replace("]", "\\]") + "]"


def _resolve_imagej_save_format(path: str) -> tuple[str, str]:
    lowered = str(path).lower()
    if lowered.endswith(".ome.tif") or lowered.endswith(".ome.tiff"):
        return "OME-TIFF", "tiff"
    if lowered.endswith(".tif") or lowered.endswith(".tiff"):
        return "Tiff", "tiff"
    if lowered.endswith(".png"):
        return "PNG", "png"
    if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        return "Jpeg", "jpg"
    raise ValueError(
        f"Unsupported output format for path {path!r}. "
        "Supported suffixes: .ome.tif, .ome.tiff, .tif, .tiff, .png, .jpg, .jpeg"
    )


def _resolve_detector_classes(model: Any) -> list[str]:
    dataset_meta = getattr(model, "dataset_meta", None)
    if isinstance(dataset_meta, dict):
        classes = dataset_meta.get("classes")
        if classes:
            return list(classes)

    classes = getattr(model, "CLASSES", None)
    if classes:
        return list(classes)

    raise RuntimeError("MMDetection model does not expose class metadata")


def _select_target_class_index(
    model: Any,
    *,
    target_class_name: Optional[str] = None,
    target_class_id: Optional[int] = None,
) -> int:
    classes = _resolve_detector_classes(model)
    if target_class_name and target_class_name in classes:
        return classes.index(target_class_name)
    if target_class_id is not None and 0 <= target_class_id < len(classes):
        return target_class_id
    if len(classes) == 1:
        return 0
    raise RuntimeError(
        f"Unable to resolve target class from model classes {classes}. "
        f"Requested name={target_class_name!r}, id={target_class_id!r}."
    )


def _extract_filtered_pred_instances(result: Any, score_thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_instances = getattr(result, "pred_instances", None)
    if pred_instances is None:
        raise RuntimeError("MMDetection 3.x result must expose pred_instances")

    bboxes = _to_numpy_array(getattr(pred_instances, "bboxes", None))
    scores = _to_numpy_array(getattr(pred_instances, "scores", None))
    labels = _to_numpy_array(getattr(pred_instances, "labels", None)).astype(int, copy=False)

    if bboxes.size == 0 or scores.size == 0 or labels.size == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    score_mask = scores >= score_thr
    if not np.any(score_mask):
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    return (
        np.asarray(bboxes[score_mask], dtype=np.float32),
        np.asarray(scores[score_mask], dtype=np.float32),
        np.asarray(labels[score_mask], dtype=np.int32),
    )


def _resolve_target_detection_spec(target_type: str) -> dict[str, Any]:
    target_specs = load_detection_targets()
    normalized = str(target_type).strip()
    if normalized in target_specs:
        return dict(target_specs[normalized])

    lowered = normalized.lower()
    for key, spec in target_specs.items():
        if key.lower() == lowered:
            return dict(spec)

    raise ValueError(
        f"Unsupported target type: {target_type}. "
        f"Available targets: {', '.join(target_specs.keys())}"
    )


def _list_supported_target_types() -> list[str]:
    return sorted(load_detection_targets().keys(), key=str.lower)


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
        """For compatibility with legacy logic, return x-direction pixel size"""
        return self.pixel_size_x_um

from tool.base import BaseTool, tool_func
class ImageJProcessor(BaseTool):
    """
    Synchronous version: Image processing utility class based on ImageJ/Fiji.
    All methods are synchronous calls, suitable for use in ordinary scripts or main threads.
    Optimizations: Resolve hardcoding, repeated model initialization, fake file registration, resource leaks, etc.
    """

    def __init__(self, storagemanger, output_path: str):
        self._storagemanger = storagemanger
        self.output_directory: str = output_path
        self.ij = None
        # Class attribute to cache MMDetection models and avoid repeated initialization
        self._organoid_model = None
        self._interaction_artifact_listener: Optional[Callable[[dict[str, Any]], None]] = None

    def _require_imagej_initialized(self):
        if self.ij is None:
            raise RuntimeError("ImageJ not initialized, please call fiji_initialize() first")
        return self.ij

    def set_interaction_artifact_listener(
        self,
        listener: Optional[Callable[[dict[str, Any]], None]],
    ) -> None:
        self._interaction_artifact_listener = listener

    def _emit_interaction_artifact(
        self,
        *,
        path: str,
        title: str,
        text: str = "",
    ) -> None:
        if self._interaction_artifact_listener is None:
            return
        try:
            self._interaction_artifact_listener(
                {
                    "kind": "image",
                    "path": path,
                    "title": title,
                    "text": text,
                    "source": "fiji_target_detection",
                }
            )
        except Exception:
            logger.exception("Failed to emit Fiji interaction artifact for %s", path)

    @tool_func
    def fiji_initialize(self, fiji_path=FIJI_PATH):
        """Synchronously initialize ImageJ environment (directly inline private interface logic without hierarchical calls)"""
        print("Initializing ImageJ environment...")
        if not os.path.exists(fiji_path):
            raise FileNotFoundError(f"Fiji.app path does not exist: {fiji_path}")
        _ensure_maven_on_path()
        _prepare_java_cache_dirs()
        self.ij = imagej.init(fiji_path, mode=imagej.Mode.INTERACTIVE)
        print(f"ImageJ version: {self.ij.getVersion()}")

    # ----------------- File IO -----------------    
    @tool_func
    def load_image(self, file_name: str) -> ImageWithMetadata:
        """
        Load an OME-TIFF image and return an ImageWithMetadata object.
        This is compatible with files saved by _save_ome_tiff and reads:
        1. Physical pixel sizes for X/Y/Z
        2. Acquisition position metadata for the XYZ center coordinates
        Args:
            file_name: OME-TIFF filename. The full path is resolved under self.output_directory.
        Returns:
            ImageWithMetadata: The loaded image dataset together with its metadata.
        Raises:
            FileNotFoundError: Raised when the file does not exist.
            RuntimeError: Raised when reading or parsing metadata fails.
        """
        # 1. Resolve the full file path and verify that it exists.
        file_path = os.path.join(self.output_directory, file_name)
        self._require_imagej_initialized()
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file does not exist: {file_path}")

        try:
            # 2. Open the raw dataset with ImageJ.
            dataset = self.ij.io().open(file_path)

            spatial_meta = load_ome_spatial_metadata(file_path)

            # 4. Build and return the ImageWithMetadata object.
            meta = ImageWithMetadata(
                dataset=dataset,
                center_x_um=float(spatial_meta["center_x_um"]),
                center_y_um=float(spatial_meta["center_y_um"]),
                center_z_um=float(spatial_meta["center_z_um"]),
                pixel_size_x_um=float(spatial_meta["pixel_size_x_um"]),
                pixel_size_y_um=float(spatial_meta["pixel_size_y_um"]),
            )

            return meta

        except Exception as e:
            # Wrap all errors to make image-loading issues easier to diagnose.
            raise RuntimeError(
                f"Failed to read OME-TIFF file {file_path}: {str(e)}"
            ) from e
    
    def _load_image_IMP(self, file_path):
        """Internal method: Load ImagePlus object (no upper hierarchical calls, directly used)"""
        self._require_imagej_initialized()
        formatted_path = _format_imagej_macro_path(file_path)
        macro = (
            'run("Bio-Formats Importer", '
            f'"open={formatted_path} autoscale color_mode=Default '
            'rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");'
        )
        self.ij.py.run_macro(macro)
        imp = self.ij.py.active_imageplus()
        if imp is None:
            raise IOError(f"Failed to load image file: {file_path}")
        return imp

    def dataset_to_imp(self, dataset):
        """Synchronously convert Dataset to ImagePlus (directly inline private interface logic without hierarchical calls)"""
        self._require_imagej_initialized()
        self.dump_info(dataset)
        np_xarray = self.ij.py.to_xarray(dataset)
        self.dump_info(np_xarray)

        target_dims = ('t', 'pln', 'ch', 'row', 'col')
        for dim in target_dims:
            if dim not in np_xarray.dims:
                np_xarray = np_xarray.expand_dims({dim: 1})

        np_xarray = np_xarray.transpose('t', 'pln', 'ch', 'row', 'col')
        self.dump_info(np_xarray)

        # Optimize temporary file management with with statement to avoid missing deletions
        with tempfile.NamedTemporaryFile(suffix='.ome.tif', delete=False) as tmpfile:
            temp_path = tmpfile.name

        try:
            tifffile.imwrite(
                temp_path,
                np_xarray.data,
                imagej=True,
                metadata={'axes': 'TZCYX'}
            )
            temp_path = temp_path.replace("\\", "/")
            imp = self._load_image_IMP(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return imp

    @tool_func
    def save_image(self, image_meta: ImageWithMetadata, filename: str, description: str):
        """Save image and register file"""
        self._save_dataset_impl(image_meta.dataset, filename)
        self._storagemanger.register_file(filename, description, 'analysis_platform', 'tiff', False)

    def _save_dataset_impl(self, dataset, filename):
        # Core logic of the original save_image
        self._require_imagej_initialized()
        self.dump_info(dataset)
        np_xarray = self.ij.py.to_xarray(dataset)
        target_dims = ('t', 'pln', 'ch', 'row', 'col')
        for dim in target_dims:
            if dim not in np_xarray.dims:
                np_xarray = np_xarray.expand_dims({dim: 1})
        np_xarray = np_xarray.transpose('t', 'pln', 'ch', 'row', 'col')
        output_path = os.path.join(self.output_directory, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        tifffile.imwrite(output_path, np_xarray.data, imagej=True, metadata={'axes': 'TZCYX'})

    # ----------------- Contrast Enhancement -----------------

    @tool_func
    def adjust_contrast(
        self,
        image_meta: ImageWithMetadata,
        saturated=0.2,
        low_percentile=None,
        high_percentile=None,
        low_value=None,
        high_value=None,
    ) -> ImageWithMetadata:
        """Synchronously rescale pixel intensities and return image with metadata."""
        print("Performing automatic contrast adjustment on pixel intensities...")
        new_dataset = self._adjust_contrast_impl(
            image_meta.dataset,
            saturated=saturated,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            low_value=low_value,
            high_value=high_value,
        )
        return ImageWithMetadata(
            dataset=new_dataset,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            center_z_um=image_meta.center_z_um,
            pixel_size_x_um=image_meta.pixel_size_x_um,
            pixel_size_y_um=image_meta.pixel_size_y_um,
        )

    def _adjust_contrast_impl(
        self,
        img,
        saturated=0.2,
        low_percentile=None,
        high_percentile=None,
        low_value=None,
        high_value=None,
    ):
        """Internal implementation that persistently rescales image pixels."""
        self._require_imagej_initialized()
        Dataset = jimport('net.imagej.Dataset')
        dataset = img if hasattr(img, 'getImgPlus') else self.ij.convert().convert(img, Dataset)

        np_xarray = self.ij.py.to_xarray(dataset)
        data = np.asarray(np_xarray.data)
        adjusted = _rescale_contrast_per_plane(
            data,
            saturated=saturated,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            low_value=low_value,
            high_value=high_value,
        )
        return self._dataset_from_array_with_same_axes(np_xarray, adjusted)

    def _dataset_from_array_with_same_axes(self, np_xarray, adjusted_array):
        """Create a Fiji Dataset from a NumPy array while preserving the expected axis order."""
        target_dims = ('t', 'pln', 'ch', 'row', 'col')
        normalized_xarray = np_xarray.copy(data=adjusted_array)
        for dim in target_dims:
            if dim not in normalized_xarray.dims:
                normalized_xarray = normalized_xarray.expand_dims({dim: 1})
        normalized_xarray = normalized_xarray.transpose('t', 'pln', 'ch', 'row', 'col')

        with tempfile.NamedTemporaryFile(suffix='.ome.tif', delete=False) as tmpfile:
            temp_path = tmpfile.name

        try:
            tifffile.imwrite(
                temp_path,
                normalized_xarray.data,
                imagej=True,
                metadata={'axes': 'TZCYX'}
            )
            return self.ij.io().open(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def dump_info(self, image):
        """Print image information (no hierarchical calls, direct implementation)"""
        print(f" type: {type(image)}")
        print(f"dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
        print(f"shape: {image.shape if hasattr(image, 'shape') else 'N/A'}")
        print(f" dims: {image.dims if hasattr(image, 'dims') else 'N/A'}")

    def _imageplus_to_numpy_2d(self, imp, *, normalize_uint8: bool = False) -> np.ndarray:
        """Extract the current ImagePlus plane as a 2D numpy array."""
        processor = imp.getProcessor()
        if processor is None:
            raise ValueError("Image processor is None; cannot extract pixels.")

        ImagePlus = jimport('ij.ImagePlus')
        if imp.getType() == ImagePlus.COLOR_RGB:
            pixel_source = processor.convertToByteProcessor().getPixels()
        else:
            pixel_source = processor.getPixels()

        pixels = self.ij.py.from_java(pixel_source)
        if pixels is None:
            raise ValueError("Failed to extract pixel data from ImagePlus.")

        width = int(imp.getWidth())
        height = int(imp.getHeight())
        pixel_array = np.asarray(pixels)
        expected_total_pixels = width * height
        if pixel_array.size != expected_total_pixels:
            raise ValueError(
                f"Pixel count mismatch for current plane: expected {expected_total_pixels}, got {pixel_array.size}"
            )

        img_array = pixel_array.reshape((height, width))
        if normalize_uint8:
            return self._safe_image_normalize(np.asarray(img_array, dtype=np.float32)).astype(np.uint8, copy=False)
        return img_array

    # ----------------- Channel Processing -----------------
    @tool_func
    def split_channels(self, image_meta: ImageWithMetadata) -> List[ImageWithMetadata]:
        """Split channels, each channel retains the same metadata"""
        datasets = self._split_channels_impl(image_meta.dataset)
        return [
            ImageWithMetadata(
                dataset=ds,
                center_x_um=image_meta.center_x_um,
                center_y_um=image_meta.center_y_um,
                center_z_um=image_meta.center_z_um,
                pixel_size_x_um=image_meta.pixel_size_x_um,
                pixel_size_y_um=image_meta.pixel_size_y_um,
            )
            for ds in datasets
        ]

    def _split_channels_impl(self, img):
        """
        Split a multi-channel image into a list of single-channel Datasets.

        Parameters:
            img (net.imagej.Dataset): Input multi-channel image (must contain CHANNEL axis)

        Returns:
            List[net.imagej.Dataset]: List of single-channel images
        """
        self._require_imagej_initialized()
        ChannelSplitter = jimport('ij.plugin.ChannelSplitter')
        Dataset = jimport('net.imagej.Dataset')

        imp = None
        owns_imp = False
        split_imps = []

        try:
            if hasattr(img, 'getImgPlus'):
                imp = self.dataset_to_imp(img)
                owns_imp = True
            else:
                imp = img

            if not hasattr(imp, 'getNChannels'):
                raise TypeError(f"Unsupported image type for channel splitting: {type(img)}")

            num_channels = int(imp.getNChannels())
            if num_channels <= 1:
                if owns_imp:
                    return [self.ij.convert().convert(imp, Dataset)]
                return [img]

            # Use Fiji's native ImageJ1 splitter instead of ij.op().create(), which
            # resolves to an incompatible SciJava PTService overload in some Fiji builds.
            split_imps = list(ChannelSplitter.split(imp))
            return [
                self.ij.convert().convert(channel_imp, Dataset)
                for channel_imp in split_imps
            ]
        finally:
            for channel_imp in split_imps:
                try:
                    channel_imp.close()
                except Exception:
                    pass
            if owns_imp and imp is not None:
                try:
                    imp.close()
                except Exception:
                    pass

    @tool_func
    def merge_channels(
        self,
        image_metas: List[ImageWithMetadata],
        colors=None,
        outpath='merge_output.ome.tif',
        preview_path: Optional[str] = None,
        preview_seconds: float = 1.5,
    ) -> ImageWithMetadata:
        """Merge channels and use metadata from the first image"""
        if not image_metas:
            raise ValueError("Input image list is empty")
        
        # Use metadata from the first image (assuming all channels are spatially aligned)
        ref_meta = image_metas[0]
        datasets = [meta.dataset for meta in image_metas]
        
        merged_dataset = self._merge_channels_impl(
            datasets,
            colors,
            outpath,
            preview_path=preview_path,
            preview_seconds=preview_seconds,
        )
        
        return ImageWithMetadata(
            dataset=merged_dataset,
            center_x_um=ref_meta.center_x_um,
            center_y_um=ref_meta.center_y_um,
            center_z_um=ref_meta.center_z_um,
            pixel_size_x_um=ref_meta.pixel_size_x_um,
            pixel_size_y_um=ref_meta.pixel_size_y_um,
        )

    def _merge_channels_impl(
        self,
        datasets,
        colors=None,
        outpath='merge_output.ome.tif',
        preview_path: Optional[str] = None,
        preview_seconds: float = 0.0,
    ):
        """Synchronously merge channels (no hierarchical calls, direct implementation of core logic)"""
        self._require_imagej_initialized()
        if colors is None:
            colors = ['Red', 'Green', 'Blue'][:len(datasets)]
        imps = []
        for idx, ds in enumerate(datasets):
            imp = self.dataset_to_imp(ds)
            imps.append(imp)
        colors_map = ['Red', 'Green', 'Blue', 'Cyan', 'Magenta', 'Yellow']

        parts = []
        for color, imp in zip(colors, imps):
            if color not in colors_map:
                raise ValueError(f"Unsupported color: {color}, available colors: {colors_map}")
            i = colors_map.index(color)
            parts.append(f"c{i + 1}={imp.getTitle()}")
        parts.append("create")
        outpath = os.path.join(self.output_directory, outpath)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        format_type, file_type = _resolve_imagej_save_format(outpath)
        escaped_path = outpath.replace('\\', '\\\\')

        macro = (
            f'run("Merge Channels...", "{" ".join(parts)}"); '
            f'run("Stack to RGB", "slices frames"); '
        )

        merged_imp = None
        try:
            print(f"Executing merge channels macro: {macro}")
            self.ij.py.run_macro(macro)

            merged_imp = self.ij.py.active_imageplus()
            if merged_imp is None:
                raise RuntimeError("Merge Channels did not produce an active ImagePlus")

            preview_seconds = max(0.0, min(float(preview_seconds), 3.0))
            if preview_seconds > 0:
                try:
                    merged_imp.show()
                except Exception:
                    logger.debug("Failed to show merged ImagePlus preview", exc_info=True)
                time.sleep(preview_seconds)

            self.ij.IJ.saveAs(merged_imp, format_type, outpath)

            if preview_path:
                preview_abs_path = os.path.join(self.output_directory, preview_path)
                os.makedirs(os.path.dirname(preview_abs_path), exist_ok=True)
                preview_format, preview_file_type = _resolve_imagej_save_format(preview_abs_path)
                self.ij.IJ.saveAs(merged_imp, preview_format, preview_abs_path)
                self._storagemanger.register_file(
                    os.path.basename(preview_abs_path),
                    f"Preview image after merging channels {colors}",
                    'analysis_platform',
                    preview_file_type,
                    False,
                )

            merged_ds = self.ij.convert().convert(merged_imp, jimport('net.imagej.Dataset'))
            description = f'Image after merging channels {colors}'
            self._storagemanger.register_file(os.path.basename(outpath), description, 'analysis_platform', file_type, False)
            return merged_ds
        finally:
            for imp in imps:
                try:
                    imp.close()
                except Exception:
                    pass
            if merged_imp:
                try:
                    merged_imp.close()
                except Exception:
                    pass

    @tool_func
    def set_lut(self, image_meta: ImageWithMetadata, color_name: str) -> ImageWithMetadata:
        """
        Synchronously set LUT (receive and return ImageWithMetadata)
        """
        print(f"Setting LUT to: {color_name}")
        new_dataset = self._set_lut_impl(image_meta.dataset, color_name)
        return ImageWithMetadata(
            dataset=new_dataset,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            center_z_um=image_meta.center_z_um,
            pixel_size_x_um=image_meta.pixel_size_x_um,
            pixel_size_y_um=image_meta.pixel_size_y_um,
        )

    def _set_lut_impl(self, img, color_name: str):
        """Internal implementation: only process image, no involvement with metadata"""
        self._require_imagej_initialized()
        if hasattr(img, 'getImgPlus'):
            imp = self.dataset_to_imp(img)
        else:
            imp = img

        ImagePlus = jimport('ij.ImagePlus')
        if imp.getType() == ImagePlus.COLOR_RGB:
            print("Warning: RGB images do not support LUT setting, skipping.")
            return img

        Color = jimport('java.awt.Color')
        color_map = {
            "Red": Color.RED,
            "Green": Color.GREEN,
            "Blue": Color.BLUE,
            "Cyan": Color.CYAN,
            "Magenta": Color.MAGENTA,
            "Yellow": Color.YELLOW,
            "Orange": Color.ORANGE,
            "Pink": Color.PINK,
            "Gray": Color.GRAY,
            "White": Color.WHITE,
            "Black": Color.BLACK,
        }

        color_key = color_name.capitalize()
        color = color_map.get(color_key)
        if color is None:
            raise ValueError(f"Unsupported color name: {color_name}, available colors: {list(color_map.keys())}")

        LUT = jimport('ij.process.LUT')
        lut = LUT.createLutFromColor(color)
        imp.setLut(lut)
        print(f"Successfully set LUT to {color_name}")
        dataset = self.ij.convert().convert(imp, jimport('net.imagej.Dataset'))
        imp.close()
        return dataset

    # ----------------- Deconvolution Related -----------------
    def _temp_tiff(self, img):
        """Generate temporary TIFF (internal auxiliary method, no upper hierarchical calls, directly used)"""
        self._require_imagej_initialized()
        ImagePlus = jimport('ij.ImagePlus')
        imp = self.ij.convert().convert(img, ImagePlus)

        if imp.getType() == ImagePlus.COLOR_RGB:
            imp = jimport('ij.plugin.ChannelSplitter').split(imp)[0]
        elif imp.getNChannels() > 1:
            imp = jimport('ij.plugin.ChannelSplitter').split(imp)[0]

        if imp.getType() != ImagePlus.GRAY16:
            jimport('ij.process.ImageConverter')(imp).convertToGray16()

        fd, tmp_path = tempfile.mkstemp(suffix='.tif')
        os.close(fd)
        jimport('ij.io.FileSaver')(imp).saveAsTiff(tmp_path)
        # Close ImagePlus object
        imp.close()

        return tmp_path.replace("\\", "/")

    @tool_func
    def richardson_lucy(
        self,
        image_meta: ImageWithMetadata,
        magnification: int,
        iterations: int = 50,
        out_filename: str = "deconvolved_result",
        out_dir: Optional[str] = None
    ) -> ImageWithMetadata:
        """
        Synchronously perform Richardson-Lucy deconvolution (receive and return ImageWithMetadata)
        
        Parameters:
            image_meta: Input image and metadata
            magnification: Objective lens magnification, supporting 40, 60, 100
            iterations: Number of deconvolution iterations
            out_filename: Output file name (without path)
            out_dir: Output directory, use self.output_directory if None
        """
        # Select PSF path based on magnification
        psf_mapping = {
            40: PSF_40X,
            60: PSF_60X,
            100: PSF_100X,
        }

        if magnification not in psf_mapping:
            raise ValueError(
                f"Unsupported magnification: {magnification}. Supported values: {list(psf_mapping.keys())}. "
                "If the metadata contains an objective label such as '4-LUCPLFLN40X', use 40; "
                "the leading '4-' is the objective turret position, not the magnification."
            )

        psf_path = _resolve_project_path(psf_mapping[magnification])
        if not os.path.exists(psf_path):
            raise FileNotFoundError(f"PSF file does not exist: {psf_path} (corresponding to {magnification}x objective)")

        if out_dir is None:
            out_dir = self.output_directory

        print(f"Running {magnification}x deconvolution (PSF: {psf_path})...")
        decon_dataset = self._richardson_lucy_impl(
            image_meta.dataset, psf_path, iterations, out_filename, out_dir
        )
        return ImageWithMetadata(
            dataset=decon_dataset,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            center_z_um=image_meta.center_z_um,
            pixel_size_x_um=image_meta.pixel_size_x_um,
            pixel_size_y_um=image_meta.pixel_size_y_um,
        )

    def _richardson_lucy_impl(
        self,
        img,
        psf_path: str,
        iterations: int = 50,
        out_filename: str = "",
        out_dir: str = ""
    ):
        """Internal implementation: only process image and return Dataset"""
        self._require_imagej_initialized()
        os.makedirs(out_dir, exist_ok=True)
        tmp_img_path = self._temp_tiff(img)

        psf_path = psf_path.replace("\\", "/")
        out_dir = out_dir.replace("\\", "/")

        macro = f"""
            image = "-image file {tmp_img_path}";
            psf = "-psf file {psf_path}";
            alg = "-algorithm RL {iterations} -out mip {out_filename} -path {out_dir}";
            run("DeconvolutionLab2 Run", image + " " + psf + " " + alg);
        """

        self.ij.py.run_macro(macro)

        out_file = f"{out_filename}.tif"
        mip_path = os.path.join(out_dir, out_file)
        timeout = 60
        start_time = time.time()

        while not os.path.isfile(mip_path):
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                if os.path.exists(tmp_img_path):
                    os.remove(tmp_img_path)
                raise TimeoutError(f"Deconvolution timed out ({timeout}s), file not generated: {mip_path}")
            time.sleep(1)

        if os.path.exists(tmp_img_path):
            os.remove(tmp_img_path)

        self._close_all_imagej_images()

        if not os.path.isfile(mip_path):
            raise FileNotFoundError("DL2 deconvolution failed, file not generated: " + mip_path)

        decon_ds = self.ij.io().open(mip_path)
        return decon_ds

    # ----------------- Denoising -----------------

    @tool_func
    def denoise(self, image_meta: ImageWithMetadata, method="Gaussian", radius=2.0) -> ImageWithMetadata:
        """Synchronously perform denoising and return image with metadata"""
        print(f"Performing {method} denoising, radius={radius}")
        new_dataset = self._denoise_impl(image_meta.dataset, method, radius)
        return ImageWithMetadata(
            dataset=new_dataset,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            center_z_um=image_meta.center_z_um,
            pixel_size_x_um=image_meta.pixel_size_x_um,
            pixel_size_y_um=image_meta.pixel_size_y_um,
        )

    def _denoise_impl(self, img, method="Gaussian", radius=2.0):
        if hasattr(img, 'getImgPlus'):
            imp = self.dataset_to_imp(img)
        else:
            imp = img

        self._require_imagej_initialized()
        valid_methods = ["Gaussian", "Median", "NLM"]
        if method not in valid_methods:
            raise ValueError(f"method must be one of {', '.join(valid_methods)}")

        if method == "Gaussian":
            GaussianBlur3D = jimport('ij.plugin.GaussianBlur3D')
            GaussianBlur3D.blur(imp, radius, radius, radius)
        elif method == "Median":
            ImagePlus = jimport('ij.ImagePlus')
            if imp.getNSlices() == 1:
                RankFilters = jimport('ij.plugin.filter.RankFilters')
                RankFilters().rank(imp.getProcessor(), radius, RankFilters.MEDIAN)
            else:
                Median3D = jimport('ij.plugin.filter.Median3D')
                Median3D.filter(imp, int(radius), int(radius), int(radius))
        elif method == "NLM":
            NLM = jimport('nlmdenoise.NLMDenoise_')
            NLM().run(imp, radius, 0.5, 0)

        dataset = self.ij.convert().convert(imp, jimport('net.imagej.Dataset'))
        imp.close()
        return dataset

    # ----------------- Z-projection -----------------
    @tool_func
    def z_projection(self, image_meta: ImageWithMetadata, method="max") -> ImageWithMetadata:
        """Synchronously perform Z-projection and return 2D image + metadata (z set to 0)"""
        new_dataset = self._z_projection_impl(image_meta.dataset, method)
        return ImageWithMetadata(
            dataset=new_dataset,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            center_z_um=0.0,  # No Z after projection
            pixel_size_x_um=image_meta.pixel_size_x_um,
            pixel_size_y_um=image_meta.pixel_size_y_um,
        )

    def _z_projection_impl(self, img, method="max"):
        if hasattr(img, 'getImgPlus'):
            imp = self.dataset_to_imp(img)
        else:
            imp = img

        self._require_imagej_initialized()
        projection_methods = {
            "max": "Max Intensity",
            "avg": "Average Intensity",
            "sum": "Sum Slices"
        }
        method_lower = method.lower()
        if method_lower not in projection_methods:
            raise ValueError(f"Unsupported projection method: {method}")

        projection_type = projection_methods[method_lower]
        self.ij.IJ.run(imp, "Z Project...", f"projection=[{projection_type}] all")
        projected_imp = self.ij.py.active_imageplus()
        dataset = self.ij.convert().convert(projected_imp, jimport('net.imagej.Dataset'))
        imp.close()
        projected_imp.close()
        return dataset

    # ----------------- Auxiliary Analysis Methods -----------------
    @tool_func
    def quantify_fluorescence(self, image_meta: ImageWithMetadata) -> float:
        """
        Fluorescence quantification: Calculate the average value of image pixel intensity (actual logic)
        Input must be ImageWithMetadata, and its dataset is extracted internally for processing.
        """
        if self.ij is None:
            raise RuntimeError("ImageJ not initialized, please call fiji_initialize() first")

        try:
            # Extract dataset from ImageWithMetadata
            dataset = image_meta.dataset

            # Convert to ImagePlus to get pixels
            imp = self.dataset_to_imp(dataset)
            try:
                pixels = self.ij.py.from_java(imp.getProcessor().getPixels())
                img_array = np.asarray(pixels, dtype=np.float32)
                
                # Handle dimensions: ensure it is 2D or 3D (multi-channel), then take the overall average
                # Note: ImagePlus is usually 2D single-channel, but just in case
                intensity = float(np.mean(img_array))
                print(f"Fluorescence signal intensity: {intensity:.2f}")
                return intensity
            finally:
                imp.close()  # Ensure ImagePlus is released

        except Exception as e:
            print(f"Fluorescence quantification failed: {e}")
            return 0.0  # Or return np.nan as needed
    # ----------------- Resource Release -----------------
    def _close_all_imagej_images(self):
        """Close ImageJ image windows without triggering save-confirmation dialogs."""
        self._require_imagej_initialized()
        WindowManager = jimport('ij.WindowManager')

        image_ids = WindowManager.getIDList()
        if image_ids is None:
            return

        for image_id in list(image_ids):
            imp = WindowManager.getImage(int(image_id))
            if imp is None:
                continue
            try:
                imp.changes = False
            except Exception:
                pass
            try:
                imp.close()
            except Exception:
                logger.debug("Failed to close ImageJ image %s", image_id, exc_info=True)

    @tool_func
    def fiji_shutdown(self, shutdown_jvm: bool = False):
        """Synchronously shut down ImageJ (no hierarchical calls, direct implementation of core logic)"""
        # Release organoid detection model
        if self._organoid_model is not None:
            del self._organoid_model
            self._organoid_model = None
            _safe_empty_cuda_cache()

        if hasattr(self, "_generic_model_cache"):
            self._generic_model_cache.clear()
            _safe_empty_cuda_cache()

        if self.ij:
            self._close_all_imagej_images()
            self.ij.dispose()
            self.ij = None
            if shutdown_jvm:
                sj.shutdown_jvm()
                print("ImageJ has released resources and JVM has been terminated")
            else:
                print("ImageJ has released resources; JVM shutdown was skipped to avoid blocking")

    def _init_generic_model(self, config, checkpoint, device):
        if not hasattr(self, '_generic_model_cache'):
            self._generic_model_cache = {}
        key = (config, checkpoint, device)
        if key not in self._generic_model_cache:
            init_detector, _ = _get_mmdet_apis()
            self._generic_model_cache[key] = init_detector(config, checkpoint, device=device)
        return self._generic_model_cache[key]

    def _safe_image_normalize(self, img: np.ndarray) -> np.ndarray:
        """Safely normalize image to uint8 [0,255] range for MMDetection input"""
        if img.dtype == np.uint8:
            return img
        img = img.astype(np.float32)
        img -= img.min()
        if img.max() > 0:
            img = img / img.max() * 255.0
        return img.astype(np.uint8)

    def _analysis_platform_find_position(
        self,
        image,
        description: str,
        center_x_um: float,
        center_y_um: float,
        pixel_size_um: float,
        pixel_size_x_um: Optional[float] = None,
        pixel_size_y_um: Optional[float] = None,
        model_config: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        score_thr: float = 0.3,
        nms_thr: float = 0.5,
        target_size: int = 512,
        target_class_id: Optional[int] = 0,
        target_class_name: Optional[str] = None,
        output_filename: str = 'target_locations_list.json'
    ) -> List[Tuple[float, float, float, float]]:
        """
        Generic target position detection function
        
        Returns: List[(center_x_px, center_y_px, width_px, height_px)] —— Pixel coordinates
        Saves: Physical coordinates in JSON file List[(cx_um, cy_um, w_um, h_um)]
        """

        device = device or _default_inference_device()
        pixel_size_x_um = float(pixel_size_um if pixel_size_x_um is None else pixel_size_x_um)
        pixel_size_y_um = float(pixel_size_um if pixel_size_y_um is None else pixel_size_y_um)
        cv2 = _get_cv2()
        _, inference_detector = _get_mmdet_apis()

        # ===================== Image Type Conversion (only extract first channel for multi-channel) =====================
        img_np = None
        if isinstance(image, np.ndarray):
            img_np = image.copy()
            if len(img_np.shape) == 3 and img_np.shape[2] > 1:
                print(f"Multi-channel np.ndarray image detected (channels: {img_np.shape[2]}), only using first channel")
                img_np = img_np[:, :, 0]
        else:
            try:
                if self.ij is None:
                    raise RuntimeError("ImageJ environment not initialized, please call fiji_initialize() first")

                owns_imp = False
                if hasattr(image, 'getImgPlus'):
                    imp = self.dataset_to_imp(image)
                    owns_imp = True
                else:
                    imp = image

                height = int(imp.getHeight())
                width = int(imp.getWidth())
                channels = int(imp.getNChannels())
                slices = int(imp.getNSlices())
                frames = int(imp.getNFrames())
                print(
                    f"ImagePlus image information: height {height}, width {width}, "
                    f"channels {channels}, slices {slices}, frames {frames}"
                )

                if channels > 1 or slices > 1 or frames > 1:
                    print(
                        "Multi-dimensional ImagePlus detected; using the current processor "
                        "plane (first channel/slice/frame by default)."
                    )

                try:
                    img_np = self._imageplus_to_numpy_2d(imp)
                finally:
                    if owns_imp:
                        imp.close()

                if len(img_np.shape) != 2:
                    print(f"Abnormal dimension after image conversion: {len(img_np.shape)}, only 2D single-channel images are supported")
                    return []
            except Exception as e:
                print(f"Image type conversion failed! Unsupported input type: {type(image)}, error message: {e}")
                return []
        # ==================================================================================

        model_config = model_config
        model_checkpoint = model_checkpoint
        regions = []

        def letterbox_image(img, target_size):
            h, w = img.shape[:2]
            scale = min(target_size / w, target_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            dx = (target_size - new_w) // 2
            dy = (target_size - new_h) // 2
            padded_img = cv2.copyMakeBorder(
                resized_img, dy, target_size - new_h - dy,
                dx, target_size - new_w - dx,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            return padded_img, (scale, dx, dy)

        def nms_with_indices(bboxes, scores, iou_threshold):
            if len(bboxes) == 0:
                return []
            bboxes_np = _boxes_xyxy_to_xywh(bboxes)
            scores_np = np.array(scores, dtype=np.float32)
            indices = cv2.dnn.NMSBoxes(
                bboxes=bboxes_np.tolist(),
                scores=scores_np.tolist(),
                score_threshold=0.0,
                nms_threshold=iou_threshold
            )
            return _normalize_nms_indices(indices)

        try:
            model = self._init_generic_model(model_config, model_checkpoint, device)
            if model is None:
                return []

            orig_img = img_np.copy()
            if len(orig_img.shape) == 2:
                try:
                    input_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
                except:
                    input_img = np.stack([orig_img] * 3, axis=-1)
            elif len(orig_img.shape) == 3 and orig_img.shape[2] == 1:
                input_img = cv2.cvtColor(orig_img.squeeze(), cv2.COLOR_GRAY2RGB)
            else:
                input_img = orig_img.copy()

            input_img = self._safe_image_normalize(input_img)
            orig_h, orig_w = input_img.shape[:2]

            resized_img, (scale, dx, dy) = letterbox_image(input_img, target_size)

            result = inference_detector(model, resized_img)
            print("Generic target detection inference completed")

            bboxes, scores, labels = _extract_filtered_pred_instances(result, score_thr)
            if len(bboxes) == 0:
                print("No valid targets detected (after confidence filtering)")
                return []

            resolved_class_id = _select_target_class_index(
                model,
                target_class_name=target_class_name,
                target_class_id=target_class_id,
            )
            target_mask = labels == resolved_class_id
            bboxes = bboxes[target_mask]
            scores = scores[target_mask]
            if len(bboxes) == 0:
                print(f"No targets detected for class {target_class_name or resolved_class_id}")
                return []

            keep_indices = nms_with_indices(bboxes, scores, nms_thr)
            bboxes = bboxes[keep_indices]
            if len(bboxes) == 0:
                print("No valid targets detected (after NMS deduplication)")
                return []

            bboxes[:, [0, 2]] -= dx
            bboxes[:, [1, 3]] -= dy
            bboxes /= scale

            orig_w_int = int(orig_w)
            orig_h_int = int(orig_h)
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, orig_w_int)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, orig_h_int)

            # === Calculate pixel coordinates (return) and physical coordinates (save) simultaneously ===
            img_h, img_w = img_np.shape[:2]
            image_center_x_px = (img_w - 1) / 2.0
            image_center_y_px = (img_h - 1) / 2.0
            pixel_regions = []
            physical_regions = []

            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                w_px = x2 - x1
                h_px = y2 - y1
                if w_px <= 0 or h_px <= 0:
                    continue

                cx_px = (x1 + x2) / 2.0
                cy_px = (y1 + y2) / 2.0
                pixel_regions.append((cx_px, cy_px, w_px, h_px))

                # Physical coordinates (for saving)
                dx_img = cx_px - image_center_x_px
                dy_img = cy_px - image_center_y_px
                cx_um = center_x_um + dx_img * pixel_size_x_um
                cy_um = center_y_um - dy_img * pixel_size_y_um  # Y-axis flip
                w_um = w_px * pixel_size_x_um
                h_um = h_px * pixel_size_y_um
                physical_regions.append([cx_um, cy_um, w_um, h_um])

            print(f"Total {len(pixel_regions)} valid targets detected")

        except Exception as e:
            print(f"Generic target detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            if device.startswith('cuda'):
                _safe_empty_cuda_cache()

        # === Save annotated image (with red boxes and numbering) ===
        if len(bboxes) > 0:
            # Use original grayscale image for display
            if len(orig_img.shape) == 2:
                display_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
            else:
                display_img = orig_img.copy()

            for idx, (x1, y1, x2, y2) in enumerate(bboxes, start=1):
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
                label = str(idx)
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = x1
                text_y = y1 - 5 if y1 > 10 else y1 + text_size[1] + 5
                # Black outline
                cv2.putText(display_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
                # White text
                cv2.putText(display_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)

            # Save to self.output_directory without registration
            img_output_filename = output_filename.replace('.json', '_annotated.jpg')
            img_output_path = os.path.join(self.output_directory, img_output_filename)
            os.makedirs(os.path.dirname(img_output_path), exist_ok=True)
            saved = bool(cv2.imwrite(img_output_path, display_img))
            if saved:
                self._emit_interaction_artifact(
                    path=img_output_path,
                    title="Fiji Detection Result",
                    text="Annotated image is ready for review.",
                )
            else:
                logger.warning("Failed to save Fiji annotated image to %s", img_output_path)

        # === Save: use physical coordinates ===
        if len(physical_regions) > 0:
            output_path = os.path.join(self.output_directory, output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(physical_regions, f, indent=2)
            self._storagemanger.register_file(output_filename, description, 'analysis_platform', 'json')
        else:
            print(f"No valid targets, skipping file {output_filename} registration")

        # === Return: pixel coordinates ===
        return pixel_regions

    @tool_func
    def analysis_platform_find_target_positions(
        self,
        image_meta: ImageWithMetadata,
        target_type: str,
        description: str,
        score_thr: float = 0.3,
        nms_thr: float = 0.5,
        target_size: int = 512,
        target_class_id: Optional[int] = None,
        target_class_name: Optional[str] = None,
        output_filename: Optional[str] = None,
        model_config: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        tile_size: int = DEFAULT_DETECTION_TILE_SIZE,
        tile_overlap: int = DEFAULT_DETECTION_TILE_OVERLAP,
        tile_edge_margin: float = DEFAULT_TILE_EDGE_MARGIN,
        global_iou_thr: float = DEFAULT_GLOBAL_TILE_IOU_THRESHOLD,
        max_tiles: int = 0,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Primary target-detection entry point.

        In normal usage, switch detection models by changing `target_type`.
        The model config/checkpoint/class/output naming are resolved from
        runtime config `detection_targets` unless explicitly overridden by arguments.
        """
        print(f"Finding {target_type} target positions in image")

        spec = _resolve_target_detection_spec(target_type)
        final_output_filename = output_filename or spec["output_filename"]
        np_xarray = self.ij.py.to_xarray(image_meta.dataset)
        data = np_xarray.data
        dims = list(getattr(np_xarray, "dims", ()))

        if not dims:
            image_np = np.asarray(data)
            if image_np.ndim != 2:
                raise ValueError(f"Dataset has no named dims and cannot be reduced to 2D safely: shape={image_np.shape}")
        else:
            row_dim = next((name for name in ("row", "y") if name in dims), None)
            col_dim = next((name for name in ("col", "x") if name in dims), None)
            if row_dim is None or col_dim is None:
                raise ValueError(f"Dataset does not expose 2D spatial axes as row/col or y/x: dims={dims}")

            selection = []
            reduced_axes: list[str] = []
            for dim_name, dim_size in zip(dims, data.shape):
                if dim_name in {row_dim, col_dim}:
                    selection.append(slice(None))
                else:
                    if int(dim_size) > 1:
                        reduced_axes.append(f"{dim_name}={int(dim_size)}->0")
                    selection.append(0)

            if reduced_axes:
                print(
                    "Image is not 2D single-channel. Extracting the first plane along non-spatial axes: "
                    + ", ".join(reduced_axes)
                )

            image_np = np.asarray(data[tuple(selection)])
            if image_np.ndim != 2:
                raise ValueError(
                    f"Expected a 2D array after plane selection, got shape={image_np.shape}, dims={dims}, "
                    f"spatial_axes=({row_dim}, {col_dim})"
                )

            selected_dims = [dim_name for dim_name, selector in zip(dims, selection) if isinstance(selector, slice)]
            if selected_dims != [row_dim, col_dim]:
                target_axes = [selected_dims.index(row_dim), selected_dims.index(col_dim)]
                image_np = np.transpose(image_np, axes=target_axes)

        height, width = image_np.shape[:2]
        pixel_size_x_um = float(image_meta.pixel_size_x_um)
        pixel_size_y_um = float(image_meta.pixel_size_y_um)
        image_center_x_um = float(image_meta.center_x_um)
        image_center_y_um = float(image_meta.center_y_um)
        image_center_x_px = (width - 1) / 2.0
        image_center_y_px = (height - 1) / 2.0

        if tile_size <= 0:
            tile_size = max(height, width)
        tiles = list(_iter_detection_tiles((height, width), int(tile_size), int(tile_overlap)))
        if max_tiles > 0:
            tiles = tiles[: int(max_tiles)]

        aggregated_pixel_regions: List[Tuple[float, float, float, float]] = []
        tile_output_root = os.path.join(
            os.path.dirname(final_output_filename),
            "_tiles",
            Path(final_output_filename).stem,
        ).replace("\\", "/")

        for tile_index, (y0, y1, x0, x1) in enumerate(tiles, start=1):
            tile = np.asarray(image_np[y0:y1, x0:x1])
            tile_height, tile_width = tile.shape[:2]
            tile_center_x_px = (x0 + x1 - 1) / 2.0
            tile_center_y_px = (y0 + y1 - 1) / 2.0
            tile_center_x_um = image_center_x_um + (tile_center_x_px - image_center_x_px) * pixel_size_x_um
            tile_center_y_um = image_center_y_um - (tile_center_y_px - image_center_y_px) * pixel_size_y_um
            tile_output_filename = f"{tile_output_root}/tile_{tile_index:03d}.json"
            tile_regions = self._analysis_platform_find_position(
                image=tile,
                description=f"{description} (tile {tile_index})",
                center_x_um=tile_center_x_um,
                center_y_um=tile_center_y_um,
                pixel_size_um=pixel_size_x_um,
                pixel_size_x_um=pixel_size_x_um,
                pixel_size_y_um=pixel_size_y_um,
                model_config=model_config or spec["model_config"],
                model_checkpoint=model_checkpoint or spec["model_checkpoint"],
                device=device,
                score_thr=score_thr,
                nms_thr=nms_thr,
                target_size=target_size,
                target_class_id=target_class_id if target_class_id is not None else spec["target_class_id"],
                target_class_name=target_class_name or spec["target_class_name"],
                output_filename=tile_output_filename,
            )
            for center_x_px, center_y_px, width_px, height_px in tile_regions:
                if _touches_internal_tile_edge(
                    center_x_px=float(center_x_px),
                    center_y_px=float(center_y_px),
                    width_px=float(width_px),
                    height_px=float(height_px),
                    tile_width=int(tile_width),
                    tile_height=int(tile_height),
                    tile_x0=int(x0),
                    tile_y0=int(y0),
                    image_width=int(width),
                    image_height=int(height),
                    edge_margin=float(tile_edge_margin),
                ):
                    continue
                aggregated_pixel_regions.append(
                    (
                        float(center_x_px) + float(x0),
                        float(center_y_px) + float(y0),
                        float(width_px),
                        float(height_px),
                    )
                )

        if aggregated_pixel_regions:
            aggregated_pixel_regions = _deduplicate_pixel_regions(
                aggregated_pixel_regions,
                iou_threshold=float(global_iou_thr),
            )

        physical_regions = []
        for cx_px, cy_px, w_px, h_px in aggregated_pixel_regions:
            dx_img = float(cx_px) - image_center_x_px
            dy_img = float(cy_px) - image_center_y_px
            physical_regions.append(
                [
                    image_center_x_um + dx_img * pixel_size_x_um,
                    image_center_y_um - dy_img * pixel_size_y_um,
                    float(w_px) * pixel_size_x_um,
                    float(h_px) * pixel_size_y_um,
                ]
            )

        output_path = os.path.join(self.output_directory, final_output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(physical_regions, f, indent=2)
        self._storagemanger.register_file(final_output_filename, description, 'analysis_platform', 'json')

        cv2 = _get_cv2()
        display_img = self._safe_image_normalize(np.asarray(image_np))
        if display_img.ndim == 2:
            display_img = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
        for idx, (cx_px, cy_px, w_px, h_px) in enumerate(aggregated_pixel_regions, start=1):
            x1 = int(round(float(cx_px) - float(w_px) / 2.0))
            y1 = int(round(float(cy_px) - float(h_px) / 2.0))
            x2 = int(round(float(cx_px) + float(w_px) / 2.0))
            y2 = int(round(float(cy_px) + float(h_px) / 2.0))
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)
            label = str(idx)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            text_x = x1
            text_y = y1 - 5 if y1 > 10 else y1 + text_size[1] + 5
            cv2.putText(display_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
            cv2.putText(display_img, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)

        img_output_filename = final_output_filename.replace('.json', '_annotated.jpg')
        img_output_path = os.path.join(self.output_directory, img_output_filename)
        os.makedirs(os.path.dirname(img_output_path), exist_ok=True)
        saved = bool(cv2.imwrite(img_output_path, display_img))
        if saved:
            self._emit_interaction_artifact(
                path=img_output_path,
                title="Fiji Detection Result",
                text="Annotated image is ready for review.",
            )
        else:
            logger.warning("Failed to save Fiji annotated image to %s", img_output_path)
        return aggregated_pixel_regions

    def list_supported_detection_targets(self) -> list[str]:
        """Return the registered target names accepted by analysis_platform_find_target_positions()."""
        return _list_supported_target_types()
    @tool_func
    def convert_to_numpy(self, image_meta: ImageWithMetadata) -> np.ndarray:
        """
        Convert the image inside an ImageWithMetadata object to a numpy array for processing.

        Args:
            image_meta (ImageWithMetadata): Input image container with metadata

        Returns:
            np.ndarray: Single-channel grayscale numpy array with shape (height, width)
                with the original pixel dtype preserved when possible.
        """
        if self.ij is None:
            raise RuntimeError("ImageJ environment not initialized. Call fiji_initialize() first.")

        dataset = image_meta.dataset
        np_xarray = self.ij.py.to_xarray(dataset)
        data = np_xarray.data
        dims = list(getattr(np_xarray, "dims", ()))

        if not dims:
            img_array = np.asarray(data)
            if img_array.ndim != 2:
                raise ValueError(f"Dataset has no named dims and cannot be reduced to 2D safely: shape={img_array.shape}")
            return img_array

        row_dim = next((name for name in ("row", "y") if name in dims), None)
        col_dim = next((name for name in ("col", "x") if name in dims), None)
        if row_dim is None or col_dim is None:
            raise ValueError(f"Dataset does not expose 2D spatial axes as row/col or y/x: dims={dims}")

        selection = []
        reduced_axes: list[str] = []
        for dim_name, dim_size in zip(dims, data.shape):
            if dim_name in {row_dim, col_dim}:
                selection.append(slice(None))
            else:
                if int(dim_size) > 1:
                    reduced_axes.append(f"{dim_name}={int(dim_size)}->0")
                selection.append(0)

        if reduced_axes:
            print(
                "Image is not 2D single-channel. Extracting the first plane along non-spatial axes: "
                + ", ".join(reduced_axes)
            )

        img_array = np.asarray(data[tuple(selection)])
        if img_array.ndim != 2:
            raise ValueError(
                f"Expected a 2D array after plane selection, got shape={img_array.shape}, dims={dims}, "
                f"spatial_axes=({row_dim}, {col_dim})"
            )

        row_axis = dims.index(row_dim)
        col_axis = dims.index(col_dim)
        selected_dims = [dim_name for dim_name, selector in zip(dims, selection) if isinstance(selector, slice)]
        if selected_dims != [row_dim, col_dim]:
            target_axes = [selected_dims.index(row_dim), selected_dims.index(col_dim)]
            img_array = np.transpose(img_array, axes=target_axes)
        return np.asarray(img_array)







