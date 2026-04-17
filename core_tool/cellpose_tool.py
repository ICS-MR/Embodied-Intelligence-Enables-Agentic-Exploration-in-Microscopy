import json
import logging
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence, Literal
from skimage import measure, restoration, io
from skimage.restoration import estimate_sigma, denoise_nl_means
import tifffile
try:
    import cv2
except Exception:
    cv2 = None
import cellpose.core as cellpose_core
import cellpose.models as cellpose_models

from core_tool.spatial_metadata import load_ome_spatial_metadata
from tool.base import BaseTool, tool_func

logger = logging.getLogger(__name__)

_DIGIT_BITMAPS: dict[str, list[str]] = {
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "001", "001", "001"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
}

_DEFAULT_SEGMENT_EVAL_KWARGS: dict[str, object] = {
    # Align runtime defaults with the GUI CPSAM workflow so different entrypoints
    # are less likely to drift in detection count.
    "niter": 200,
    "normalize": {"percentile": (1.0, 99.0)},
}
_DEFAULT_TILE_SIZE = 1024
_DEFAULT_TILE_OVERLAP = 128
_DEFAULT_LARGE_IMAGE_MAX_SIDE = 2048
_DEFAULT_LARGE_IMAGE_MAX_PIXELS = 2048 * 2048
_DEFAULT_TILED_NITER = 100


class Cellpose2D(BaseTool):
    """
    Minimal 2D Cellpose Wrapper (Compatible with Cellpose 4.0+).
    """

    def __init__(
            self,
            storagemanger,
            output_path: str
    ):
        self._storagemanger = storagemanger
        self.output_directory: str = output_path
        self.model = None  # Initialize model as None
        self._use_GPU = None  # Record GPU usage status

    def _resolve_input_path(self, file_path: str | Path) -> Path:
        candidate = Path(file_path).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()
        return Path(self.output_directory, candidate).resolve()

    def _load_image_spatial_metadata(self, file_path: str | Path) -> dict[str, float]:
        return load_ome_spatial_metadata(self._resolve_input_path(file_path))

    def _coerce_masks_to_2d(self, masks: np.ndarray) -> np.ndarray:
        masks_array = np.asarray(masks)
        squeezed = np.squeeze(masks_array)
        if squeezed.ndim != 2:
            raise ValueError(f"Mask must be a 2D array after squeezing singleton dimensions, got {squeezed.ndim}D")
        return squeezed

    def _normalize_loaded_image_shape(self, image: np.ndarray) -> np.ndarray:
        image_array = np.asarray(image)
        if image_array.ndim != 6:
            return image_array

        # tifffile may preserve an extra singleton/non-singleton samples axis and return
        # TCZYXS for OME-TIFF inputs when squeeze=False. Convert that into TCZYX so
        # existing generated workflow code that indexes image[0, 0, 0, :, :] keeps working.
        t, c, z, y, x, s = image_array.shape
        if s == 1:
            return image_array[..., 0]
        return image_array.transpose(0, 1, 5, 2, 3, 4).reshape(t, c * s, z, y, x)

    def _prepare_image_for_segmentation(self, image: np.ndarray) -> np.ndarray:
        image_array = np.asarray(image)
        squeezed = np.squeeze(image_array)

        while squeezed.ndim > 3:
            squeezed = squeezed[0]

        if squeezed.ndim == 2:
            return squeezed.astype(np.float32, copy=False)

        if squeezed.ndim != 3:
            raise ValueError(f"Unsupported image shape for 2D Cellpose segmentation: {image_array.shape}")

        # Standard HWC RGB/RGBA image: preserve color information because CPSAM
        # can produce better brightfield detections from natural-color inputs.
        if squeezed.shape[-1] in (3, 4):
            return squeezed[..., :3].astype(np.float32, copy=False)

        # CHW-style microscopy data: keep single-channel inputs grayscale, and only
        # promote true 3-channel data to HWC RGB. Other channel counts default to
        # the first channel to avoid inventing pseudo-RGB composites.
        if squeezed.shape[0] in (1, 2, 3, 4):
            if squeezed.shape[0] == 1:
                return squeezed[0].astype(np.float32, copy=False)
            if squeezed.shape[0] >= 3:
                return np.moveaxis(squeezed[:3], 0, -1).astype(np.float32, copy=False)
            return squeezed[0].astype(np.float32, copy=False)

        return squeezed[..., 0].astype(np.float32, copy=False)

    def _should_tile_segmentation(
        self,
        image: np.ndarray,
        *,
        tile_size: int,
        large_image_max_side: int,
        large_image_max_pixels: int,
    ) -> bool:
        h, w = image.shape[:2]
        if tile_size > 0 and (h > tile_size or w > tile_size):
            return True
        if large_image_max_side > 0 and max(h, w) > large_image_max_side:
            return True
        if large_image_max_pixels > 0 and (h * w) > large_image_max_pixels:
            return True
        return False

    def _maybe_denoise_for_segmentation(self, image: np.ndarray, denoise: bool) -> np.ndarray:
        if not denoise or image.ndim != 2:
            return image
        sigma_est = estimate_sigma(image, channel_axis=None)
        return denoise_nl_means(image, h=1.15 * sigma_est, fast_mode=True, channel_axis=None)

    def _iter_tile_windows(self, image_shape: Sequence[int], tile_size: int, tile_overlap: int):
        h, w = image_shape[:2]
        if tile_size <= 0:
            raise ValueError("tile_size must be a positive integer")
        if tile_overlap < 0:
            raise ValueError("tile_overlap must be non-negative")
        if tile_overlap >= tile_size:
            raise ValueError("tile_overlap must be smaller than tile_size")

        stride = tile_size - tile_overlap
        y_starts = list(range(0, max(1, h - tile_size + 1), stride))
        x_starts = list(range(0, max(1, w - tile_size + 1), stride))
        if not y_starts:
            y_starts = [0]
        if not x_starts:
            x_starts = [0]
        if y_starts[-1] != max(0, h - tile_size):
            y_starts.append(max(0, h - tile_size))
        if x_starts[-1] != max(0, w - tile_size):
            x_starts.append(max(0, w - tile_size))

        for y0 in y_starts:
            y1 = min(h, y0 + tile_size)
            core_y0 = y0 if y0 == 0 else y0 + tile_overlap // 2
            core_y1 = y1 if y1 == h else y1 - tile_overlap // 2
            for x0 in x_starts:
                x1 = min(w, x0 + tile_size)
                core_x0 = x0 if x0 == 0 else x0 + tile_overlap // 2
                core_x1 = x1 if x1 == w else x1 - tile_overlap // 2
                if core_y1 <= core_y0 or core_x1 <= core_x0:
                    continue
                yield {
                    "tile": (slice(y0, y1), slice(x0, x1)),
                    "core": (slice(core_y0, core_y1), slice(core_x0, core_x1)),
                    "local_core": (slice(core_y0 - y0, core_y1 - y0), slice(core_x0 - x0, core_x1 - x0)),
                }

    def _relabel_mask_block(self, mask_block: np.ndarray, next_label: int) -> tuple[np.ndarray, int]:
        block = np.asarray(mask_block)
        relabeled = np.zeros(block.shape, dtype=np.int32)
        unique_labels = np.unique(block)
        unique_labels = unique_labels[unique_labels > 0]
        if unique_labels.size == 0:
            return relabeled, next_label

        for label in unique_labels.tolist():
            relabeled[block == label] = next_label
            next_label += 1
        return relabeled, next_label

    def _segment_large_image_by_tiles(
        self,
        image: np.ndarray,
        *,
        diameter: float | None,
        flow_threshold: float,
        cellprob_threshold: float,
        min_size: int,
        denoise: bool,
        eval_kwargs: dict[str, object],
        tile_size: int,
        tile_overlap: int,
    ) -> np.ndarray:
        h, w = image.shape[:2]
        merged_masks = np.zeros((h, w), dtype=np.int32)
        next_label = 1

        for tile_window in self._iter_tile_windows(image.shape, tile_size, tile_overlap):
            tile_y, tile_x = tile_window["tile"]
            core_y, core_x = tile_window["core"]
            local_core_y, local_core_x = tile_window["local_core"]

            if image.ndim == 2:
                tile_image = image[tile_y, tile_x]
            else:
                tile_image = image[tile_y, tile_x, ...]

            tile_image = self._maybe_denoise_for_segmentation(tile_image, denoise)
            tile_masks, _flows, _styles = self.model.eval(
                tile_image,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size,
                **eval_kwargs,
            )

            tile_masks_2d = self._coerce_masks_to_2d(tile_masks)
            core_block = tile_masks_2d[local_core_y, local_core_x]
            relabeled_block, next_label = self._relabel_mask_block(core_block, next_label)
            merged_masks[core_y, core_x] = relabeled_block

        return merged_masks

    def _normalize_preview_image(self, image: np.ndarray) -> np.ndarray:
        image_array = np.asarray(image)
        squeezed = np.squeeze(image_array)

        if squeezed.ndim == 2:
            img = squeezed.astype(np.float32, copy=False)
            img_min = float(np.min(img)) if img.size else 0.0
            img_max = float(np.max(img)) if img.size else 0.0
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img, dtype=np.float32)
            gray = (img * 255).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=-1)

        if squeezed.ndim >= 3:
            if squeezed.shape[-1] in (3, 4):
                rgb = squeezed[..., :3]
            else:
                rgb = squeezed[0]
                while rgb.ndim > 3:
                    rgb = rgb[0]
                if rgb.ndim == 2:
                    return self._normalize_preview_image(rgb)
                if rgb.ndim == 3 and rgb.shape[-1] not in (3, 4):
                    return self._normalize_preview_image(rgb[0])
                rgb = rgb[..., :3]

            rgb = rgb.astype(np.float32, copy=False)
            img_min = float(np.min(rgb)) if rgb.size else 0.0
            img_max = float(np.max(rgb)) if rgb.size else 0.0
            if img_max > img_min:
                rgb = (rgb - img_min) / (img_max - img_min)
            else:
                rgb = np.zeros_like(rgb, dtype=np.float32)
            return (rgb * 255).astype(np.uint8)

        raise ValueError(f"Unsupported source image shape for annotation preview: {squeezed.shape}")

    def _load_preview_base_image(self, source_image_path: str | Path, masks_2d: np.ndarray) -> np.ndarray:
        resolved = self._resolve_input_path(source_image_path)
        loaders = (
            lambda path: tifffile.imread(str(path)),
            lambda path: io.imread(str(path)),
        )
        for loader in loaders:
            try:
                image = loader(resolved)
                preview = self._normalize_preview_image(image)
                if preview.shape[:2] == masks_2d.shape:
                    return preview
            except Exception:
                continue

        fallback = masks_2d.astype(np.float32, copy=False)
        if fallback.size and float(np.max(fallback)) > 0:
            fallback = fallback / float(np.max(fallback))
        gray = (fallback * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    def _draw_red_box(self, image: np.ndarray, min_row: int, min_col: int, max_row: int, max_col: int, thickness: int = 2) -> None:
        h, w = image.shape[:2]
        min_row = max(0, min(int(min_row), h - 1))
        max_row = max(0, min(int(max_row), h))
        min_col = max(0, min(int(min_col), w - 1))
        max_col = max(0, min(int(max_col), w))
        if max_row <= min_row or max_col <= min_col:
            return

        for offset in range(max(1, int(thickness))):
            top = min(min_row + offset, h - 1)
            bottom = max(min(max_row - 1 - offset, h - 1), 0)
            left = min(min_col + offset, w - 1)
            right = max(min(max_col - 1 - offset, w - 1), 0)
            image[top, left:right + 1] = [255, 0, 0]
            image[bottom, left:right + 1] = [255, 0, 0]
            image[top:bottom + 1, left] = [255, 0, 0]
            image[top:bottom + 1, right] = [255, 0, 0]

    def _draw_region_index(self, image: np.ndarray, index: int, min_row: int, min_col: int) -> None:
        h, w = image.shape[:2]
        text = str(int(index))
        anchor_x = max(0, min(int(min_col) + 4, w - 1))
        anchor_y = max(16, min(int(min_row) + 18, h - 1))

        if cv2 is None or not hasattr(cv2, "putText"):
            self._draw_region_index_fallback(image, text, anchor_y - 12, anchor_x)
            return

        # White outline first so the red index stays readable on brightfield backgrounds.
        cv2.putText(
            image,
            text,
            (anchor_x, anchor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            (anchor_x, anchor_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def _draw_region_index_fallback(self, image: np.ndarray, text: str, top: int, left: int) -> None:
        h, w = image.shape[:2]
        row = max(0, int(top))
        col = max(0, int(left))

        for char in text:
            glyph = _DIGIT_BITMAPS.get(char)
            if glyph is None:
                col += 4
                continue
            glyph_h = len(glyph)
            glyph_w = len(glyph[0])

            # White outline block.
            outline_top = max(0, row - 1)
            outline_left = max(0, col - 1)
            outline_bottom = min(h, row + glyph_h + 1)
            outline_right = min(w, col + glyph_w + 1)
            image[outline_top:outline_bottom, outline_left:outline_right] = [255, 255, 255]

            for r_idx, pattern in enumerate(glyph):
                for c_idx, value in enumerate(pattern):
                    if value != "1":
                        continue
                    rr = row + r_idx
                    cc = col + c_idx
                    if 0 <= rr < h and 0 <= cc < w:
                        image[rr, cc] = [255, 0, 0]
            col += glyph_w + 2

    def _save_target_location_preview(self, source_image_path: str | Path, masks_2d: np.ndarray, regions: list[dict[str, object]], output_json_path: Path) -> Path:
        preview = self._load_preview_base_image(source_image_path, masks_2d).copy()
        for index, region in enumerate(regions, start=1):
            min_row, min_col, max_row, max_col = region["bbox_px"]
            self._draw_red_box(preview, min_row, min_col, max_row, max_col)
            self._draw_region_index(preview, index, min_row, min_col)

        preview_path = output_json_path.with_name(f"{output_json_path.stem}_annotated.png")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        io.imsave(preview_path, preview, check_contrast=False)
        return preview_path

    @tool_func
    def cellpose_initialize(
            self,
            gpu: bool | None = None,
            model_type: str = "cpsam",
            device: str | None = None,
    ):
        """
        Initialize Cellpose 4.0 Model

        Parameters
        ----------
        gpu : bool
            Whether to use GPU (Cellpose 4.0 automatically detects CUDA)
        model_type : str
            Model type or custom model path/name. Defaults to "cpsam" for Cellpose-SAM.
        device : str | None
            Specify computing device (e.g., "cpu", "cuda", "cuda:0")
        **kwargs :
            Other parameters passed to models.CellposeModel
        """
        resolved_device = (device or "").strip().lower()
        if resolved_device:
            use_gpu = resolved_device.startswith("cuda")
        else:
            detected_gpu = bool(cellpose_core.use_gpu())
            use_gpu = detected_gpu if gpu is None else bool(gpu and detected_gpu)

        self._use_GPU = use_gpu
        if model_type is None:
            self.model = cellpose_models.CellposeModel(gpu=use_gpu)
        else:
            self.model = cellpose_models.CellposeModel(gpu=use_gpu, pretrained_model=model_type)
    @tool_func
    def cellpose_read(
        self,
        file_path: str | Path
    ) -> np.ndarray:
        """
        Read an OME-TIFF/regular TIFF file and return the image as a numpy array.
        
        This function loads the entire image data into a numpy array, preserving the 
        original dimension order (e.g., ZCYX, CYX, YX for OME-TIFF). For 2D Cellpose 
        segmentation, you may need to select the appropriate channel/slice afterward.
        
        Args:
            file_path: Path to the OME-TIFF/TIFF file.
        
        Returns:
            image: Image data as a numpy array with original dimension order.
        
        Raises:
            FileNotFoundError: If the file does not exist.
            IOError: If the file cannot be read (e.g., invalid format).
        """
        img_path = self._resolve_input_path(file_path)
        
        try:
            with tifffile.TiffFile(str(img_path)) as tif:
                # Keep singleton axes such as T/C/Z=1 so generated workflow code that
                # indexes TCZYX data remains compatible with real OME-TIFF inputs.
                image = self._normalize_loaded_image_shape(tif.asarray(squeeze=False))
            return image
        
        except Exception as e:
            raise IOError(f"Failed to read image file: {e}")
    @tool_func
    def segment(
            self,
            image: np.ndarray,
            diameter: float | None = None,
            flow_threshold: float = 0.4,
            cellprob_threshold: float = 0.0,
            min_size: int = 15,
            denoise: bool = False,** kwargs
    ) -> np.ndarray:
        """
        2D Instance Segmentation (Compatible with Cellpose 4.0).

        Parameters
        ----------
        image : (H, W) or (H, W, C) ndarray.
        diameter : Cell diameter (pixels), automatically estimated if None.
        flow_threshold : Boundary sensitivity (default 0.4 for Cellpose 4.0).
        cellprob_threshold : Cell probability threshold (filters low-confidence regions).
        min_size : Minimum connected component area.
        denoise : Whether to perform NL-means denoising first (grayscale images only).
        **kwargs : Other parameters passed to model.eval

        Returns
        -------
        masks : (H, W) int32 mask, with 0 as background.

        Raises
        ------
        ValueError: If model not initialized or input is not numpy array.
        """
        # 1. Validate preconditions.
        if self.model is None:
            raise ValueError("Cellpose model not initialized! Call cellpose_initialize first.")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"segment only accepts numpy.ndarray, got {type(image)}")

        tile_size = int(kwargs.pop("tile_size", _DEFAULT_TILE_SIZE) or _DEFAULT_TILE_SIZE)
        tile_overlap = int(kwargs.pop("tile_overlap", _DEFAULT_TILE_OVERLAP) or _DEFAULT_TILE_OVERLAP)
        large_image_max_side = int(
            kwargs.pop("large_image_max_side", _DEFAULT_LARGE_IMAGE_MAX_SIDE) or _DEFAULT_LARGE_IMAGE_MAX_SIDE
        )
        large_image_max_pixels = int(
            kwargs.pop("large_image_max_pixels", _DEFAULT_LARGE_IMAGE_MAX_PIXELS) or _DEFAULT_LARGE_IMAGE_MAX_PIXELS
        )

        # 2. Preprocess: optional denoising.
        img = self._prepare_image_for_segmentation(image)
        eval_kwargs = dict(_DEFAULT_SEGMENT_EVAL_KWARGS)
        eval_kwargs.update(kwargs)
        use_tiling = self._should_tile_segmentation(
            img,
            tile_size=tile_size,
            large_image_max_side=large_image_max_side,
            large_image_max_pixels=large_image_max_pixels,
        )
        if use_tiling:
            if "niter" not in kwargs:
                eval_kwargs["niter"] = min(int(eval_kwargs.get("niter", _DEFAULT_TILED_NITER)), _DEFAULT_TILED_NITER)
            logger.info(
                "Cellpose segmentation is using tiled inference for image shape %s (tile_size=%s, tile_overlap=%s)",
                img.shape,
                tile_size,
                tile_overlap,
            )
            return self._segment_large_image_by_tiles(
                img,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                min_size=min_size,
                denoise=denoise,
                eval_kwargs=eval_kwargs,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
            )

        img = self._maybe_denoise_for_segmentation(img, denoise)

        # 3. Run segmentation (Cellpose 4.0 returns masks, flows, styles).
        masks, flows, styles = self.model.eval(
            img,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            **eval_kwargs,
        )
        return masks
    @tool_func
    def analyze_masks(
            self,
            masks: np.ndarray,
            px_size: float = 1.0,  # Actual length per pixel (μm/px), default 1 pixel
            unit: Literal["px", "μm2"] = "px",
            bins: int | np.ndarray = 20,  # Histogram bins: int for number of bins; array for bin edges
            plot: bool = False,
            **bar_kwargs
    ) -> pd.DataFrame:
        """
        Statistics of cell area-quantity relationship based on 2D masks.

        Parameters
        ----------
        masks : (H, W) int32, with 0 as background
        px_size : Pixel size (μm / px), used for converting to actual area
        unit : Returned area unit, "px" or "μm2"
        bins : Histogram bins
        plot : Whether to plot the histogram
        **bar_kwargs : Parameters passed to plt.bar

        Returns
        -------
        df : DataFrame with three columns
            cell_id : Cell instance ID
            area    : Corresponding area
            bin_idx : Histogram bin index (-1 for background)
        """
        # Calculate area (pixels) for each cell
        props = measure.regionprops(masks)
        areas_px = np.array([p.area for p in props])  # Pixel area
        cell_ids = np.array([p.label for p in props])  # Corresponding label id

        # Unit conversion
        if unit == "μm2":
            areas = areas_px * (px_size ** 2)
        else:
            areas = areas_px

        # Histogram
        hist, bin_edges = np.histogram(areas, bins=bins)
        # Which bin each cell belongs to
        bin_idx = np.digitize(areas, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, len(hist) - 1)

        # Generate DataFrame
        df = pd.DataFrame({
            "cell_id": cell_ids,
            "area": areas,
            "bin_idx": bin_idx,
        })

        # Optional plotting
        if plot:
            plt.figure(figsize=(5, 3))
            plt.bar(
                bin_edges[:-1],
                hist,
                width=np.diff(bin_edges),
                align="edge",** bar_kwargs
            )
            plt.xlabel(f"Area ({unit})")
            plt.ylabel("Number of cells")
            plt.title("Area–number distribution")
            plt.tight_layout()
            plt.show()

        return df

    @tool_func
    def save_masks(self, masks: np.ndarray, filename: str | Path, description: str) -> Path:
        """Save masks (Compatible with Cellpose 4.0 output format)"""
        if masks.ndim != 2:
            raise ValueError(f"Mask must be a 2D array, but got {masks.ndim} dimensions")

        output_path = Path(self.output_directory, filename).expanduser().resolve()

        try:
            # Cellpose 4.0 recommended save format
            tifffile.imwrite(output_path, masks.astype('uint16'), compression='zlib')
            self._storagemanger.register_file(filename, description, 'cellpose', 'tiff')
            return output_path
        except Exception as e:
            raise IOError(f"Failed to save mask: {e}")

    @tool_func
    def save_csv(self, df: pd.DataFrame, filename: str | Path) -> Path:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        output_path = Path(self.output_directory, filename).expanduser().resolve()
        try:
            df.to_csv(output_path, index=False)
            self._storagemanger.register_file(str(Path(filename).name), "Cellpose analysis CSV", 'cellpose', 'csv')
            return output_path
        except Exception as e:
            raise IOError(f"Failed to save CSV file: {e}")

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
        """
        Convert Cellpose masks into microscope-compatible target locations and save them as JSON.

        The output JSON format is a list of `[center_x_um, center_y_um, width_um, height_um]`,
        which is directly compatible with the microscope `load_target_locations()` interface.
        """
        masks_2d = self._coerce_masks_to_2d(masks)
        metadata = self._load_image_spatial_metadata(source_image_path)
        if not bool(metadata.get("stage_positions_present", False)):
            raise ValueError(
                "Source image is missing stage position metadata; cannot export target locations "
                f"for reacquisition from {source_image_path}"
            )
        img_h, img_w = masks_2d.shape
        image_center_x_px = (img_w - 1) / 2.0
        image_center_y_px = (img_h - 1) / 2.0

        regions = []
        for region in measure.regionprops(masks_2d):
            area_px = int(region.area)
            if area_px < int(min_area_px):
                continue
            if max_area_px is not None and area_px > int(max_area_px):
                continue

            cy_px, cx_px = region.centroid
            min_row, min_col, max_row, max_col = region.bbox
            width_px = max_col - min_col
            height_px = max_row - min_row

            # OME stage metadata stores the physical location of the image center.
            # Pixel-center coordinates therefore need to be measured from the center
            # pixel coordinate ((size - 1) / 2), not from size / 2, otherwise target
            # locations drift by half a pixel along each axis.
            dx_px = cx_px - image_center_x_px
            dy_px = cy_px - image_center_y_px
            cx_um = metadata["center_x_um"] + dx_px * metadata["pixel_size_x_um"]
            cy_um = metadata["center_y_um"] + dy_px * metadata["pixel_size_y_um"]
            width_um = width_px * metadata["pixel_size_x_um"]
            height_um = height_px * metadata["pixel_size_y_um"]

            regions.append(
                {
                    "area_px": area_px,
                    "location": [
                        int(round(cx_um)),
                        int(round(cy_um)),
                        int(round(width_um)),
                        int(round(height_um)),
                    ],
                    "bbox_px": [int(min_row), int(min_col), int(max_row), int(max_col)],
                }
            )

        regions.sort(key=lambda item: item["area_px"], reverse=True)
        if top_k is not None:
            regions = regions[: max(0, int(top_k))]

        locations = [item["location"] for item in regions]
        if not locations:
            raise ValueError("No valid target locations were generated from the provided masks")

        output_path = Path(self.output_directory, filename).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(output_path, "w", encoding="utf-8") as handle:
                json.dump(locations, handle, indent=2)
            self._storagemanger.register_file(output_path.name, description, "cellpose", "json")
            try:
                self._save_target_location_preview(source_image_path, masks_2d, regions, output_path)
            except Exception as preview_exc:
                logger.warning("Failed to save Cellpose annotated target preview for %s: %s", output_path, preview_exc)
            return output_path
        except Exception as e:
            raise IOError(f"Failed to save target locations JSON: {e}")

    @tool_func
    def color_masks(self, masks: np.ndarray) -> np.ndarray:
        """
        Generate color-rendered image for masks (colored by area)

        Returns
        -------
        colored_mask_rgb : (H, W, 3) uint8 array
        """
        # Calculate area of each region
        area_dict = {}
        for label in np.unique(masks):
            if label == 0:  # Skip background
                continue
            area_dict[label] = np.sum(masks == label)

        # Sort by area
        sorted_labels = sorted(area_dict.items(), key=lambda x: x[1], reverse=True)
        labels, areas = zip(*sorted_labels) if sorted_labels else ([], [])

        # Create area-based color map
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(labels))) if labels else []

        # Create color mapping table
        color_map = {0: [0, 0, 0, 1]}  # Black for background
        for i, label in enumerate(labels):
            color_map[label] = colors[i]

        # Apply color mapping
        h, w = masks.shape
        colored_mask = np.zeros((h, w, 4), dtype=np.float32)  # RGBA

        for label, color in color_map.items():
            colored_mask[masks == label] = color

        # Convert to RGB (remove Alpha channel)
        colored_mask_rgb = (colored_mask[:, :, :3] * 255).astype(np.uint8)

        # Save colored mask (optional)
        output_path = Path(self.output_directory, "colored_mask.png")
        io.imsave(output_path, colored_mask_rgb)

        return colored_mask_rgb

    @tool_func
    def export_results(self, masks: np.ndarray, base_filename: str, image: np.ndarray | None = None):
        """
        Export complete analysis results (Compatible with Cellpose 4.0 format)

        Parameters
        ----------
        masks : Segmentation mask
        base_filename : Base filename (without extension)
        image : Original image (used for generating overlay)
        """
        # Save original mask
        self.save_masks(masks, f"{base_filename}_masks.tif", "Cellpose segmentation mask")

        # Generate and save colored mask
        colored_mask = self.color_masks(masks)
        color_path = Path(self.output_directory, f"{base_filename}_colored.png")
        io.imsave(color_path, colored_mask)

        # Analyze and save statistical data
        df = self.analyze_masks(masks)
        self.save_csv(df, f"{base_filename}_analysis.csv")

        # Generate overlay if original image is provided
        if image is not None:
            overlay = self._create_overlay(image, masks)
            overlay_path = Path(self.output_directory, f"{base_filename}_overlay.png")
            io.imsave(overlay_path, overlay)

    def _create_overlay(self, image: np.ndarray, masks: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Create overlay of original image and mask"""
        # Normalize image to 0-255
        img_array = np.asarray(image)
        if image.dtype != np.uint8:
            img_norm = (img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8)
            img_norm = (img_norm * 255).astype(np.uint8)
        else:
            img_norm = img_array

        # Convert single channel to RGB if needed
        if img_norm.ndim == 2:
            img_rgb = np.stack([img_norm] * 3, axis=-1)
        elif img_norm.shape[-1] == 1:
            img_rgb = np.repeat(img_norm, 3, axis=-1)
        else:
            img_rgb = img_norm

        # Get colored mask
        colored_mask = self.color_masks(masks)

        # Overlay
        overlay = (img_rgb * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

        return overlay

