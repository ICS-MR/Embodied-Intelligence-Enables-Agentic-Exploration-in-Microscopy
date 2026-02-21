import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Sequence, Literal
from skimage import measure, restoration, io
from skimage.restoration import estimate_sigma, denoise_nl_means
import tifffile
from cellpose import models, core, io as cellpose_io

from tool.base import BaseTool, tool_func
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

    @tool_func
    def cellpose_initialize(
            self,
            gpu: bool = False,
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
            Model type (e.g., "cyto3", "nuclei", "livecell", etc.)
        device : str | None
            Specify computing device (e.g., "cpu", "cuda", "cuda:0")
        **kwargs :
            Other parameters passed to models.CellposeModel
        """
        # Cellpose 4.0 uses CellposeModel class
        self._use_GPU = gpu and core.use_gpu() if device is None else (device.startswith("cuda"))
        if model_type is None:
            # Use cyto model by default
            self.model = models.CellposeModel(gpu=gpu)
        else:
            self.model = models.CellposeModel(gpu=gpu, pretrained_model=model_type)
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
        img_path = os.path.join(self.output_directory, file_path)
        
        try:
            with tifffile.TiffFile(img_path) as tif:
                image = tif.asarray()
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
        # 1. 校验前置条件
        if self.model is None:
            raise ValueError("Cellpose model not initialized! Call cellpose_initialize first.")
        if not isinstance(image, np.ndarray):
            raise ValueError(f"segment only accepts numpy.ndarray, got {type(image)}")
        if image.ndim not in [2, 3]:
            raise ValueError(f"Image must be 2D (H,W) or 3D (H,W,C), got {image.ndim}D")

        # 2. 预处理：可选去噪
        img = image.astype(np.float32)
        if denoise and img.ndim == 2:
            sigma_est = estimate_sigma(img, channel_axis=None)
            img = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True, channel_axis=None)

        # 3. 执行分割（Cellpose 4.0 return format: masks, flows, styles）
        masks, flows, styles = self.model.eval(
            img,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,** kwargs
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
            return output_path
        except Exception as e:
            raise IOError(f"Failed to save CSV file: {e}")

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
        if image.dtype != np.uint8:
            img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
            img_norm = (img_norm * 255).astype(np.uint8)

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
