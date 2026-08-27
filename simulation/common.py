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
import json

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


def raise_mock_mode_real_runtime_error(*, subsystem: str, mode_field: str, capability: str) -> None:
    raise RuntimeError(
        f"{subsystem} is running in mock mode; switch {mode_field} to real for {capability}."
    )
