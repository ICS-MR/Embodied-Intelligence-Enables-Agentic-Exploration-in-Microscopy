"""Pure demo virtual-environment image transforms (no MMCore dependency).

These helpers simulate the camera-side effects that the Micro-Manager
DemoCamera does not provide natively in "Fluorescent Beads" mode:

- exposure -> intensity mapping (the bead generator ignores exposure), and
- sensor defect injection (dead / saturated pixels).

Keeping them pure and MMCore-free lets them be unit-tested standalone.
"""

from typing import Dict, Optional, Tuple

import numpy as np


def apply_demo_exposure_gain(
    image: np.ndarray,
    *,
    exposure_ms: float,
    reference_ms: float,
    max_gain: float,
) -> np.ndarray:
    """Scale a grayscale demo image by exposure/reference, clamped to max_gain.

    Deterministic linear camera model: low exposure dims the image and high
    exposure saturates at the dtype maximum. Integer dtypes are handled safely.
    """
    arr = np.asarray(image)
    if arr.ndim != 2 or arr.size == 0:
        return arr
    if reference_ms <= 0 or max_gain <= 0:
        return arr.copy()
    gain = max(0.0, min(float(max_gain), float(exposure_ms) / float(reference_ms)))
    if abs(gain - 1.0) < 1e-9:
        return arr.copy()
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scaled = arr.astype(np.float32) * gain
        return np.clip(np.rint(scaled), info.min, info.max).astype(arr.dtype)
    return (arr * gain).astype(arr.dtype, copy=False)


def build_demo_defect_mask(
    shape: Tuple[int, int],
    *,
    fraction: float,
    drop: bool,
    saturate: bool,
    seed: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Build a static sensor-defect index map for a given image shape.

    Returns None when neither drop nor saturate is enabled. Dead pixels are set
    to the dtype minimum (0) and saturated pixels to the dtype maximum. The mask
    is deterministic for a given seed, and drop/saturate sets are disjoint.
    """
    if not drop and not saturate:
        return None
    height, width = shape
    n_pixels = height * width
    if n_pixels <= 0:
        return None
    fraction = max(0.0, min(1.0, float(fraction)))
    n_each = int(round(n_pixels * fraction))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_pixels)
    mask: Dict[str, np.ndarray] = {}
    if drop:
        mask["drop"] = permutation[:n_each]
    if saturate:
        start = n_each if drop else 0
        mask["saturate"] = permutation[start:start + n_each]
    return mask


def apply_demo_defects(image: np.ndarray, mask: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    """Apply a pre-built sensor-defect mask to a grayscale demo image."""
    arr = np.array(image, copy=True)
    if arr.ndim != 2 or arr.size == 0 or not mask:
        return arr
    flat = arr.reshape(-1)
    drop_idx = mask.get("drop")
    if drop_idx is not None and len(drop_idx):
        flat[drop_idx] = 0
    sat_idx = mask.get("saturate")
    if sat_idx is not None and len(sat_idx):
        if np.issubdtype(arr.dtype, np.integer):
            flat[sat_idx] = np.iinfo(arr.dtype).max
        else:
            flat[sat_idx] = float(np.finfo(arr.dtype).max)
    return arr
