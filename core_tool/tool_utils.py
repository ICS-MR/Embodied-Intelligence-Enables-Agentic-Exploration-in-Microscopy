import numpy as np

try:
    import cv2 as cv
except Exception:  # pragma: no cover - optional in test and mock environments
    cv = None


def quadratic_function(z, a, b, c):
    """Define quadratic function model: f(z) = a*z² + b*z + c"""
    return a * z ** 2 + b * z + c


def find_peak_position(a, b):
    """Calculate the peak position of the quadratic function: z = -b/(2a)"""
    if a == 0:
        raise ValueError("Quadratic coefficient cannot be zero")
    return -b / (2 * a)


def _require_cv2() -> None:
    if cv is None:
        raise ImportError("cv2 is required for image sharpness utilities")


def _blur(gray):
    """Optimized median filtering for 16-bit grayscale images"""
    _require_cv2()
    kernel_size = 5
    filtered = cv.medianBlur(gray, kernel_size)
    return filtered


def _var_calculate_sharpness(image):
    """Calculate sharpness score of 16-bit images using Laplacian operator"""
    _require_cv2()
    laplacian = cv.Laplacian(image, cv.CV_16S, ksize=3)
    variance = np.var(laplacian)
    return variance


def _get_center_region(img, output_size=1024):
    """Extract central region of image with optimized processing for 16-bit images"""
    height, width = img.shape[:2]

    output_size = min(output_size, height, width)

    y_start = (height - output_size) // 2
    x_start = (width - output_size) // 2

    center_region = img[y_start:y_start + output_size, x_start:x_start + output_size]

    return center_region


def tenengrad_calculate_sharpness(img, threshold_scale=2.0, center_roi_size=None):
    """
    Calculate sharpness score based on adaptive Tenengrad algorithm for input image.

    :param img: Input image data
    :param threshold_scale: Threshold scaling factor (default: 2.0)
    :param center_roi_size: Optional center ROI width/height in pixels.
    :return: Sharpness score (higher values indicate sharper images)
    """
    _require_cv2()
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        pass
    else:
        raise ValueError("Invalid image format. Please provide a valid image.")

    if center_roi_size is not None:
        img = _get_center_region(img, int(center_roi_size))

    img = _blur(img)
    img_float = img.astype(np.float32)
    gx = cv.Sobel(img_float, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(img_float, cv.CV_32F, 0, 1, ksize=3)
    gradient_sq = gx ** 2 + gy ** 2
    mean_gradient = np.mean(gradient_sq)
    dynamic_threshold = threshold_scale * mean_gradient
    mask = gradient_sq > dynamic_threshold
    score = np.sum(gradient_sq[mask])

    return score


from typing import Callable, List, Optional


class SayCapture:
    def __init__(self):
        self.messages: List[str] = []
        self._listener: Optional[Callable[[str], None]] = None

    def say(self, message):
        msg_str = str(message)
        self.messages.append(msg_str)
        if self._listener is not None:
            self._listener(msg_str)
        print(f"[Robot action] {msg_str}")

    def get_messages(self) -> List[str]:
        return self.messages.copy()

    def set_listener(self, listener: Optional[Callable[[str], None]]) -> None:
        self._listener = listener

    def clear(self):
        self.messages.clear()
