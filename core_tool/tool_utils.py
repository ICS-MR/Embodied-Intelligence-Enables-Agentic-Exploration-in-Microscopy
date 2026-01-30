import numpy as np
import cv2 as cv


def quadratic_function(z, a, b, c):
    """Define quadratic function model: f(z) = a*z² + b*z + c"""
    return a * z ** 2 + b * z + c


def find_peak_position(a, b):
    """Calculate the peak position of the quadratic function: z = -b/(2a)"""
    if a == 0:
        raise ValueError("Quadratic coefficient cannot be zero")
    return -b / (2 * a)


def _blur(gray):
    """Optimized median filtering for 16-bit grayscale images"""
    # Use small kernel for 16-bit images to prevent data loss
    kernel_size = 5
    filtered = cv.medianBlur(gray, kernel_size)
    return filtered


def _var_calculate_sharpness(image):
    """Calculate sharpness score of 16-bit images using Laplacian operator"""
    # Use Laplacian operator to compute variance of second-order derivatives as sharpness metric
    laplacian = cv.Laplacian(image, cv.CV_16S, ksize=3)
    # Calculate variance of Laplacian results
    variance = np.var(laplacian)
    return variance


def _get_center_region(img, output_size=1024):
    """Extract central region of image with optimized processing for 16-bit images"""
    height, width = img.shape[:2]

    # Ensure output size does not exceed original dimensions
    output_size = min(output_size, height, width)

    y_start = (height - output_size) // 2
    x_start = (width - output_size) // 2

    # Extract central region
    center_region = img[y_start:y_start + output_size, x_start:x_start + output_size]

    return center_region


def tenengrad_calculate_sharpness(img, threshold_scale=2.0):
    """
    Calculate sharpness score based on adaptive Tenengrad algorithm for input image.

    :param img: Input image data
    :param threshold_scale: Threshold scaling factor (default: 2.0)
    :return: Sharpness score (higher values indicate sharper images)
    """
    if len(img.shape) == 3:  # Color image
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:  # Grayscale image
        pass
    else:
        raise ValueError("Invalid image format. Please provide a valid image.")

    img = _blur(img)

    # Convert to float to avoid overflow
    img_float = img.astype(np.float32)

    # Calculate Sobel gradients (horizontal and vertical directions)
    gx = cv.Sobel(img_float, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(img_float, cv.CV_32F, 0, 1, ksize=3)

    # Calculate sum of squared gradient magnitudes
    gradient_sq = gx ** 2 + gy ** 2

    # Dynamically calculate threshold: mean gradient * scaling factor
    mean_gradient = np.mean(gradient_sq)
    dynamic_threshold = threshold_scale * mean_gradient

    # Only count gradient regions above threshold
    mask = gradient_sq > dynamic_threshold
    score = np.sum(gradient_sq[mask])

    return score


from collections import deque
from typing import List

class SayCapture:
    def __init__(self):
        self.messages: List[str] = []

    def say(self, message):
        msg_str = str(message)
        self.messages.append(msg_str)
        print(f"[Robot action] {msg_str}")

    def get_messages(self) -> List[str]:
        return self.messages.copy()

    def clear(self):
        self.messages.clear()
        