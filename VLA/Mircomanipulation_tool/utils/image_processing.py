import cv2
import numpy as np

_LUT_CACHE = {
    'gray': [np.arange(256, dtype=np.uint8)] * 3,
    
    'blue': [
        np.arange(256, dtype=np.uint8),
        np.zeros(256, dtype=np.uint8),
        np.zeros(256, dtype=np.uint8)
    ],
    
    'green': [
        np.zeros(256, dtype=np.uint8),
        np.arange(256, dtype=np.uint8),
        np.zeros(256, dtype=np.uint8)
    ],
    
    'red': [
        np.zeros(256, dtype=np.uint8),
        np.zeros(256, dtype=np.uint8),
        np.arange(256, dtype=np.uint8)
    ]
}

def _auto_adjust_channel(channel, saturation, use_lut, downsample = 1):
    """Adjust one image channel by percentile clipping."""
    if downsample > 1:
        h, w = channel.shape
        small_ch = cv2.resize(channel, (w//downsample, h//downsample), 
                            interpolation=cv2.INTER_AREA)
    else:
        small_ch = channel

    low_p = saturation / 100.0
    high_p = 1 - low_p
    low_val, high_val = np.percentile(small_ch, [low_p * 100, high_p * 100])
    
    if use_lut:
        lut = _build_lut(low_val, high_val)
        return cv2.LUT(channel, lut)
    else:
        adjusted = np.clip(
            (channel - low_val) * (255.0 / (high_val - low_val)), 0, 255
        )
        return adjusted.astype(np.uint8)
    
def _build_lut(low_val, high_val):

    indices = np.arange(256, dtype=np.float32)
    scale = 255.0 / (high_val - low_val + 1e-5)
    lut = (indices - low_val) * scale
    return np.clip(lut, 0, 255).astype(np.uint8)

    
class image_process:
    def __init__(self, saturation, downsample = 1, use_lut = True):

        self.saturation = saturation
        self.downsample = downsample
        self.use_lut = use_lut

    def auto_adjust_brightness_contrast(self, image):
        """Apply ImageJ-style auto brightness/contrast adjustment."""
        if image is None:
            return None
        
        if len(image.shape) == 3:
            channels = cv2.split(image)
            adjusted_channels = [
                _auto_adjust_channel(ch, self.saturation, self.use_lut) for ch in channels
            ]
            return cv2.merge(adjusted_channels)
        else:
            return _auto_adjust_channel(image, self.saturation, self.use_lut)
        
    def set_color(self, image_8bit: np.ndarray, color: str):
        """Apply a cached gray/blue/green/red pseudocolor map."""
        if image_8bit.ndim != 2:
            image_8bit = cv2.cvtColor(image_8bit, cv2.COLOR_BGR2GRAY)

        lut_b, lut_g, lut_r = _LUT_CACHE[color.lower()]
        
        return cv2.merge([
            cv2.LUT(image_8bit, lut_b),
            cv2.LUT(image_8bit, lut_g),
            cv2.LUT(image_8bit, lut_r)
        ])

def adjust_hsv_brightness(img_bgr, saturation=0.1):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    v_adjusted = _auto_adjust_channel(v, saturation, use_lut=True)
    img_hsv_adj = cv2.merge([h, s, v_adjusted])
    return cv2.cvtColor(img_hsv_adj, cv2.COLOR_HSV2BGR)
