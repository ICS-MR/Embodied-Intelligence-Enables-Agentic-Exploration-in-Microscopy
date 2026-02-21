import math
import os
import threading
import time
from datetime import datetime
from queue import Queue, Empty
from typing import List, Dict, Optional, Tuple
import cv2
import numpy as np
from aicsimageio.types import PhysicalPixelSizes
from aicsimageio.writers import OmeTiffWriter
from pymmcore_plus import CMMCorePlus
from sympy import im
from tool import tool_utils
import signal
import json
from mmdet.apis import init_detector, inference_detector
import torch

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

from config.system_config import (
    # Model configuration and weights
    TUMOR_MODEL_CONFIG,
    TUMOR_MODEL_CHECKPOINT,
    BACTERIA_MODEL_CONFIG,
    BACTERIA_MODEL_CHECKPOINT,
    CELL_2D_MODEL_CONFIG,
    CELL_2D_MODEL_CHECKPOINT,
    ORGANOID_MODEL_CONFIG,
    ORGANOID_MODEL_CHECKPOINT
)

TARGET_MODEL_MAP = {
    "organoid": (ORGANOID_MODEL_CONFIG, ORGANOID_MODEL_CHECKPOINT),
    "tumor": (TUMOR_MODEL_CONFIG, TUMOR_MODEL_CHECKPOINT),
    "2Dcell": (CELL_2D_MODEL_CONFIG, CELL_2D_MODEL_CHECKPOINT),
    "bacteria": (BACTERIA_MODEL_CONFIG, BACTERIA_MODEL_CHECKPOINT)
}

global_controller = None

def signal_handler(sig, frame):
    if global_controller:
        global_controller.shutdown_event.set()
        global_controller.shutdown()

def brightness_fitness(gray: np.ndarray, target_ratio: float = 0.5, sigma: float = 0.2) -> float:
    if gray.dtype == np.uint16:
        max_val = 65535.0
    elif gray.dtype == np.uint8:
        max_val = 255.0
    else:
        raise ValueError("Only uint8 or uint16 images are supported")
    mean_val = np.mean(gray.astype(np.float64))
    normalized_mean = mean_val / max_val
    fitness = np.exp(-0.5 * ((normalized_mean - target_ratio) / sigma) ** 2)
    return float(fitness)

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
        self.core = CMMCorePlus()
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
        self.acquisition_thread = None
        self.image_queue = Queue(maxsize=5)  # Only stores image arrays for display
        self.preview_window_name = "micro live"
        self.is_continuous = False
        self.shutdown_event = threading.Event()
        self.img_lock = threading.Lock()

        # Auto contrast
        self.auto_contrast_enabled = True
        self.contrast_percentile = 0.1

    def initialize(self):
        self.core.reset()
        self.core.unloadAllDevices()
        time.sleep(1.0)

        if self.app_dir and self.app_dir not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + self.app_dir

        with self.device_lock:
            self.core.loadSystemConfiguration(self.config_path)

        loaded_devices = self.core.getLoadedDevices()
        required_devices = [self.camera_device, self.xy_stage_device, self.objective_device, self.focus_drive]
        missing = [d for d in required_devices if d not in loaded_devices]
        if missing:
            raise RuntimeError(f"Core devices not loaded: {missing}")

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
        was_continuous = False
        with self.device_lock:
            if self.is_continuous:
                self.core.stopSequenceAcquisition()
                was_continuous = True
        exposure_time = max(self.Min_exposure, min(exposure_time, self.Max_exposure))
        self.core.setProperty(self.camera_device, 'Exposure', exposure_time)
        self.core.waitForDevice(self.camera_device)
        with self.device_lock:
            self.current_exposure_time = exposure_time
            if was_continuous and self.preview_running:
                self.core.startContinuousSequenceAcquisition(0)
                self.is_continuous = True
    @tool_func
    def get_exposure(self) -> float:
        exp = self.core.getProperty(self.camera_device, "Exposure")
        with self.device_lock:
            self.current_exposure_time = float(exp)
        return float(exp)
    @tool_func
    def set_brightness(self, brightness: int):
        current_channel = self.get_channel()
        is_brightfield = (current_channel == '1-NONE')
        if is_brightfield:
            brightness = max(self.Min_brightness, min(brightness, self.Max_brightness))
            self._user_brightness = brightness
        else:
            brightness = 0
        self.core.setProperty(self.transmittedIllumination, 'Brightness', brightness)
        self.core.waitForDevice(self.transmittedIllumination)
        with self.device_lock:
            self.current_brightness = self.get_brightness()
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
        self.core.setStateLabel(self.Dichroic, channel)
        self.core.waitForDevice(self.Dichroic)
        with self.device_lock:
            self.current_channel = channel
        if channel == '1-NONE':
            self.set_brightness(self._user_brightness)
        else:
            self.set_brightness(0)
    @tool_func
    def get_channel(self) -> str:
        with self.device_lock:
            self.current_channel = self.core.getStateLabel(self.Dichroic)
        return self.current_channel

    # ====== Real-time Preview (No Modifications) ======
    def _display_loop(self):
        """独立的显示线程：负责OpenCV窗口渲染，不阻塞主线程"""
        # 初始化窗口
        cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.preview_window_name, 1024, 1024)
        current_frame = np.zeros((self.current_img_height, self.current_img_width, 3), dtype=np.uint8)

        # 显示循环（运行在独立线程）
        self.display_running = True
        while self.display_running and not self.shutdown_event.is_set():
            # 处理键盘事件（非阻塞，30ms超时）
            key = cv2.waitKey(1) & 0xFF  # 缩短等待时间，提升响应性
            if key == ord('q'):
                self.stop_preview()
                break
            
            # 获取队列中的图像（非阻塞）
            try:
                current_frame = self.image_queue.get_nowait()
            except Empty:
                pass  # 无新图像则显示上一帧
            
            # 显示图像
            cv2.imshow(self.preview_window_name, current_frame)

        # 线程结束：清理窗口
        cv2.destroyWindow(self.preview_window_name)
        self.display_running = False

    def start_preview(self):
        if self.preview_running:
            return
        
        # 重置状态
        self.preview_running = True
        self.shutdown_event.clear()
        
        # 1. 启动采集线程
        self.acquisition_thread = threading.Thread(target=self._acquisition_loop, daemon=True)
        self.acquisition_thread.start()
        
        # 2. 启动显示线程（核心改造：把显示逻辑移到独立线程）
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        
        print("预览已启动（采集+显示双线程，主线程未阻塞）")

    @tool_func
    def stop_preview(self):
        """停止预览（安全停止所有线程，清理资源）"""
        if not self.preview_running:
            return
        
        # 1. 停止标志位
        self.preview_running = False
        self.display_running = False
        self.shutdown_event.set()
        
        # 2. 停止采集
        with self.device_lock:
            if self.is_continuous:
                self.core.stopSequenceAcquisition()
                self.is_continuous = False
        
        # 3. 等待线程结束（带超时，避免卡死）
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=1.0)
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
        
        # 4. 清空队列
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except Empty:
                break
        
        print("预览已停止，资源已清理")

    def _acquisition_loop(self):
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
                # Only process display images, no metadata encapsulation, directly store in queue
                processed_img = self._process_image_for_display(img.copy())
                if self.image_queue.full():
                    try:
                        self.image_queue.get_nowait()
                    except Empty:
                        pass
                try:
                    self.image_queue.put(processed_img, timeout=0.01)
                except:
                    pass
                time.sleep(0.01)
        except Exception as e:
            pass
        finally:
            with self.device_lock:
                if self.is_continuous:
                    self.core.stopSequenceAcquisition()
                    self.is_continuous = False

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

    def get_live_preview_image(self) -> Optional[np.ndarray]:
        """Only returns image array for preview display, no metadata"""
        if not self.preview_running:
            return None
        try:
            with self.img_lock:
                if not self.image_queue.empty():
                    while self.image_queue.qsize() > 1:
                        self.image_queue.get_nowait()
                    return self.image_queue.get_nowait().copy()
            # If no queue data, acquire one and process as display image
            raw_img = self._acquire_single_image().image
            return self._process_image_for_display(raw_img)
        except:
            return None

    # ====== Image Acquisition (No Modifications, Formal Acquisition Returns ImagingData) ======
    def _get_image(self, width_micro=None, height_micro=None) -> ImagingData:
        if width_micro and height_micro:
            return self._acquire_stitch_mosaic(width_micro, height_micro)
        else:
            return self._acquire_single_image()

    def _acquire_single_image(self) -> ImagingData:
        """Formal single image acquisition: returns ImagingData with metadata"""
        with self.device_lock:
            was_continuous = self.is_continuous
            if was_continuous and self.preview_running:
                time.sleep(1)
                img = self.img.copy()
            else:
                self.core.snapImage()
                img = self.core.getImage()
                if img is None:
                    raise RuntimeError("Acquisition failed")
        # Get current center coordinates and objective information
        current_x, current_y = self.get_x_y_position()
        current_z = self.get_z_position()
        current_obj = self.get_objective()
        return ImagingData(
            image=img,
            center_x=current_x,
            center_y=current_y,
            center_z=current_z,
            objective_magnification=current_obj,
            pixel_size=self.pixel_size
        )

    def _acquire_stitch_mosaic(self, width_micro: float, height_micro: float, overlap=0) -> ImagingData:
        """Formal stitched acquisition: returns ImagingData with metadata"""
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


    # ====== Auto Acquisition (Key Modification: Returns List[ImagingData]) ======
    @tool_func
    def add_acquisition_position(self, name: str, x: float, y: float, width: float, height: float) -> None:
        """添加自动采集的位置点"""
        self.acquisition_positions.append({
            "name": name,
            "x": x,
            "y": y,
            'width': width,
            'height': height
        })
    @tool_func
    def add_channels(self, channel: str, exposure: float) -> None:
        """添加自动采集的通道配置"""
        self.acquisition_channels.append({
            "channel": channel,
            "exposure": exposure
        })
    @tool_func
    def set_z_stack(self, z_start: float, z_end: float, z_step: float) -> None:
        """配置Z轴层扫参数"""
        if z_step <= 0:
            raise ValueError("Z轴步长必须为正数")
        if (z_end - z_start) * z_step < 0:
            raise ValueError("Z轴步长方向与起止范围冲突")
        self.z_stack_params = {
            "z_start": z_start,
            "z_end": z_end,
            "z_step": z_step
        }
    @tool_func
    def set_time_series(self, num_frames: int, interval_sec: float) -> None:
        """配置时间序列参数"""
        self.time_lapse_params = {
            "num_frames": num_frames,
            "interval_sec": interval_sec
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
        # Initialize return result list
        acquisition_imaging_data_list = []
        if not self.acquisition_positions:
            raise ValueError("Please add acquisition positions")
        if not self.acquisition_channels:
            raise ValueError("Please configure channels")

        if not self.time_lapse_params:
            self.time_lapse_params = {"num_frames": 1, "interval_sec": 0}
        if not self.z_stack_params:
            current_z = self.get_z_position()
            self.z_stack_params = {"z_start": current_z, "z_end": current_z, "z_step": 1}

        init_x, init_y = self.get_x_y_position()
        init_z = self.get_z_position()
        init_channel = self.get_channel()
        init_exposure = self.get_exposure()

        time_num_frames = self.time_lapse_params["num_frames"]
        tim_interval = self.time_lapse_params["interval_sec"]
        z_start = self.z_stack_params['z_start']
        z_end = self.z_stack_params['z_end']
        z_step = self.z_stack_params['z_step']

        num_steps = 1 if z_start == z_end else int(round((z_end - z_start) / z_step)) + 1
        z_positions = np.linspace(z_start, z_end, num_steps)
        channels_name = [ch["channel"] for ch in self.acquisition_channels]

        position_data = []
        # Initialize basic information for each acquisition position
        for pos in self.acquisition_positions:
            try:
                # Get test image of this position (for initializing data shape)
                test_imaging_data = self._get_image(pos["width"], pos["height"])
                test_img = test_imaging_data.image
                data = np.zeros((time_num_frames, len(channels_name), num_steps, test_img.shape[0], test_img.shape[1]),
                                dtype=self.img_dtype)
                # Replace original metadata to create row
                metadata = self._create_ome_metadata(
                    channel_names=channels_name,
                    time_interval=tim_interval,
                    microscope="olympus lx83",
                    objective=self.current_objective,
                    pixel_type=self.img_dtype,
                    center_x=test_imaging_data.center_x,
                    center_y=test_imaging_data.center_y,
                    center_z=test_imaging_data.center_z
                )
                position_data.append({
                    "name": pos["name"],
                    "metadata": metadata,
                    "data": data,
                    "x": pos["x"],
                    "y": pos["y"],
                    "width": pos["width"],
                    "height": pos["height"],
                    "base_imaging_data": test_imaging_data  # Store basic metadata
                })
            except Exception as e:
                pass

        try:
            for t_idx in range(time_num_frames):
                if self.shutdown_event.is_set():
                    break
                start_time = time.time()
                for pos_item in position_data:
                    self.set_x_y_position(pos_item["x"], pos_item["y"])
                    time.sleep(1)
                    orig_ch = self.current_channel
                    orig_z = self.current_Z_position
                    for c, ch_cfg in enumerate(self.acquisition_channels):
                        self.set_channel(ch_cfg["channel"])
                        self.set_exposure(ch_cfg["exposure"])
                        for z_idx, z in enumerate(z_positions):
                            self.set_z_position(z)
                            time.sleep(1)
                            # Acquire image for current position-channel-Z layer
                            imaging_data = self._get_image(pos_item["width"], pos_item["height"])
                            img = imaging_data.image
                            pos_item["data"][t_idx, c, z_idx] = img
                    self.set_channel(orig_ch)
                    self.set_z_position(orig_z)

                if time_num_frames > 1 and t_idx < time_num_frames - 1:
                    elapsed = time.time() - start_time
                    wait_time = max(0, tim_interval - elapsed)
                    time.sleep(wait_time)

            # Save files and build returned ImagingData list
            pixel_sizes = PhysicalPixelSizes(
                Z=self.z_stack_params['z_step'],
                Y=self.pixel_size,
                X=self.pixel_size
            )
            for pos_item in position_data:
                # Save OME-TIFF file
                save_path = os.path.join(self.output_directory, f"{pos_item['name']}.ome.tif")
                os.makedirs(self.output_directory, exist_ok=True)
                self._save_ome_tiff(pos_item["data"], save_path, pixel_sizes, pos_item["metadata"])
                channel_colors = [dichroic_colors.get(ch, "Unknown") for ch in channels_name]
                desc = f'"channel_names": {channel_colors}, pixel_size: {self.pixel_size}, "magnification": {self.current_objective}'
                self._storagemanger.register_file(f"{pos_item['name']}.ome.tif", desc, 'microscope', 'ome-tiff')

                # Build ImagingData instance for this position (merge time series/Z-stack/channel data into multi-dimensional image)
                position_imaging_data = ImagingData(
                    image=pos_item["data"],  # Multi-dimensional image data: (T, C, Z, Y, X)
                    center_x=pos_item["base_imaging_data"].center_x,
                    center_y=pos_item["base_imaging_data"].center_y,
                    center_z=pos_item["base_imaging_data"].center_z,
                    objective_magnification=pos_item["base_imaging_data"].objective_magnification
                )
                # Set acquisition position name for easy identification
                position_imaging_data.position_name = pos_item["name"]
                # Add to return list
                acquisition_imaging_data_list.append(position_imaging_data)

        except Exception as e:
            pass
        finally:
            # Reset acquisition parameters and device state
            self.acquisition_positions.clear()
            self.acquisition_channels.clear()
            self.z_stack_params = None
            self.time_lapse_params = None
            self.set_x_y_position(init_x, init_y)
            self.set_z_position(init_z)
            self.set_channel(init_channel)
            self.set_exposure(init_exposure)

        # Return ImagingData list containing all acquisition information
        return acquisition_imaging_data_list

    def _save_ome_tiff(self, data, save_path, pixel_sizes, metadata):
        # Extract center position (if exists)
        center_x = metadata.get("center_x")
        center_y = metadata.get("center_y")
        center_z = metadata.get("center_z")

        # Calculate Position for each plane (assuming all planes share the same XY position, Z can vary with stack)
        # data shape: (T, C, Z, Y, X)
        t, c, z = data.shape[:3]

        # Default: all planes use the same XY center, Z increases from center_z by step
        if hasattr(pixel_sizes, 'Z') and pixel_sizes.Z not in (None, 0):
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

        # Call OmeTiffWriter and pass position parameters
        OmeTiffWriter.save(
            data,
            save_path,
            dim_order="TCZYX",
            physical_pixel_sizes=pixel_sizes,
            # Basic metadata
            channel_names=metadata["channel_names"],
            channel_colors=metadata["channel_colors"],
            time_interval=metadata["time_interval"],
            microscope=metadata["microscope"],
            objective=metadata["objective"],
            # Key: Embed physical positions
            plane_position_x=plane_position_x,
            plane_position_y=plane_position_y,
            plane_position_z=plane_position_z,
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
            "datetime": datetime.now().isoformat(),
            "pixel_type": pixel_type.name
        }

        # If center position is provided, add to metadata (for _save_ome_tiff usage)
        if center_x is not None and center_y is not None:
            metadata["center_x"] = center_x
            metadata["center_y"] = center_y
            metadata["center_z"] = center_z if center_z is not None else 0.0

        return metadata
    
    # ====== Auto Focus / Brightness (No Modifications) ======
    @tool_func
    def perform_autofocus(self, tolerance=0.5, search_range=500.0) -> float:
        initial_step = min(50.0, search_range)
        center_z = self.get_z_position()
        best_z = center_z
        best_score = -float('inf')

        imaging_data = self._acquire_single_image()
        img = imaging_data.image
        current_score = tool_utils.tenengrad_calculate_sharpness(img)
        best_score = current_score

        direction = 0
        for d in [1, -1]:
            test_z = center_z + d * initial_step
            if abs(test_z - center_z) > search_range or not (self.Min_Z_position <= test_z <= self.Max_Z_position):
                continue
            self.set_z_position(test_z)
            imaging_data = self._acquire_single_image()
            img = imaging_data.image
            score = tool_utils.tenengrad_calculate_sharpness(img)
            if score > best_score:
                best_score = score
                best_z = test_z
                direction = d

        if direction == 0:
            direction = 1

        step = initial_step
        iterations = 0
        while step >= tolerance and iterations < 100:
            next_z = best_z + direction * step
            if abs(next_z - center_z) > search_range or not (self.Min_Z_position <= next_z <= self.Max_Z_position):
                step /= 2.0
                direction *= -1
                continue
            self.set_z_position(next_z)
            imaging_data = self._acquire_single_image()
            img = imaging_data.image
            score = tool_utils.tenengrad_calculate_sharpness(img)
            if score > best_score:
                best_score = score
                best_z = next_z
            else:
                step /= 2.0
                direction *= -1
            iterations += 1

        if abs(best_z - self.get_z_position()) > 0.1:
            self.set_z_position(best_z)
        return best_z

    @tool_func
    def perform_autobrightness(self, target_fitness_threshold=0.95, fitness_target_ratio=0.6, fitness_sigma=0.2, max_iterations=8) -> int:
        max_pixel_value = 65535.0 if self.is_16bit else 255.0
        original_brightness = self.current_brightness
        imaging_data = self._acquire_single_image()
        img = imaging_data.image
        best_fitness = brightness_fitness(img, fitness_target_ratio, fitness_sigma)
        best_brightness = original_brightness

        min_br = max(self.Min_brightness, original_brightness - 150)
        max_br = min(self.Max_brightness, original_brightness + 50, int(original_brightness * 0.8))
        test_brightnesses = sorted(set([
            max(min_br, original_brightness - 120),
            max(min_br, original_brightness - 60),
            original_brightness,
            min(max_br, original_brightness + 30)
        ]))

        brightness_samples = []
        fitness_samples = []

        for br in test_brightnesses:
            self.set_brightness(br)
            imaging_data = self._acquire_single_image()
            img = imaging_data.image
            fitness = brightness_fitness(img, fitness_target_ratio, fitness_sigma)
            brightness_samples.append(br)
            fitness_samples.append(fitness)
            if fitness > best_fitness:
                best_fitness = fitness
                best_brightness = br
            if best_fitness >= target_fitness_threshold:
                self.set_brightness(best_brightness)
                return best_brightness

        for _ in range(max_iterations):
            if len(brightness_samples) < 3:
                break
            try:
                coeffs = np.polyfit(brightness_samples, fitness_samples, 2)
                a, b, c = coeffs
                if abs(a) < 1e-10:
                    pred_br = brightness_samples[np.argmax(fitness_samples)]
                else:
                    pred_br = int(-b / (2 * a))
                    pred_br = max(min_br, min(max_br, pred_br))
                    if best_fitness < target_fitness_threshold:
                        pred_br = max(min_br, pred_br - 15)
                if pred_br in brightness_samples:
                    pred_br = max(min_br, pred_br - 10)
                if pred_br in brightness_samples:
                    break

                self.set_brightness(pred_br)
                imaging_data = self._acquire_single_image()
                img = imaging_data.image
                fitness = brightness_fitness(img, fitness_target_ratio, fitness_sigma)
                brightness_samples.append(pred_br)
                fitness_samples.append(fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_brightness = pred_br
                if best_fitness >= target_fitness_threshold:
                    break
            except Exception as e:
                break

        self.set_brightness(best_brightness)
        return best_brightness

    # ====== System Control (No Modifications) ======

    @tool_func
    def shutdown(self):
        self.shutdown_event.set()
        if self.preview_running:
            self.stop_preview()
        if self.acquisition_thread and self.acquisition_thread.is_alive():
            self.acquisition_thread.join(timeout=5.0)
        try:
            with self.device_lock:
                self.set_brightness(self.Min_brightness)
                self.core.stopSequenceAcquisition()
                self.core.reset()
                self.core.unloadAllDevices()
        except Exception as e:
            pass
    @tool_func
    def load_target_locations(self, filename: str) -> List[Tuple[int, int, int, int]]:
        try:
            filename = os.path.join(self.output_directory, filename)
            with open(filename, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            regions = []
            for item in loaded_data:
                if isinstance(item, (list, tuple)) and len(item) == 4:
                    x, y, width, height = map(int, item)
                    regions.append((x, y, width, height))
                else:
                    continue

            return regions
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            return []
        except Exception as e:
            return []
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
        ROWS = 8  
        COLS = 12  

        X_SPACING_UM = 9000.0  
        Y_SPACING_UM = 9000.0  
        A1_X_UM = 0.0
        A1_Y_UM = 0.0

        # 存储所有孔位坐标的列表
        well_positions = []

        for row_idx in range(ROWS):
            for col_idx in range(COLS):
                current_x_um = A1_X_UM + (col_idx * X_SPACING_UM)
                current_y_um = A1_Y_UM + (row_idx * Y_SPACING_UM)
                well_positions.append((current_x_um, current_y_um))

        return well_positions
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

        ROWS = 4  
        COLS = 6  
        X_SPACING_UM = 12700.0  
        Y_SPACING_UM = 12700.0  
        A1_X_UM = 0.0
        A1_Y_UM = 0.0

        well_positions = []

        for row_idx in range(ROWS):
            for col_idx in range(COLS):
                current_x_um = A1_X_UM + (col_idx * X_SPACING_UM)
                current_y_um = A1_Y_UM + (row_idx * Y_SPACING_UM)
                well_positions.append((current_x_um, current_y_um))

        return well_positions

    @tool_func
    def z_stack_range(self) -> Tuple[float, float]:
        """
        Calculates recommended Z-stack scanning range (μm) based on current image.
        
        Strategy:
            - Explore Z range: [current_z - 200, current_z + 200] μm
            - Two-thread parallel execution:
                * Z-motion thread: move Z continuously without pause
                * Capture thread: fetch live preview image + sharpness score synchronously
            - Identify plateau region (stable high sharpness) as recommended range

        Returns:
            (z_max, z_min): Recommended maximum and minimum Z positions for stacking (μm)
        """
        from collections import deque
        import threading

        # === 参数配置 ===
        RANGE_HALF_SPAN = 200.0  # ±200 μm
        MOVE_SPEED_UM_PER_SEC = 80.0  # 匀速移动速度（根据硬件能力调整）
        SAMPLE_RATE_HZ = 10  # 采集频率（每秒采样次数）
        SLEEP_INTERVAL = 1.0 / SAMPLE_RATE_HZ
        PLATEAU_REL_TOL = 0.15  # 平台容忍度（相对最大值的百分比）
        MIN_PLATEAU_WIDTH_UM = 30.0  # 平台最小物理宽度（避免噪声）

        # === 初始状态保存 ===
        orig_z = self.get_z_position()
        orig_brightness = self.get_brightness()
        orig_exposure = self.get_exposure()
        orig_channel = self.get_channel()

        # 确保处于预览状态以获取实时图像
        was_preview_running = self.preview_running
        if not was_preview_running:
            self.start_preview()
            time.sleep(1.0)  # 等待预览稳定

        z_start = max(self.Min_Z_position, orig_z - RANGE_HALF_SPAN)
        z_end = min(self.Max_Z_position, orig_z + RANGE_HALF_SPAN)
        total_distance = abs(z_end - z_start)
        move_duration = total_distance / MOVE_SPEED_UM_PER_SEC

        # === 共享数据结构 ===
        sharpness_log = deque()  # 存储 (z_pos, score)
        shutdown采集 = threading.Event()

        # === 采集线程：边移动边打分 ===
        def capture_and_score():
            last_z = self.get_z_position()
            while not shutdown采集.is_set():
                try:
                    # 获取当前Z（必须与图像严格对齐！）
                    current_z = self.get_z_position()
                    # 避免重复记录相同位置
                    if abs(current_z - last_z) < 0.1:
                        time.sleep(SLEEP_INTERVAL)
                        continue

                    # 获取最新预览图像（已处理为 uint8 但不影响 sharpness）
                    img = self.get_live_preview_image()
                    if img is None:
                        time.sleep(SLEEP_INTERVAL)
                        continue

                    # 转回灰度（若为彩色）
                    if len(img.shape) == 3:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = img

                    # 注意：原始图像为 uint8（因预览已归一化），但 tenengrad 仍有效
                    # 若需更准确，可临时切出 preview 获取 raw img，但会复杂化；
                    # 此处接受轻微精度损失以换取速度。
                    score = tool_utils.tenengrad_calculate_sharpness(gray)

                    sharpness_log.append((current_z, score))
                    last_z = current_z
                    time.sleep(SLEEP_INTERVAL)
                except Exception as e:
                    time.sleep(SLEEP_INTERVAL)

        # === 启动采集线程 ===
        capture_thread = threading.Thread(target=capture_and_score, daemon=True)
        capture_thread.start()

        # === 移动线程：连续移动 Z 轴 ===
        try:
            # 先跳到起始位置（允许停顿）
            self.set_z_position(z_start)
            time.sleep(0.5)

            # 启动匀速移动：通过小步连续移动模拟“连续”
            step_um = MOVE_SPEED_UM_PER_SEC * SLEEP_INTERVAL  # 每次移动距离
            direction = 1 if z_end > z_start else -1
            current_move_z = z_start

            # 开始移动
            start_time = time.time()
            while (time.time() - start_time) < move_duration and not self.shutdown_event.is_set():
                next_z = current_move_z + direction * step_um
                if (direction > 0 and next_z > z_end) or (direction < 0 and next_z < z_end):
                    break
                # 无停顿设置 Z（仅硬件延迟）
                self.core.setFocusDevice(self.focus_drive)
                self.core.setPosition(next_z)
                current_move_z = next_z
                time.sleep(SLEEP_INTERVAL)

            # 确保最终位置
            self.set_z_position(z_end)

        finally:
            # 停止采集线程
            shutdown采集.set()
            capture_thread.join(timeout=1.0)

            # 恢复原始状态
            if not was_preview_running:
                self.stop_preview()
            self.set_z_position(orig_z)
            self.set_channel(orig_channel)
            self.set_exposure(orig_exposure)
            self.set_brightness(orig_brightness)

        # === 数据后处理：找平台区间 ===
        if len(sharpness_log) < 5:
            return (orig_z + 50.0, orig_z - 50.0)

        z_vals, scores = zip(*sharpness_log)
        z_vals = np.array(z_vals)
        scores = np.array(scores)

        # 平滑（可选）
        from scipy.signal import savgol_filter
        try:
            if len(scores) > 11:
                scores_smooth = savgol_filter(scores, window_length=11, polyorder=2)
            else:
                scores_smooth = scores
        except:
            scores_smooth = scores

        max_score = np.max(scores_smooth)
        threshold = max_score * (1.0 - PLATEAU_REL_TOL)

        # 找出高于阈值的连续区域
        above = scores_smooth >= threshold
        # 找连续段
        from itertools import groupby
        best_span = (orig_z - 50, orig_z + 50)  # fallback
        max_width = 0

        for k, g in groupby(enumerate(above), key=lambda x: x[1]):
            if k:  # True segment
                indices = [i for i, _ in g]
                if len(indices) < 2:
                    continue
                z_low = z_vals[indices[0]]
                z_high = z_vals[indices[-1]]
                width = abs(z_high - z_low)
                if width >= MIN_PLATEAU_WIDTH_UM and width > max_width:
                    max_width = width
                    best_span = (z_high, z_low) if z_high > z_low else (z_low, z_high)

        z_max, z_min = best_span
        return (z_max, z_min)

    @tool_func
    def detect_targets_in_image(
            self,
            image: np.ndarray,
            target_class: str,
            pixel_size: float,
            confidence_threshold: float = 0.5,
            device: Optional[torch.device] = None
    ) -> List[Dict[str, float]]:
        if image.ndim != 2:
            raise ValueError("Only 2D grayscale image supported")

        h, w = image.shape
        img_center_x_px = w / 2.0
        img_center_y_px = h / 2.0

        if device is None:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_cache_key = f"_mmdet_model_{target_class}"

        # 加载模型（带缓存）
        try:
            if hasattr(self, model_cache_key) and getattr(self, model_cache_key) is not None:
                model = getattr(self, model_cache_key)
            else:
                if target_class not in TARGET_MODEL_MAP:
                    raise ValueError(f"Target class '{target_class}' not in TARGET_MODEL_MAP")
                config_path, ckpt_path = TARGET_MODEL_MAP[target_class]
                model = init_detector(config_path, ckpt_path, device=device)
                setattr(self, model_cache_key, model)
        except Exception as e:
            return []

        # 图像预处理 → uint8 RGB
        if self.auto_contrast_enabled:
            low, high = np.percentile(image, [self.contrast_percentile, 100 - self.contrast_percentile])
            img_norm = np.clip(image, low, high)
            img_uint8 = ((img_norm - low) / (high - low + 1e-8) * 255).astype(np.uint8)
        else:
            img_max = float(np.max(image))
            img_uint8 = (image / (img_max + 1e-8) * 255).astype(np.uint8) if img_max > 0 else image.astype(np.uint8)

        img_rgb = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)

        # 推理
        try:
            det_results = inference_detector(model, img_rgb)
        except Exception as e:
            return []

        # 解析结果
        if target_class not in model.CLASSES:
            return []

        class_idx = model.CLASSES.index(target_class)
        class_dets = det_results[class_idx]  # (N, 5)

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
            offset_y_um = -(cy_px - img_center_y_px) * pixel_size  # Y 轴翻转

            results.append({
                "offset_x_um": float(offset_x_um),
                "offset_y_um": float(offset_y_um),
                "confidence": float(score),
            })
        return results
