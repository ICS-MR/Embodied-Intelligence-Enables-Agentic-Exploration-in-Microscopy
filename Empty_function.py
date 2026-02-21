from __future__ import annotations

from tool.base import BaseTool,tool_func
import math
import threading
from datetime import datetime
from queue import Queue  # Thread-safe queue for image transmission
from typing import List, Dict, Tuple, Optional
import numpy as np
import os

from aicsimageio.types import PhysicalPixelSizes


# Map objective magnification to objective labels
objective_labels = {
    '1-UPLFLN4XPH': 4,
    '2-SOB': 10,
    '3-LUCPLFLN20XRC': 20,
    '4-LUCPLFLN40X': 40,
    '5-LUCPLFLN60X': 60,
    '6-UPLSAPO30XS': 30
}

# Channel to color mapping (RGB values)
dichroic_colors = {
    '1-NONE': (128, 128, 128),  # Gray (brightfield)
    '2-U-FUNA': (0, 0, 255),  # Red
    '3-U-FBNA': (0, 255, 0),  # Green
    '4-U-FGNA': (255, 0, 0),  # Blue
}


class MicroscopeController(BaseTool):
    """
    Core class for microscope control system, supporting real-time preview, image acquisition, 
    and multi-dimensional automated acquisition.
    All physical units are in micrometers (μm) by default, and time units are in milliseconds (ms) by default.
    Test version: Keep input/output structure without implementing actual hardware control functions
    """

    def __init__(self, config_path: str, app_dir: str, output_path: str, storagemanger):
        self._storagemanger = storagemanger
        self.app_dir = app_dir
        self.config_path = config_path

        self.device_lock = threading.RLock()

        self.camera_device = 'Camera-1'
        self.xy_stage_device = 'XYStage'
        self.objective_device = 'Objective'
        self.transmittedIllumination = 'TransmittedIllumination 2'
        self.focus_drive = 'FocusDrive'
        self.Dichroic = 'Dichroic 2'

        self.Max_X_position: float = 500000
        self.Min_X_position: float = 0
        self.Max_Y_position: float = 500000
        self.Min_Y_position: float = 0
        self.Max_Z_position: float = 10000
        self.Min_Z_position: float = 0
        self.Max_brightness = 250
        self.Min_brightness = 0
        self.Max_exposure = 1000
        self.Min_exposure = 0

        self.current_channel = '1-NONE'
        self.current_objective = '1-UPLFLN4XPH'
        self.current_X_position = 0.0
        self.current_Y_position = 0.0
        self.current_Z_position = 0.0
        self.current_brightness = 100
        self.current_exposure_time = 100.0

        self.acquisition_positions: List[Dict] = []
        self.acquisition_channels: List[Dict] = []
        self.z_stack_params: Optional[Dict] = None
        self.time_lapse_params: Optional[Dict] = None
        self.output_directory: str = output_path

        self.pixel_size = 1.6234  # μm/pixel
        self.img_dtype = np.uint16
        self.current_img_height = 1024
        self.current_img_width = 1024
        self.is_16bit = True

        self.preview_running = False
        self.acquisition_thread = None
        self.image_queue = Queue(maxsize=5)
        self.preview_interval = 0.04
        self.preview_window_name = "micro live"
        self.is_continuous = False
        self.display_ready = threading.Event()

        self.acquisition_running = False
        self.acquisition_abort = False

        self.auto_contrast_enabled = True
        self.contrast_percentile = 0.1

    def initialize(self):
        pass

    @tool_func
    def set_x_y_position(self, x: float, y: float):
        print("Running function: set_x_y_position")
        x = max(self.Min_X_position, min(x, self.Max_X_position))
        y = max(self.Min_Y_position, min(y, self.Max_Y_position))
        self.current_X_position = x
        self.current_Y_position = y

    @tool_func
    def set_z_position(self, z: float):
        print("Running function: set_z_position")
        z = max(self.Min_Z_position, min(z, self.Max_Z_position))
        self.current_Z_position = z

    @tool_func
    def get_x_y_position(self) -> Tuple[float, float]:
        print("Running function: get_x_y_position")
        return (self.current_X_position, self.current_Y_position)

    @tool_func
    def get_z_position(self) -> float:
        print("Running function: get_z_position")
        return self.current_Z_position

    @tool_func
    def set_exposure(self, exposure_time: float):
        print("Running function: set_exposure")
        exposure_time = max(self.Min_exposure, min(exposure_time, self.Max_exposure))
        self.current_exposure_time = exposure_time

    @tool_func
    def get_exposure(self) -> float:
        print("Running function: get_exposure")
        return self.current_exposure_time

    @tool_func
    def set_brightness(self, brightness: int):
        print("Running function: set_brightness")
        if self.current_channel != '1-NONE':
            pass
        else:
            brightness = max(self.Min_brightness, min(brightness, self.Max_brightness))
            self.current_brightness = brightness

    @tool_func
    def get_brightness(self) -> int:
        print("Running function: get_brightness")
        return self.current_brightness

    @tool_func
    def set_objective(self, objective_label: str):
        print("Running function: set_objective")
        if objective_label in objective_labels:
            self.current_objective = objective_label
            self.pixel_size = 1.6234 * 4 / objective_labels[self.current_objective]

    @tool_func
    def set_channel(self, channel: str):
        print("Running function: set_channel")
        if channel != '1-NONE':
            self.current_brightness = 0
        if channel in dichroic_colors:
            self.current_channel = channel

    @tool_func
    def get_channel(self):
        print("Running function: get_channel")
        return self.current_channel

    @tool_func
    def get_objective(self):
        print("Running function: get_objective")
        return self.current_objective

    @tool_func
    def start_preview(self):
        print("Running function: start_preview")
        pass

    @tool_func
    def stop_preview(self):
        print("Running function: stop_preview")
        pass

    @tool_func
    def get_image(self, width_micro=None, height_micro=None):
        print("Running function: get_image")
        if width_micro and height_micro:
            return self._acquire_stitch_mosaic(width_micro, height_micro)
        else:
            return self._acquire_single_image()

    def _acquire_single_image(self) -> np.ndarray:
        return np.random.randint(0, 65535,
                                 (self.current_img_height, self.current_img_width),
                                 dtype=self.img_dtype)

    def _acquire_stitch_mosaic(self, width_micro: float, height_micro: float, overlap=0.1) -> np.ndarray:
        if overlap < 0.05 or overlap >= 0.5:
            raise ValueError("Overlap rate must be between 5% and 50%")

        fov_width = self.current_img_width * self.pixel_size
        fov_height = self.current_img_height * self.pixel_size
        step_x = fov_width * (1 - overlap)
        step_y = fov_height * (1 - overlap)
        cols = max(1, math.ceil(width_micro / step_x))
        rows = max(1, math.ceil(height_micro / step_y))

        mosaic_height = self.current_img_height * rows
        mosaic_width = self.current_img_width * cols
        mosaic = np.random.randint(0, 65535, (mosaic_height, mosaic_width), dtype=self.img_dtype)

        return mosaic

    @tool_func
    def add_acquisition_position(self, name: str, x: float, y: float, width: float, height: float) -> None:
        print("Running function: add_acquisition_position")
        for pos in self.acquisition_positions:
            if pos["name"] == name:
                return
        self.acquisition_positions.append({
            "name": name,
            "x": x,
            "y": y,
            'width': width,
            'height': height
        })

    @tool_func
    def add_channels(self, channel: str, exposure: float) -> None:
        print("Running function: add_channels")
        for existing in self.acquisition_channels:
            if existing["channel"] == channel:
                return
        self.acquisition_channels.append({
            "channel": channel,
            "exposure": exposure
        })

    @tool_func
    def set_z_stack(self, z_start: float, z_end: float, z_step: float) -> None:
        print("Running function: set_z_stack")
        if z_step <= 0:
            raise ValueError("Z-axis step size must be positive")
        if (z_end - z_start) * z_step < 0:
            raise ValueError("Z-axis step direction conflicts with start and end range")
        self.z_stack_params = {
            "z_start": z_start,
            "z_end": z_end,
            "z_step": z_step
        }

    @tool_func
    def set_time_series(self, num_frames: int, interval_sec: float) -> None:
        print("Running function: set_time_series")
        self.time_lapse_params = {
            "num_frames": num_frames,
            "interval_sec": interval_sec
        }

    @tool_func
    def run_acquisition(self) -> None:
        print("Running function: run_acquisition")
        if not self.acquisition_positions:
            raise ValueError("Please add at least one acquisition position")
        if not self.acquisition_channels:
            raise ValueError("Please configure at least one acquisition channel")

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

        z_positions = np.arange(z_start, z_end + z_step, z_step)
        num_steps = len(z_positions)
        channels_name = [ch["channel"] for ch in self.acquisition_channels]

        position_data = []
        for pos in self.acquisition_positions:
            try:
                x_pos = pos["x"]
                y_pos = pos["y"]
                width = pos["width"]
                height = pos["height"]
                self.set_x_y_position(x_pos, y_pos)

                metadata = self._create_ome_metadata(
                    channel_names=channels_name,
                    time_interval=tim_interval if time_num_frames > 1 else None,
                    microscope="olympus lx83",
                    objective=self.current_objective,
                    pixel_type=self.img_dtype
                )

                position_data.append({
                    "name": pos["name"],
                    "metadata": metadata,
                    "x": x_pos,
                    "y": y_pos,
                    "width": width,
                    "height": height
                })
            except Exception:
                continue

        pixel_sizes = PhysicalPixelSizes(
            Z=self.z_stack_params['z_step'],
            Y=self.pixel_size,
            X=self.pixel_size
        )

        for pos in position_data:
            save_path = os.path.join(self.output_directory, f"{pos['name']}.ome.tif")
            os.makedirs(self.output_directory, exist_ok=True)
            channel_colors = [dichroic_colors.get(ch, "Unknown") for ch in channels_name]
            description = (f'channel_names: {channel_colors}, '
                           f'pixel_size: {self.pixel_size:.2f}, '
                           f'magnification: {objective_labels[self.current_objective]}')
            self._storagemanger.register_file(f"{pos['name']}.ome.tif", description, 'microscope', 'ome-tiff')

        self.acquisition_positions.clear()
        self.acquisition_channels.clear()
        self.z_stack_params = None
        self.time_lapse_params = None
        self.set_x_y_position(init_x, init_y)
        self.set_z_position(init_z)
        self.set_channel(init_channel)
        self.set_exposure(init_exposure)
        self.acquisition_running = False

    def _create_ome_metadata(self, channel_names, time_interval, microscope, objective, pixel_type) -> Dict:
        channel_colors = [dichroic_colors.get(ch, (128, 128, 128)) for ch in channel_names]
        return {
            "channel_names": channel_names,
            "channel_colors": channel_colors,
            "time_interval": time_interval,
            "microscope": microscope,
            "objective": objective,
            "datetime": datetime.now().isoformat(),
            "pixel_type": pixel_type
        }

    def _get_focus_params_for_magnification(self, magnification: float) -> Tuple[float, float]:
        if magnification < 5:
            return 50.0, 10.0
        elif magnification < 10:
            return 30.0, 5.0
        elif magnification < 20:
            return 20.0, 2.0
        elif magnification < 50:
            return 10.0, 1.0
        else:
            return 5.0, 0.5

    @tool_func
    def perform_autofocus(self, tolerance=0.5, use_auto_params=True) -> float:
        print("Running function: perform_autofocus")
        current_z = self.get_z_position()
        best_z = current_z + np.random.uniform(-50, 50)
        best_z = max(self.Min_Z_position, min(best_z, self.Max_Z_position))
        best_z = float(best_z)  # Convert string input to float (core correction)
        self.set_z_position(best_z)
        return best_z

    @tool_func
    def perform_autobrightness(self, tolerance=0.5, max_iterations=5) -> int:
        print("Running function: perform_autobrightness")
        best_brightness = int(self.current_brightness * np.random.uniform(0.8, 1.2))
        best_brightness = max(self.Min_brightness, min(best_brightness, self.Max_brightness))
        best_brightness = float(best_brightness)  # Convert string input to float (core correction)
        self.set_brightness(best_brightness)
        return best_brightness

    @tool_func
    def shutdown(self):
        print("Running function: shutdown")
        self.stop_preview()
        self.acquisition_abort = True
        self.set_brightness(self.Min_brightness)

    @tool_func
    def z_stack_range(self) -> Tuple[int, int]:
        print("Running function: z_stack_range")
        return (5000, 1000)

    @tool_func
    def load_target_locations(self, filename: str) -> List[Tuple[int, int, int, int]]:
        print("Running function: load_target_locations")
        num_regions = 19
        regions = []
        for _ in range(num_regions):
            x = np.random.randint(0, 100)
            y = np.random.randint(0, 100)
            width = np.random.randint(50, 200)
            height = np.random.randint(50, 200)
            regions.append((x, y, width, height))
        
        regions[3] = (54184.6, 33513.0, 500, 500)
        regions[12] = (54415.9, 34164.4, 500, 500)
        return regions

    @tool_func
    def create_96_wells_positions(self, start_position=(0, 0)):
        print("Running function: create_96_wells_positions")
        rows = 6
        cols = 4
        spacing_x = 18
        spacing_y = 18
        start_x, start_y = start_position
        positions = []
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * spacing_x
                y = start_y + row * spacing_y
                positions.append((x, y))
        return positions

    @tool_func
    def create_24_wells_positions(self, start_position=(0, 0)):
        print("Running function: create_24_wells_positions")
        rows = 6
        cols = 4
        spacing_x = 18
        spacing_y = 18
        start_x, start_y = start_position
        positions = []
        for row in range(rows):
            for col in range(cols):
                x = start_x + col * spacing_x
                y = start_y + row * spacing_y
                positions.append((x, y))
        return positions

    def open_laser(self) -> None:
        pass

    def close_laser(self) -> None:
        pass


import pandas as pd
from pathlib import Path
from typing import Sequence, Literal


class Cellpose2D(BaseTool):
    def __init__(self, storagemanger, output_path: str):
        self._storagemanger = storagemanger
        self.output_directory: str = output_path
        self.model = None

    @tool_func
    def cellpose_initialize(self, gpu: bool = False, model_type: str = "cyto3"):
        print("Running function: cellpose_initialize")
        self.model = "MOCK_MODEL"

    @tool_func
    def cellpose_read(self, file_path: str) -> np.ndarray:
        print("Running function: cellpose_read")
        shape = (3, 3, 3, 256, 256)
        image = np.random.rand(*shape).astype(np.float32)
        return image

    @tool_func
    def segment(
        self,
        image: np.ndarray,
        channels: Sequence[int] | None = None,
        diameter: float | None = None,
        flow_threshold: float = 0.4,
        cellprob_threshold: float = 0.0,
        min_size: int = 15,
        denoise: bool = False,
    ) -> np.ndarray:
        print("Running function: segment")
        return np.random.randint(0, 100, (100, 100), dtype=np.int32)

    @tool_func
    def analyze_masks(
        self,
        masks: np.ndarray,
        px_size: float = 1.0,
        unit: Literal["px", "μm2"] = "px",
        bins: int | np.ndarray = 20,
        plot: bool = False,
        **bar_kwargs
    ) -> pd.DataFrame:
        print("Running function: analyze_masks")
        return pd.DataFrame({
            "cell_id": np.arange(1, 101),
            "area": np.random.randint(10, 1000, 100),
            "bin_idx": np.random.randint(0, 10, 100)
        })

    @tool_func
    def save_masks(self, masks: np.ndarray, filename: str | Path, description) -> Path:
        print("Running function: save_masks")
        output_path = Path(self.output_directory, filename).expanduser().resolve()
        return output_path

    @tool_func
    def save_csv(self, df: pd.DataFrame, filename: str | Path) -> Path:
        print("Running function: save_csv")
        output_path = Path(self.output_directory, filename).expanduser().resolve()
        return output_path

    @staticmethod
    def _unique(p: Path) -> Path:
        return p


class ImageJProcessor(BaseTool):
    def __init__(self, storagemanger, output_path: str):
        self._storagemanger = storagemanger
        self.output_directory: str = output_path
        self.ij = None

    @tool_func
    def fiji_initialize(self, fiji_path=None):
        print("Running function: fiji_initialize")
        self.ij = "mock_imagej_instance"
        return True

    @tool_func
    def load_image(self, file_name):
        print("Running function: load_image")
        return f"mock_dataset_{file_name}"

    def _load_image_IMP(self, file_path):
        return f"mock_imp_{os.path.basename(file_path)}"

    def dataset_to_imp(self, dataset):
        return f"mock_imp_from_{dataset}"

    @tool_func
    def save_image(self, dataset, filename, description):
        print("Running function: save_image")
        output_path = os.path.join(self.output_directory, filename)
        self._storagemanger.register_file(filename, description, 'analysis_platform', 'ome-tiff')

    @tool_func
    def adjust_contrast(self, img, saturated=5):
        print("Running function: adjust_contrast")
        return f"mock_contrast_enhanced_{saturated}_{img}"

    def dump_info(self, image):
        pass

    @tool_func
    def split_channels(self, img):
        print("Running function: split_channels")
        return [f"mock_channel_0_{img}", f"mock_channel_1_{img}", f"mock_channel_2_{img}", f"mock_channel_3_{img}"]

    @tool_func
    def merge_channels(self, datasets, colors=None, outpath='merge_output.ome.tif'):
        print("Running function: merge_channels")
        outpath = os.path.join(self.output_directory, outpath)
        return f"mock_merged_{'_'.join(colors)}"

    @tool_func
    def set_lut(self, img, color_name):
        print("Running function: set_lut")
        return f"mock_lut_{color_name}_{img}"

    def _temp_tiff(self, img):
        mock_path = f"/tmp/mock_{img}.tif"
        return mock_path

    @tool_func
    def richardson_lucy(self, img, magnification: int, iterations: int = 50,
                        out_filename: str = "deconvolved_result",
                        out_dir: str = "E:/desk/LLM-MICRO") -> str:
        print("Running function: richardson_lucy")
        return f"mock_deconvolved_{out_filename}"

    @tool_func
    def denoise(self, img, method="Gaussian", radius=2.0):
        print("Running function: denoise")
        return f"mock_denoised_{method}_{img}"

    @tool_func
    def fiji_shutdown(self):
        print("Running function: fiji_shutdown")
        pass

    @tool_func
    def z_projection(self, img, method="max"):
        print("Running function: z_projection")
        return f"mock_proj_{method}_{img}"

    @tool_func
    def quantify_fluorescence(self, image) -> float:
        print("Running function: quantify_fluorescence")
        return 123.45

    @tool_func
    def analysis_platform_find_tumor_position(self, image: np.ndarray, description: str) -> List[
        Tuple[int, int, int, int]]:
        print("Running function: analysis_platform_find_tumor_position")
        num_regions = 19
        regions = []
        for _ in range(num_regions):
            x = np.random.randint(0, 100)
            y = np.random.randint(0, 100)
            width = np.random.randint(50, 200)
            height = np.random.randint(50, 200)
            regions.append((x, y, width, height))
        filename = 'tumor_locations_list.json'
        self._storagemanger.register_file(filename, description, 'analysis_platform', 'json')
        return regions

    @tool_func
    def analysis_platform_find_organoid_position(self, image: np.ndarray, description: str) -> List[
        Tuple[int, int, int, int]]:
        print("Running function: analysis_platform_find_organoid_position")
        num_regions = np.random.randint(0, 10)
        regions = []
        for _ in range(num_regions):
            x = np.random.randint(0, 100)
            y = np.random.randint(0, 100)
            width = np.random.randint(50, 200)
            height = np.random.randint(50, 200)
            regions.append((x, y, width, height))
        filename = 'organoid_locations_list.json'
        self._storagemanger.register_file(filename, description, 'analysis_platform', 'json')
        return regions

    @tool_func
    def analysis_platform_find_lesion_position(self, image: np.ndarray, description: str) -> List[
        Tuple[int, int, int, int]]:
        print("Running function: analysis_platform_find_lesion_position")
        num_regions = np.random.randint(0, 10)
        regions = []
        for _ in range(num_regions):
            x = np.random.randint(0, 100)
            y = np.random.randint(0, 100)
            width = np.random.randint(50, 200)
            height = np.random.randint(50, 200)
            regions.append((x, y, width, height))
        filename = 'lesion_locations_list.json'
        self._storagemanger.register_file(filename, description, 'analysis_platform', 'json')
        return regions

    @tool_func
    def analysis_platform_find_bacteria_position(self, image: np.ndarray, description: str) -> List[
        Tuple[int, int, int, int]]:
        print("Running function: analysis_platform_find_bacteria_position")
        num_regions = np.random.randint(0, 10)
        regions = []
        for _ in range(num_regions):
            x = np.random.randint(0, 100)
            y = np.random.randint(0, 100)
            width = np.random.randint(50, 200)
            height = np.random.randint(50, 200)
            regions.append((x, y, width, height))
        filename = 'bacteria_locations_list.json'
        self._storagemanger.register_file(filename, description, 'analysis_platform', 'json')
        return regions

    @tool_func
    def analysis_platform_find_2Dcell_position(self, image: np.ndarray, description: str) -> List[
        Tuple[int, int, int, int]]:
        print("Running function: analysis_platform_find_2Dcell_position")
        num_regions = np.random.randint(0, 10)
        regions = []
        for _ in range(num_regions):
            x = np.random.randint(0, 100)
            y = np.random.randint(0, 100)
            width = np.random.randint(50, 200)
            height = np.random.randint(50, 200)
            regions.append((x, y, width, height))
        filename = '2Dcell_locations_list.json'
        self._storagemanger.register_file(filename, description, 'analysis_platform', 'json')
        return regions

    @tool_func
    def convert_to_numpy(self, image) -> np.ndarray:
        print("Running function: convert_to_numpy")
        height = np.random.randint(100, 501)
        width = np.random.randint(100, 501)
        random_image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
        return random_image
    