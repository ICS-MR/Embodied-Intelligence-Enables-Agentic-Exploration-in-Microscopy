import os
import tempfile
import time
import warnings
import torch

import imagej
import numpy as np
import scyjava as sj
import tifffile
from scyjava import jimport
import cv2
import json
from mmdet.apis import init_detector, inference_detector
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional
from aicsimageio import AICSImage 

from config.system_config import (
    # Model configuration and weights
    TUMOR_MODEL_CONFIG,
    TUMOR_MODEL_CHECKPOINT,
    LESION_MODEL_CONFIG,
    LESION_MODEL_CHECKPOINT,
    BACTERIA_MODEL_CONFIG,
    BACTERIA_MODEL_CHECKPOINT,
    CELL_2D_MODEL_CONFIG,
    CELL_2D_MODEL_CHECKPOINT,
    ORGANOID_MODEL_CONFIG,
    ORGANOID_MODEL_CHECKPOINT,
    # PSF and FIJI paths
    PSF_40X,
    PSF_60X,
    PSF_100X,
    FIJI_PATH
)

from aicsimageio.types import PhysicalPixelSizes

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

class ImageJProcessor:
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

    def fiji_initialize(self, fiji_path=FIJI_PATH):
        """Synchronously initialize ImageJ environment (directly inline private interface logic without hierarchical calls)"""
        print("Initializing ImageJ environment...")
        if not os.path.exists(fiji_path):
            raise FileNotFoundError(f"Fiji.app path does not exist: {fiji_path}")
        self.ij = imagej.init(fiji_path, mode=imagej.Mode.INTERACTIVE)
        print(f"ImageJ version: {self.ij.getVersion()}")

    # ----------------- File IO -----------------    
    def load_image(self, file_name: str) -> ImageWithMetadata:
        """
        加载OME-TIFF图像并返回包含元数据的ImageWithMetadata对象
        适配 _save_ome_tiff 保存的文件，能读取：
        1. 像素物理尺寸（X/Y/Z轴）
        2. 采集时的物理位置（XYZ中心坐标）
        Args:
            file_name: OME-TIFF文件名（无需完整路径，自动拼接self.output_directory）
        Returns:
            ImageWithMetadata: 包含图像数据集和元数据的对象
        Raises:
            FileNotFoundError: 文件不存在时抛出
            RuntimeError: 读取/解析元数据失败时抛出
        """
        # 1. 拼接完整文件路径并校验文件存在性
        file_path = os.path.join(self.output_directory, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"输入文件不存在: {file_path}")

        try:
            # 2. 用ImageJ打开原始数据集（保留你的原有逻辑）
            dataset = self.ij.io().open(file_path)

            # 3. 用aicsimageio加载OME-TIFF，解析元数据
            aics_img = AICSImage(file_path)

            # --------------------------
            # 核心1：读取像素物理尺寸（适配 _save_ome_tiff 保存的参数）
            # --------------------------
            pixel_sizes: PhysicalPixelSizes = aics_img.physical_pixel_sizes
            # 提取X/Y/Z轴像素尺寸，缺失时给默认值（和保存逻辑对齐）
            pixel_size_x_um = float(pixel_sizes.X or 1.0)
            pixel_size_y_um = float(pixel_sizes.Y or 1.0)
            pixel_size_z_um = float(pixel_sizes.Z or 1.0)  # Z轴尺寸（Z-stack步长）

            # --------------------------
            # 核心2：读取采集时的物理位置（中心坐标）
            # 适配 _save_ome_tiff 中 plane_position_x/y/z 写入的元数据
            # --------------------------
            center_x_um = 0.0
            center_y_um = 0.0
            center_z_um = 0.0

            # 方式1：优先从OME元数据读取平面位置（推荐，适配标准OME-TIFF）
            try:
                # 获取第一个平面的位置（所有平面XY位置相同，Z取第一个即可）
                ome_meta = aics_img.ome_metadata
                center_x_um = float(ome_meta.get_image_position_x(image_index=0) or 0.0)
                center_y_um = float(ome_meta.get_image_position_y(image_index=0) or 0.0)
                center_z_um = float(ome_meta.get_image_position_z(image_index=0) or 0.0)
            except (AttributeError, TypeError, ValueError, IndexError):
                # 方式2：备用方案 - 从原始元数据字典读取
                try:
                    plane_meta = aics_img.metadata.get("Plane", [])
                    if plane_meta and isinstance(plane_meta, list) and len(plane_meta) > 0:
                        center_x_um = float(plane_meta[0].get("PositionX", 0.0))
                        center_y_um = float(plane_meta[0].get("PositionY", 0.0))
                        center_z_um = float(plane_meta[0].get("PositionZ", 0.0))
                except (AttributeError, KeyError, TypeError, ValueError):
                    # 所有方式都失败时，中心坐标默认0.0（不崩溃）
                    center_x_um = center_y_um = center_z_um = 0.0

            # 4. 构造并返回ImageWithMetadata对象
            meta = ImageWithMetadata(
                dataset=dataset,
                center_x_um=center_x_um,
                center_y_um=center_y_um,
                center_z_um=center_z_um,
                pixel_size_x_um=pixel_size_x_um,
                pixel_size_y_um=pixel_size_y_um
            )

            return meta

        except Exception as e:
            # 捕获所有异常并封装，方便排查问题
            raise RuntimeError(
                f"读取OME-TIFF文件失败 {file_path}：{str(e)}"
            ) from e
    
    def _load_image_IMP(self, file_path):
        """Internal method: Load ImagePlus object (no upper hierarchical calls, directly used)"""
        # Unify path conversion to Unix style to avoid Fiji macro command errors
        file_path = file_path.replace("\\", "/")
        macro = f'run("Bio-Formats Importer", "open={file_path} autoscale color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");'
        self.ij.py.run_macro(macro)
        imp = self.ij.py.active_imageplus()
        if imp is None:
            raise IOError(f"Failed to load image file: {file_path}")
        return imp

    def dataset_to_imp(self, dataset):
        """Synchronously convert Dataset to ImagePlus (directly inline private interface logic without hierarchical calls)"""
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

    def save_image(self, image_meta: ImageWithMetadata, filename: str, description: str):
        """Save image and register file"""
        self._save_dataset_impl(image_meta.dataset, filename)
        self._storagemanger.register_file(filename, description, 'Fiji', 'tiff', False)

    def _save_dataset_impl(self, dataset, filename):
        # Core logic of the original save_image
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

    def adjust_contrast(self, image_meta: ImageWithMetadata, saturated=5) -> ImageWithMetadata:
        """Synchronously perform contrast enhancement and return image with metadata"""
        print("Performing automatic contrast adjustment …")
        new_dataset = self._adjust_contrast_impl(image_meta.dataset, saturated)
        return ImageWithMetadata(
            dataset=new_dataset,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            center_z_um=image_meta.center_z_um,
            pixel_size_x_um=image_meta.pixel_size_x_um,
            pixel_size_y_um=image_meta.pixel_size_y_um,
        )

    def _adjust_contrast_impl(self, img, saturated=5):
        """Internal implementation, does not handle metadata"""
        if hasattr(img, 'getImgPlus'):
            imp = self.dataset_to_imp(img)
        else:
            imp = img

        ContrastEnhancer = jimport('ij.plugin.ContrastEnhancer')
        enhancer = ContrastEnhancer()
        enhancer.setNormalize(True)
        enhancer.stretchHistogram(imp, float(saturated))

        dataset = self.ij.convert().convert(imp, jimport('net.imagej.Dataset'))
        # Do not close imp as it may be passed from external
        return dataset

    def dump_info(self, image):
        """Print image information (no hierarchical calls, direct implementation)"""
        print(f" type: {type(image)}")
        print(f"dtype: {image.dtype if hasattr(image, 'dtype') else 'N/A'}")
        print(f"shape: {image.shape if hasattr(image, 'shape') else 'N/A'}")
        print(f" dims: {image.dims if hasattr(image, 'dims') else 'N/A'}")

    # ----------------- Channel Processing -----------------

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
        Split multi-channel Dataset into a list of single-channel Datasets.
        Each output Dataset maintains exactly the same dimension order and axis type as the input,
        only setting the size of the CHANNEL axis to 1 (e.g., [T=2, C=3, Z=5, Y=100, X=100] → [T=2, C=1, Z=5, Y=100, X=100]).

        Parameters:
            img (net.imagej.Dataset): Input multi-channel image (must contain CHANNEL axis)

        Returns:
            List[net.imagej.Dataset]: List of single-channel images, each containing a CHANNEL axis with size=1
        """
        Axes = sj.jimport('net.imagej.axis.Axes')
        Views = sj.jimport('net.imglib2.view.Views')
        Intervals = sj.jimport('net.imglib2.util.Intervals')
        
        ch_idx = img.dimensionIndex(Axes.CHANNEL)
        if ch_idx < 0:
            return [img]  # No channel axis, return original image directly

        num_channels = int(img.dimension(ch_idx))
        channels = []
        
        # Get original dimension sizes and axis objects (for reconstruction)
        orig_num_dims = img.numDimensions()
        orig_dims = [int(img.dimension(d)) for d in range(orig_num_dims)]
        orig_axes = [img.axis(d) for d in range(orig_num_dims)]
        
        for c in range(num_channels):
            # 1. Construct new dimensions: only set channel dimension to 1
            new_dims = orig_dims.copy()
            new_dims[ch_idx] = 1
            
            # 2. Create new image (same data type as original image)
            mins = [0] * orig_num_dims
            maxs = [d - 1 for d in new_dims]
            interval = Intervals.createMinMax(*mins, *maxs)
            new_img = self.ij.op().create().img(interval, img.firstElement())
            
            # 3. Slice the c-th channel from the original image (remove CHANNEL dimension)
            sliced = Views.hyperSlice(img, ch_idx, c)
            
            # 4. Safely copy pixel data (element-wise)
            src_cursor = sliced.cursor()
            dst_cursor = new_img.cursor()
            while src_cursor.hasNext():
                src_cursor.fwd()
                dst_cursor.fwd()
                dst_cursor.get().set(src_cursor.get())
            
            # 5. Create Dataset and set axes (reuse original axes, only modify CHANNEL size)
            ds = self.ij.dataset().create(new_img)
            for dim in range(orig_num_dims):
                axis = orig_axes[dim].copy()
                if dim == ch_idx:
                    axis.setSize(1)
                ds.setAxis(axis, dim)
                
            channels.append(ds)
            
        return channels
    
    def merge_channels(
    self,
    image_metas: List[ImageWithMetadata],
    colors=None,
    outpath='merge_output.ome.tif'
) -> ImageWithMetadata:
        """Merge channels and use metadata from the first image"""
        if not image_metas:
            raise ValueError("Input image list is empty")
        
        # Use metadata from the first image (assuming all channels are spatially aligned)
        ref_meta = image_metas[0]
        datasets = [meta.dataset for meta in image_metas]
        
        merged_dataset = self._merge_channels_impl(datasets, colors, outpath)
        
        return ImageWithMetadata(
            dataset=merged_dataset,
            center_x_um=ref_meta.center_x_um,
            center_y_um=ref_meta.center_y_um,
            center_z_um=ref_meta.center_z_um,
            pixel_size_x_um=ref_meta.pixel_size_x_um,
            pixel_size_y_um=ref_meta.pixel_size_y_um,
        )

    def _merge_channels_impl(self, datasets, colors=None, outpath='merge_output.ome.tif'):
        """Synchronously merge channels (no hierarchical calls, direct implementation of core logic)"""
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
        format_type = "OME-TIFF" if outpath.lower().endswith('.ome.tif') else "Tiff"
        escaped_path = outpath.replace('\\', '\\\\')

        macro = (
            f'run("Merge Channels...", "{" ".join(parts)}"); '
            f'run("Stack to RGB", "slices frames"); '
            f'saveAs("{format_type}", "{escaped_path}");'
        )

        print(f"Executing merge channels macro: {macro}")
        self.ij.py.run_macro(macro)

        merged_imp = self.ij.py.active_imageplus()
        merged_ds = self.ij.convert().convert(merged_imp, jimport('net.imagej.Dataset'))
        description = f'Image after merging channels {colors}'
        self._storagemanger.register_file(os.path.basename(outpath), description, 'Fiji', 'tiff', False)

        # Explicitly close ImagePlus objects to release memory
        for imp in imps:
            imp.close()
        if merged_imp:
            merged_imp.close()

        return merged_ds

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
            raise ValueError(f"Unsupported magnification: {magnification}. Supported values: {list(psf_mapping.keys())}")

        psf_path = psf_mapping[magnification]
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

        self.ij.py.run_macro('run("Close All");')

        if not os.path.isfile(mip_path):
            raise FileNotFoundError("DL2 deconvolution failed, file not generated: " + mip_path)

        decon_ds = self.ij.io().open(mip_path)
        return decon_ds

    # ----------------- Denoising -----------------

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
    def fiji_shutdown(self):
        """Synchronously shut down ImageJ (no hierarchical calls, direct implementation of core logic)"""
        # Release organoid detection model
        if self._organoid_model is not None:
            del self._organoid_model
            self._organoid_model = None
            # Release GPU cache
            torch.cuda.empty_cache()

        if self.ij:
            self.ij.py.run_macro('run("Close All");')
            self.ij.dispose()
            sj.shutdown_jvm()
            self.ij = None
            print("ImageJ has released resources and JVM has been terminated")

    def _init_generic_model(self, config, checkpoint, device):
        if not hasattr(self, '_generic_model_cache'):
            self._generic_model_cache = {}
        key = (config, checkpoint, device)
        if key not in self._generic_model_cache:
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
        model_config: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
        device: str = 'cuda:0',
        score_thr: float = 0.3,
        nms_thr: float = 0.5,
        target_size: int = 512,
        target_class_id: int = 0,
        output_filename: str = 'target_locations_list.json'
    ) -> List[Tuple[float, float, float, float]]:
        """
        Generic target position detection function
        
        Returns: List[(center_x_px, center_y_px, width_px, height_px)] —— Pixel coordinates
        Saves: Physical coordinates in JSON file List[(cx_um, cy_um, w_um, h_um)]
        """

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
                
                if hasattr(image, 'getImgPlus'):
                    imp = self.dataset_to_imp(image)
                else:
                    imp = image
                
                img_np = self.ij.py.from_java(imp.getProcessor().getPixels())
                height = imp.getHeight()
                width = imp.getWidth()
                channels = imp.getNChannels()
                print(f"ImagePlus image information: height {height}, width {width}, channels {channels}")

                if channels == 1:
                    img_np = img_np.reshape((height, width))
                else:
                    print(f"Multi-channel image detected (channels: {channels}), only using first channel")
                    img_np = img_np.reshape((height, width, channels))
                    img_np = img_np[:, :, 0]
                
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
            bboxes_np = np.array(bboxes, dtype=np.float32)
            scores_np = np.array(scores, dtype=np.float32)
            indices = cv2.dnn.NMSBoxes(
                bboxes=bboxes_np.tolist(),
                scores=scores_np.tolist(),
                score_threshold=0.0,
                nms_threshold=iou_threshold
            )
            if len(indices) == 0:
                return []
            if isinstance(indices[0], (list, np.ndarray)):
                indices = [i[0] for i in indices]
            else:
                indices = indices.flatten().tolist()
            return indices

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
            pred = result.pred_instances.cpu()
            print("Generic target detection inference completed")

            score_mask = pred.scores >= score_thr
            if score_mask.sum() == 0:
                print("No valid targets detected (after confidence filtering)")
                return []

            bboxes = pred.bboxes[score_mask].numpy()
            scores = pred.scores[score_mask].numpy()
            labels = pred.labels[score_mask].numpy().astype(int)

            target_mask = labels == target_class_id
            bboxes = bboxes[target_mask]
            scores = scores[target_mask]
            if len(bboxes) == 0:
                print(f"No targets with class ID {target_class_id} detected")
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
                dx_img = cx_px - img_w / 2.0
                dy_img = cy_px - img_h / 2.0
                cx_um = center_x_um + dx_img * pixel_size_um
                cy_um = center_y_um - dy_img * pixel_size_um  # Y-axis flip
                w_um = w_px * pixel_size_um
                h_um = h_px * pixel_size_um
                physical_regions.append([cx_um, cy_um, w_um, h_um])

            print(f"Total {len(pixel_regions)} valid targets detected")

        except Exception as e:
            print(f"Generic target detection failed: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            if device.startswith('cuda'):
                torch.cuda.empty_cache()

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


    def analysis_platform_find_tumor_position(
        self,
        image_meta: ImageWithMetadata,
        description: str,
    ) -> List[Tuple[float, float, float, float]]:
        """Find tumor positions (automatically extract metadata from ImageWithMetadata)"""
        print("Finding tumor target positions in image")
        return self._analysis_platform_find_position(
            image=image_meta.dataset,
            description=description,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            pixel_size_um=image_meta.pixel_size_um,
            target_class_id=0,
            output_filename='tumor_locations_list.json',
            model_config=TUMOR_MODEL_CONFIG,
            model_checkpoint=TUMOR_MODEL_CHECKPOINT
        )

    def analysis_platform_find_lesion_position(
        self,
        image_meta: ImageWithMetadata,
        description: str,
    ) -> List[Tuple[float, float, float, float]]:
        """Find lesion positions (automatically extract metadata from ImageWithMetadata)"""
        print("Finding lesion target positions in image")
        return self._analysis_platform_find_position(
            image=image_meta.dataset,
            description=description,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            pixel_size_um=image_meta.pixel_size_um,
            target_class_id=0,
            output_filename='lesion_locations_list.json',
            model_config=LESION_MODEL_CONFIG,
            model_checkpoint=LESION_MODEL_CHECKPOINT
        )

    def analysis_platform_find_bacteria_position(
        self,
        image_meta: ImageWithMetadata,
        description: str,
    ) -> List[Tuple[float, float, float, float]]:
        """Find bacteria positions (automatically extract metadata from ImageWithMetadata)"""
        print("Finding bacteria target positions in image")
        return self._analysis_platform_find_position(
            image=image_meta.dataset,
            description=description,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            pixel_size_um=image_meta.pixel_size_um,
            target_class_id=0,
            output_filename='bacteria_locations_list.json',
            model_config=BACTERIA_MODEL_CONFIG,
            model_checkpoint=BACTERIA_MODEL_CHECKPOINT
        )

    def analysis_platform_find_2Dcell_position(
        self,
        image_meta: ImageWithMetadata,
        description: str,
    ) -> List[Tuple[float, float, float, float]]:
        """Find 2D cell positions (automatically extract metadata from ImageWithMetadata)"""
        print("Finding 2Dcell target positions in image")
        return self._analysis_platform_find_position(
            image=image_meta.dataset,
            description=description,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            pixel_size_um=image_meta.pixel_size_um,
            target_class_id=0,
            output_filename='2Dcell_locations_list.json',
            model_config=CELL_2D_MODEL_CONFIG,
            model_checkpoint=CELL_2D_MODEL_CHECKPOINT
        )

    def analysis_platform_find_organoid_position(
        self,
        image_meta: ImageWithMetadata,
        description: str,
    ) -> List[Tuple[float, float, float, float]]:
        """Find organoid positions (automatically extract metadata from ImageWithMetadata)"""
        print("Finding organoid target positions in image")
        return self._analysis_platform_find_position(
            image=image_meta.dataset,
            description=description,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            pixel_size_um=image_meta.pixel_size_um,
            target_class_id=0,
            output_filename='organoid_locations_list.json',
            model_config=ORGANOID_MODEL_CONFIG,
            model_checkpoint=ORGANOID_MODEL_CHECKPOINT
        )
    
    def convert_to_numpy(self, image_meta: ImageWithMetadata) -> np.ndarray:
        """
        Convert the image inside an ImageWithMetadata object to a numpy array for processing.

        Args:
            image_meta (ImageWithMetadata): Input image container with metadata

        Returns:
            np.ndarray: Single-channel grayscale numpy array with shape (height, width)
                dtype is uint8 (0-255 pixel values)
        """
        if self.ij is None:
            raise RuntimeError("ImageJ environment not initialized. Call fiji_initialize() first.")

        # Extract the Dataset from ImageWithMetadata
        dataset = image_meta.dataset

        # Convert Dataset to ImagePlus
        imp = self.dataset_to_imp(dataset)
        should_close_imp = True  # We created this ImagePlus, so we must close it

        try:
            # Extract pixel data
            processor = imp.getProcessor()
            if processor is None:
                raise ValueError("Image processor is None; cannot extract pixels.")

            pixels = self.ij.py.from_java(processor.getPixels())
            if pixels is None:
                raise ValueError("Failed to extract pixel data from ImagePlus.")

            # Get dimensions
            width = imp.getWidth()
            height = imp.getHeight()
            n_channels = imp.getNChannels()
            n_slices = imp.getNSlices()
            n_frames = imp.getNFrames()

            expected_total_pixels = width * height
            if len(pixels) != expected_total_pixels:
                # Handle multi-dimensional images by extracting first channel/slice/frame
                if n_channels > 1 or n_slices > 1 or n_frames > 1:
                    print(
                        f"Image is not 2D single-channel (C={n_channels}, Z={n_slices}, T={n_frames}). "
                        "Extracting first channel of first slice and frame."
                    )
                    # Reuse existing channel splitting logic
                    channels = self._split_channels_impl(dataset)
                    if not channels:
                        raise ValueError("Failed to split channels from dataset.")
                    # Recursively process the first (single-channel) dataset
                    first_channel_meta = ImageWithMetadata(
                        dataset=channels[0],
                        center_x_um=image_meta.center_x_um,
                        center_y_um=image_meta.center_y_um,
                        center_z_um=image_meta.center_z_um,
                        pixel_size_x_um=image_meta.pixel_size_x_um,
                        pixel_size_y_um=image_meta.pixel_size_y_um,
                    )
                    return self.convert_to_numpy(first_channel_meta)
                else:
                    raise ValueError(f"Pixel count mismatch: expected {expected_total_pixels}, got {len(pixels)}")

            # Reshape to 2D
            img_array = np.asarray(pixels, dtype=np.float32).reshape((height, width))

            # Handle RGB (unlikely for scientific TIFF, but safe to check)
            if imp.getType() == jimport('ij.ImagePlus').COLOR_RGB:
                img_array = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)

            # Normalize to uint8 [0, 255]
            img_array = self._safe_image_normalize(img_array)

            return img_array.astype(np.uint8)

        finally:
            if should_close_imp:
                imp.close()
