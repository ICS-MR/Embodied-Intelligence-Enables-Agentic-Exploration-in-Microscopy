from __future__ import annotations

from dataclasses import dataclass
import shutil

import tifffile

from core_tool.spatial_metadata import load_ome_spatial_metadata

from .common import *


@dataclass(frozen=True)
class MockImageDataset:
    source_path: str
    label: str

    def __str__(self) -> str:
        return self.label


def _extract_first_2d_plane(image: np.ndarray) -> np.ndarray:
    image_array = np.asarray(image)
    if image_array.ndim == 2:
        return image_array

    squeezed = np.squeeze(image_array)
    if squeezed.ndim == 2:
        return squeezed

    if squeezed.ndim == 3 and squeezed.shape[-1] in (3, 4):
        rgb = np.asarray(squeezed[..., :3], dtype=np.float32)
        return np.mean(rgb, axis=2).astype(squeezed.dtype, copy=False)

    if squeezed.ndim > 2:
        selection = (0,) * (squeezed.ndim - 2) + (slice(None), slice(None))
        first_plane = np.asarray(squeezed[selection])
        if first_plane.ndim == 2:
            return first_plane

    raise ValueError(f"Mock ImageJ expected a 2D image plane, got shape={image_array.shape}")


class ImageJProcessor(BaseTool):
    REAL_ONLY_METHODS = {
        "richardson_lucy": "Fiji-backed Richardson-Lucy deconvolution",
        "trackmate_tracking": "Fiji-backed TrackMate tracking",
    }

    def __init__(
        self,
        storagemanger,
        output_path: str,
        *,
        system_config: Any = None,
        detection_targets: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        del system_config
        self._storagemanger = storagemanger
        self.output_directory: str = output_path
        self.detection_targets = {
            str(target_name): dict(spec)
            for target_name, spec in (detection_targets or {}).items()
        }
        self.ij = None
        self.initialized = False

    def _require_real_runtime(self, method_name: str) -> None:
        capability = self.REAL_ONLY_METHODS.get(method_name)
        if capability:
            raise_mock_mode_real_runtime_error(
                subsystem="Image analysis",
                mode_field="image_analysis_mode",
                capability=capability,
            )

    def _resolve_input_path(self, file_name: str | Path) -> Path:
        candidate = Path(file_name).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = Path(self.output_directory, candidate).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Mock input file does not exist: {resolved}")
        return resolved

    @tool_func
    def fiji_initialize(self, fiji_path=None):
        print("Running function: fiji_initialize")
        del fiji_path
        self.ij = "mock_imagej_instance"
        self.initialized = True
        return True

    @tool_func
    def load_image(self, file_name) -> ImageWithMetadata:
        print("Running function: load_image")
        resolved_path = self._resolve_input_path(file_name)
        spatial_meta = load_ome_spatial_metadata(
            resolved_path,
            require_stage_positions=True,
            require_pixel_sizes=True,
        )
        return ImageWithMetadata(
            dataset=MockImageDataset(
                source_path=str(resolved_path),
                label=f"mock_dataset_{resolved_path.name}",
            ),
            center_x_um=float(spatial_meta["center_x_um"]),
            center_y_um=float(spatial_meta["center_y_um"]),
            center_z_um=float(spatial_meta["center_z_um"]),
            pixel_size_x_um=float(spatial_meta["pixel_size_x_um"]),
            pixel_size_y_um=float(spatial_meta["pixel_size_y_um"]),
        )

    def _load_image_IMP(self, file_path):
        return f"mock_imp_{os.path.basename(file_path)}"

    @tool_func
    def dataset_to_imp(self, dataset):
        dataset_value = dataset.dataset if isinstance(dataset, ImageWithMetadata) else dataset
        return f"mock_imp_from_{dataset_value}"

    @tool_func
    def save_image(self, image_meta, filename, description):
        print("Running function: save_image")
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        source_path = self._source_path_from_dataset(source.dataset)
        if source_path is None:
            raise ValueError("Mock ImageJ save_image requires an image loaded from a real source file")
        output_path = Path(self.output_directory, filename).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, output_path)
        self._storagemanger.register_file(filename, description, 'analysis_platform', 'ome-tiff')
        return str(output_path)

    @tool_func
    def adjust_contrast(self, image_meta, saturated=5) -> ImageWithMetadata:
        print("Running function: adjust_contrast")
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        return self._clone_image_meta(source, dataset=f"mock_contrast_enhanced_{saturated}_{source.dataset}")

    @tool_func
    def dump_info(self, image):
        return {
            "image": str(image),
            "backend": "mock_imagej_instance",
            "output_directory": self.output_directory,
        }

    @tool_func
    def split_channels(self, image_meta) -> List[ImageWithMetadata]:
        print("Running function: split_channels")
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        return [
            self._clone_image_meta(source, dataset=f"mock_channel_{index}_{source.dataset}")
            for index in range(4)
        ]

    @tool_func
    def merge_channels(
        self,
        image_metas,
        colors=None,
        outpath='merge_output.ome.tif',
        preview_path=None,
        preview_seconds: float = 0.0,
        *,
        description: str,
    ) -> ImageWithMetadata:
        print("Running function: merge_channels")
        del preview_seconds
        def normalize_merge_color(color):
            raw_color = str(color).strip()
            color_key = " ".join(raw_color.replace("-", " ").replace("_", " ").split()).lower()
            aliases = {
                "brightfield": "Gray",
                "bright field": "Gray",
                "bf": "Gray",
                "transmitted": "Gray",
                "transmitted light": "Gray",
                "brightfield transmitted": "Gray",
                "gray": "Gray",
                "grey": "Grey",
                "red": "Red",
                "green": "Green",
                "blue": "Blue",
                "cyan": "Cyan",
                "magenta": "Magenta",
                "yellow": "Yellow",
            }
            if color_key not in aliases:
                available_colors = ['Red', 'Green', 'Blue', 'Gray', 'Grey', 'Cyan', 'Magenta', 'Yellow', 'Brightfield']
                raise ValueError(f"Unsupported color: {color}, available colors: {available_colors}")
            return aliases[color_key]

        resolved_colors = [normalize_merge_color(color) for color in (colors or ['Red', 'Green', 'Blue'])]
        output_path = Path(self.output_directory, outpath).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=True)
        task_context = str(description or "").strip()
        merge_details = f"Image after merging channels {resolved_colors}"
        merged_description = f"{task_context}; {merge_details}" if task_context else merge_details
        self._storagemanger.register_file(output_path.name, merged_description, 'analysis_platform', 'tiff', False)
        if preview_path:
            preview_output_path = Path(self.output_directory, preview_path).expanduser().resolve()
            preview_output_path.parent.mkdir(parents=True, exist_ok=True)
            preview_output_path.touch(exist_ok=True)
            preview_details = f"Preview image after merging channels {resolved_colors}"
            preview_description = f"{task_context}; {preview_details}" if task_context else preview_details
            self._storagemanger.register_file(
                preview_output_path.name,
                preview_description,
                'analysis_platform',
                preview_output_path.suffix.lstrip(".") or "png",
                False,
            )
        source = self._coerce_first_image_meta(image_metas)
        return self._clone_image_meta(source, dataset=f"mock_merged_{'_'.join(resolved_colors)}")

    @tool_func
    def set_lut(self, image_meta, color_name) -> ImageWithMetadata:
        print("Running function: set_lut")
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        return self._clone_image_meta(source, dataset=f"mock_lut_{color_name}_{source.dataset}")

    def _temp_tiff(self, img):
        mock_path = f"/tmp/mock_{img}.tif"
        return mock_path

    @tool_func
    def richardson_lucy(self, image_meta, magnification: int, iterations: int = 50,
                        out_filename: str = "deconvolved_result",
                        out_dir: str = "E:/desk/LLM-MICRO") -> ImageWithMetadata:
        print("Running function: richardson_lucy")
        self._require_real_runtime("richardson_lucy")
        del magnification, iterations, out_dir
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        return self._clone_image_meta(source, dataset=f"mock_deconvolved_{out_filename}")

    @tool_func
    def denoise(self, image_meta, method="Gaussian", radius=2.0) -> ImageWithMetadata:
        print("Running function: denoise")
        del radius
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        return self._clone_image_meta(source, dataset=f"mock_denoised_{method}_{source.dataset}")

    @tool_func
    def fiji_shutdown(self):
        print("Running function: fiji_shutdown")
        self.ij = None
        self.initialized = False
        return True

    @tool_func
    def z_projection(self, image_meta, method="max") -> ImageWithMetadata:
        print("Running function: z_projection")
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        return self._clone_image_meta(source, dataset=f"mock_proj_{method}_{source.dataset}")

    @tool_func
    def trackmate_tracking(
        self,
        image_meta,
        spot_radius_um: float | None = None,
        max_linking_distance_um: float | None = None,
        min_track_length: int = 3,
        out_prefix: str = "trackmate",
        *,
        description: str,
    ) -> dict:
        print("Running function: trackmate_tracking")
        self._require_real_runtime("trackmate_tracking")
        del image_meta
        out_prefix = str(out_prefix or "trackmate").strip().replace("\\", "/").strip("/")
        if not out_prefix:
            out_prefix = "trackmate"

        overlay_path = Path(self.output_directory, f"{out_prefix}_overlay.png").expanduser().resolve()
        tracks_csv_path = Path(self.output_directory, f"{out_prefix}_tracks.csv").expanduser().resolve()
        summary_path = Path(self.output_directory, f"{out_prefix}_summary.json").expanduser().resolve()

        for path in (overlay_path, tracks_csv_path, summary_path):
            path.parent.mkdir(parents=True, exist_ok=True)

        overlay_path.touch(exist_ok=True)
        tracks_csv_path.write_text(
            "track_id,frame,t,x_px,y_px,x_um,y_um,x_image_um,y_image_um,quality\n"
            "1,0,0,100,120,54000.0,33400.0,65.0,78.0,1.0\n"
            "1,1,1,112,132,54007.8,33407.8,72.8,85.8,1.0\n"
            "1,2,2,126,146,54016.9,33416.9,81.9,94.9,1.0\n",
            encoding="utf-8",
        )
        summary = {
            "overlay_path": str(overlay_path),
            "tracks_csv_path": str(tracks_csv_path),
            "summary_path": str(summary_path),
            "track_count": 1,
            "spot_count": 3,
            "parameters": {
                "spot_radius_um": spot_radius_um,
                "max_linking_distance_um": max_linking_distance_um,
                "min_track_length": min_track_length,
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        task_context = str(description or "").strip()
        tracking_stats = "track_count: 1; spot_count: 3"
        overlay_details = f"Mock TrackMate trajectory overlay image; {tracking_stats}"
        csv_details = f"Mock TrackMate trajectory coordinates; {tracking_stats}"
        summary_details = f"Mock TrackMate tracking summary; {tracking_stats}"
        self._storagemanger.register_file(
            overlay_path.name,
            f"{task_context}; {overlay_details}" if task_context else overlay_details,
            "analysis_platform",
            "png",
            False,
        )
        self._storagemanger.register_file(
            tracks_csv_path.name,
            f"{task_context}; {csv_details}" if task_context else csv_details,
            "analysis_platform",
            "csv",
            False,
        )
        self._storagemanger.register_file(
            summary_path.name,
            f"{task_context}; {summary_details}" if task_context else summary_details,
            "analysis_platform",
            "json",
            False,
        )
        return summary

    @tool_func
    def quantify_fluorescence(self, image_meta) -> float:
        print("Running function: quantify_fluorescence")
        del image_meta
        return 123.45

    @tool_func
    def analysis_platform_find_target_positions(
        self,
        image_meta,
        target_type: str,
        description: str,
        output_filename: Optional[str] = None,
    ) -> List[Tuple[int, int, int, int]]:
        print(f"Running function: analysis_platform_find_target_positions for {target_type}")
        spec = self.detection_targets.get(str(target_type), {})
        default_filename = spec.get("output_filename") or f"{target_type}_locations_list.json"
        filename = self._validate_detection_output_filename(
            default_filename if output_filename is None else output_filename
        )
        target = str(target_type).lower()
        if target == "tumor":
            regions = [(10, 20, 30, 40), (60, 80, 25, 25)]
        elif target == "organoid":
            regions = [(15, 15, 20, 20), (45, 35, 18, 18)]
        elif target == "lesion":
            regions = [(20, 30, 40, 50)]
        elif target == "bacteria":
            regions = [(5, 5, 12, 12), (25, 18, 10, 10), (40, 33, 8, 8)]
        elif target == "bloodvessel":
            regions = [(30, 10, 60, 12), (22, 48, 44, 10)]
        else:
            regions = [(12, 12, 16, 16)]
        return self._save_target_positions(image_meta, regions, description, filename, emit_preview=False)

    def _reserved_detection_output_filenames(self) -> set[str]:
        reserved_filenames: set[str] = set()
        for target_name, spec in self.detection_targets.items():
            output_filename = ""
            if isinstance(spec, dict):
                output_filename = str(spec.get("output_filename") or "")
            if not output_filename:
                output_filename = f"{target_name}_locations_list.json"
            normalized_filename = output_filename.replace("\\", "/").strip().lower()
            if normalized_filename:
                reserved_filenames.add(normalized_filename)
        return reserved_filenames

    def _validate_detection_output_filename(self, output_filename: str) -> str:
        normalized_filename = str(output_filename or "").replace("\\", "/").strip()
        if not normalized_filename:
            raise ValueError("Detection output_filename must be a non-empty relative JSON filename")
        output_path = Path(normalized_filename)
        if output_path.is_absolute() or ".." in output_path.parts:
            raise ValueError("Detection output_filename must stay under the analysis output directory")
        if output_path.suffix.lower() != ".json":
            raise ValueError("Detection output_filename must end with .json")
        return normalized_filename

    def _validate_custom_detection_output_filename(self, output_filename: str) -> str:
        normalized_filename = self._validate_detection_output_filename(output_filename)
        if normalized_filename.lower() in self._reserved_detection_output_filenames():
            raise ValueError(
                "Custom detection output_filename is reserved by configured model detection targets; "
                "choose a distinct filename such as demo_dot_locations.json"
            )
        return normalized_filename

    @tool_func
    def analysis_platform_save_custom_detection_regions(
        self,
        image_meta,
        regions_px: Sequence[Sequence[float]],
        output_filename: str,
        description: str,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Save agent-generated detection regions without running model detection.

        `regions_px` must contain pixel-space regions in
        (center_x_px, center_y_px, width_px, height_px) format. The function
        converts them to physical microscope coordinates, saves a JSON file,
        registers the artifact, and returns the normalized pixel regions.
        """
        final_output_filename = self._validate_custom_detection_output_filename(output_filename)
        return self._save_target_positions(
            image_meta,
            regions_px,
            description,
            final_output_filename,
            emit_preview=False,
        )

    def _save_target_positions(
        self,
        image_meta,
        regions_px,
        description: str,
        output_filename: str,
        emit_preview: bool = True,
    ) -> List[Tuple[float, float, float, float]]:
        print("Running function: _save_target_positions")
        del emit_preview
        image_meta = self._coerce_image_meta(image_meta)
        normalized_regions = []
        for index, region in enumerate(regions_px):
            if not isinstance(region, (list, tuple)) or len(region) != 4:
                raise ValueError(f"Target region at index {index} must be a 4-item list/tuple, got: {region!r}")
            normalized_regions.append(tuple(map(float, region)))

        image_np = self.convert_to_numpy(image_meta)
        height, width = image_np.shape[:2]
        pixel_size_x_um = float(image_meta.pixel_size_x_um)
        pixel_size_y_um = float(image_meta.pixel_size_y_um)
        image_center_x_um = float(image_meta.center_x_um)
        image_center_y_um = float(image_meta.center_y_um)
        image_center_x_px = (width - 1) / 2.0
        image_center_y_px = (height - 1) / 2.0

        output_rows = []
        for cx_px, cy_px, w_px, h_px in normalized_regions:
            dx_img = float(cx_px) - image_center_x_px
            dy_img = float(cy_px) - image_center_y_px
            output_rows.append([
                image_center_x_um + dx_img * pixel_size_x_um,
                image_center_y_um + dy_img * pixel_size_y_um,
                float(w_px) * pixel_size_x_um,
                float(h_px) * pixel_size_y_um,
            ])

        output_path = Path(self.output_directory, output_filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_rows, indent=2), encoding="utf-8")
        target_count = len(output_rows)
        description_with_count = f"{description}; target_count: {target_count}"
        self._storagemanger.register_file(output_filename, description_with_count, 'analysis_platform', 'json')
        return normalized_regions

    @tool_func
    def analysis_platform_find_tumor_position(self, image_meta, description: str) -> List[
        Tuple[int, int, int, int]]:
        return self.analysis_platform_find_target_positions(image_meta, "tumor", description)

    @tool_func
    def analysis_platform_find_organoid_position(self, image_meta, description: str) -> List[
        Tuple[int, int, int, int]]:
        return self.analysis_platform_find_target_positions(image_meta, "organoid", description)

    @tool_func
    def analysis_platform_find_lesion_position(self, image_meta, description: str) -> List[
        Tuple[int, int, int, int]]:
        return self.analysis_platform_find_target_positions(image_meta, "lesion", description)

    @tool_func
    def analysis_platform_find_bacteria_position(self, image_meta, description: str) -> List[
        Tuple[int, int, int, int]]:
        return self.analysis_platform_find_target_positions(image_meta, "bacteria", description)

    @tool_func
    def analysis_platform_find_2Dcell_position(self, image_meta, description: str) -> List[
        Tuple[int, int, int, int]]:
        return self.analysis_platform_find_target_positions(image_meta, "2Dcell", description)

    @tool_func
    def analysis_platform_find_BloodVessel_position(self, image_meta, description: str) -> List[
        Tuple[int, int, int, int]]:
        return self.analysis_platform_find_target_positions(image_meta, "BloodVessel", description)

    @tool_func
    def convert_to_numpy(self, image_meta) -> np.ndarray:
        print("Running function: convert_to_numpy")
        source = image_meta if isinstance(image_meta, ImageWithMetadata) else self._coerce_image_meta(image_meta)
        source_path = self._source_path_from_dataset(source.dataset)
        if source_path is None:
            raise ValueError("Mock ImageJ cannot convert an image without a real source file")
        try:
            image = tifffile.imread(str(source_path))
        except Exception as exc:
            raise RuntimeError(f"Mock ImageJ failed to read image pixels from {source_path}: {exc}") from exc
        return _extract_first_2d_plane(image)

    def _source_path_from_dataset(self, dataset: Any) -> Optional[Path]:
        if isinstance(dataset, MockImageDataset):
            path = Path(dataset.source_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"Mock ImageJ source image does not exist: {path}")
            return path
        if isinstance(dataset, (str, Path)):
            candidate = Path(dataset).expanduser()
            if candidate.exists():
                return candidate.resolve()
        return None

    def _coerce_image_meta(self, value: Any) -> ImageWithMetadata:
        if isinstance(value, ImageWithMetadata):
            return value
        if isinstance(value, (str, Path)):
            return self.load_image(value)
        raise TypeError(f"Expected ImageWithMetadata or image filename, got {type(value).__name__}")

    def _coerce_first_image_meta(self, image_metas: Any) -> ImageWithMetadata:
        if isinstance(image_metas, ImageWithMetadata):
            return image_metas
        if isinstance(image_metas, Sequence) and not isinstance(image_metas, (str, bytes, bytearray)) and image_metas:
            return self._coerce_image_meta(image_metas[0])
        raise ValueError("Input image list is empty")

    def _clone_dataset_with_source(self, source_dataset: Any, label: Any) -> Any:
        if isinstance(label, MockImageDataset):
            return label
        if isinstance(source_dataset, MockImageDataset):
            return MockImageDataset(source_path=source_dataset.source_path, label=str(label))
        return label

    def _clone_image_meta(self, image_meta: ImageWithMetadata, *, dataset: Any) -> ImageWithMetadata:
        return ImageWithMetadata(
            dataset=self._clone_dataset_with_source(image_meta.dataset, dataset),
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            center_z_um=image_meta.center_z_um,
            pixel_size_x_um=image_meta.pixel_size_x_um,
            pixel_size_y_um=image_meta.pixel_size_y_um,
        )
