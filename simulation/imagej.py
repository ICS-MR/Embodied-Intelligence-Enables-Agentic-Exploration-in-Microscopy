from __future__ import annotations

from .common import *

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
        return ImageWithMetadata(
            dataset=f"mock_dataset_{resolved_path.name}",
            center_x_um=0.0,
            center_y_um=0.0,
            center_z_um=0.0,
            pixel_size_x_um=1.0,
            pixel_size_y_um=1.0,
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
        del image_meta
        output_path = Path(self.output_directory, filename).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=True)
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
        self._storagemanger.register_file(output_path.name, f"Image after merging channels {resolved_colors}", 'analysis_platform', 'tiff', False)
        if preview_path:
            preview_output_path = Path(self.output_directory, preview_path).expanduser().resolve()
            preview_output_path.parent.mkdir(parents=True, exist_ok=True)
            preview_output_path.touch(exist_ok=True)
            self._storagemanger.register_file(
                preview_output_path.name,
                f"Preview image after merging channels {resolved_colors}",
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

        self._storagemanger.register_file(overlay_path.name, "Mock TrackMate trajectory overlay image", "analysis_platform", "png", False)
        self._storagemanger.register_file(tracks_csv_path.name, "Mock TrackMate trajectory coordinates", "analysis_platform", "csv", False)
        self._storagemanger.register_file(summary_path.name, "Mock TrackMate tracking summary", "analysis_platform", "json", False)
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
    ) -> List[Tuple[int, int, int, int]]:
        print(f"Running function: analysis_platform_find_target_positions for {target_type}")
        spec = self.detection_targets.get(str(target_type), {})
        filename = spec.get("output_filename") or f"{target_type}_locations_list.json"
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
        return self.save_target_positions(image_meta, regions, description, filename, emit_preview=False)

    @tool_func
    def save_target_positions(
        self,
        image_meta,
        regions_px,
        description: str,
        output_filename: str,
        emit_preview: bool = True,
    ) -> List[Tuple[float, float, float, float]]:
        print("Running function: save_target_positions")
        del emit_preview
        image_meta = self._coerce_image_meta(image_meta)
        normalized_regions = []
        for index, region in enumerate(regions_px):
            if not isinstance(region, (list, tuple)) or len(region) != 4:
                raise ValueError(f"Target region at index {index} must be a 4-item list/tuple, got: {region!r}")
            normalized_regions.append(tuple(map(float, region)))

        output_rows = []
        for cx_px, cy_px, w_px, h_px in normalized_regions:
            output_rows.append([
                float(image_meta.center_x_um) + float(cx_px) * float(image_meta.pixel_size_x_um),
                float(image_meta.center_y_um) + float(cy_px) * float(image_meta.pixel_size_y_um),
                float(w_px) * float(image_meta.pixel_size_x_um),
                float(h_px) * float(image_meta.pixel_size_y_um),
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
        del image_meta
        return np.full((256, 256), fill_value=7, dtype=np.uint8)

    def _coerce_image_meta(self, value: Any) -> ImageWithMetadata:
        if isinstance(value, ImageWithMetadata):
            return value
        return ImageWithMetadata(
            dataset=value,
            center_x_um=0.0,
            center_y_um=0.0,
            center_z_um=0.0,
            pixel_size_x_um=1.0,
            pixel_size_y_um=1.0,
        )

    def _coerce_first_image_meta(self, image_metas: Any) -> ImageWithMetadata:
        if isinstance(image_metas, ImageWithMetadata):
            return image_metas
        if isinstance(image_metas, Sequence) and image_metas:
            return self._coerce_image_meta(image_metas[0])
        return self._coerce_image_meta("mock_empty_dataset")

    def _clone_image_meta(self, image_meta: ImageWithMetadata, *, dataset: Any) -> ImageWithMetadata:
        return ImageWithMetadata(
            dataset=dataset,
            center_x_um=image_meta.center_x_um,
            center_y_um=image_meta.center_y_um,
            center_z_um=image_meta.center_z_um,
            pixel_size_x_um=image_meta.pixel_size_x_um,
            pixel_size_y_um=image_meta.pixel_size_y_um,
        )
