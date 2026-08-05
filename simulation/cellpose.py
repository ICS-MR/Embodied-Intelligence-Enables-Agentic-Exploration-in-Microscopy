from __future__ import annotations

from .common import *

class Cellpose2D(BaseTool):
    REAL_ONLY_METHODS = {
        "export_results": "Cellpose-backed segmentation result export",
    }

    def __init__(self, storagemanger, output_path: str):
        self._storagemanger = storagemanger
        self.output_directory: str = output_path
        self.model = None

    def _require_real_runtime(self, method_name: str) -> None:
        capability = self.REAL_ONLY_METHODS.get(method_name)
        if capability:
            raise_mock_mode_real_runtime_error(
                subsystem="Segmentation",
                mode_field="segmentation_mode",
                capability=capability,
            )

    def _resolve_input_path(self, file_path: str | Path) -> Path:
        candidate = Path(file_path).expanduser()
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = Path(self.output_directory, candidate).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Mock input file does not exist: {resolved}")
        return resolved

    def _save_mock_target_preview(self, output_json_path: Path, masks: np.ndarray) -> Path:
        preview_path = output_json_path.with_name(f"{output_json_path.stem}_annotated.png")
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.touch(exist_ok=True)
        return preview_path

    @tool_func
    def cellpose_initialize(self, gpu: bool = False, model_type: str = "cpsam"):
        print("Running function: cellpose_initialize")
        self.model = "MOCK_MODEL"

    @tool_func
    def cellpose_read(self, file_path: str) -> np.ndarray:
        print("Running function: cellpose_read")
        self._resolve_input_path(file_path)
        shape = (3, 3, 3, 32, 32)
        return np.zeros(shape, dtype=np.float32)

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
        del image, channels, diameter, flow_threshold, cellprob_threshold, min_size, denoise
        return np.ones((32, 32), dtype=np.int32)

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
            "area": np.linspace(50, 500, 100, dtype=np.float64),
            "bin_idx": np.arange(100) % 10,
        })

    @tool_func
    def save_masks(self, masks: np.ndarray, filename: str | Path, description) -> Path:
        print("Running function: save_masks")
        del masks
        output_path = Path(self.output_directory, filename).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.touch(exist_ok=True)
        self._storagemanger.register_file(output_path.name, str(description), 'cellpose', 'tiff')
        return output_path

    @tool_func
    def save_csv(self, df: pd.DataFrame, filename: str | Path) -> Path:
        print("Running function: save_csv")
        output_path = Path(self.output_directory, filename).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        df.to_csv(output_path, index=False)
        self._storagemanger.register_file(output_path.name, "Cellpose analysis CSV", 'cellpose', 'csv')
        return output_path

    @tool_func
    def save_target_locations(
        self,
        masks: np.ndarray,
        source_image_path: str | Path,
        filename: str | Path = "cellpose_target_locations.json",
        description: str = "Cellpose target locations for microscope reacquisition",
        min_area_px: int = 15,
        max_area_px: int | None = None,
        top_k: int | None = None,
    ) -> Path:
        print("Running function: save_target_locations")
        del source_image_path, min_area_px, max_area_px, top_k
        if np.asarray(masks).ndim < 2:
            raise ValueError("masks must be at least 2D")

        output_path = Path(self.output_directory, filename).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("[[1000, 2000, 64, 64]]", encoding="utf-8")
        self._storagemanger.register_file(output_path.name, str(description), "cellpose", "json")
        self._save_mock_target_preview(output_path, np.asarray(masks))
        return output_path

    @tool_func
    def color_masks(self, masks: np.ndarray) -> np.ndarray:
        print("Running function: color_masks")
        if masks.ndim != 2:
            raise ValueError("masks must be a 2D array")

        colored = np.zeros((masks.shape[0], masks.shape[1], 3), dtype=np.uint8)
        unique_labels = [int(label) for label in np.unique(masks) if int(label) != 0]
        palette = [
            (220, 20, 60),
            (65, 105, 225),
            (50, 205, 50),
            (255, 165, 0),
            (138, 43, 226),
            (0, 206, 209),
        ]
        for index, label in enumerate(unique_labels):
            colored[masks == label] = palette[index % len(palette)]
        return colored

    @tool_func
    def export_results(self, masks: np.ndarray, base_filename: str, image: np.ndarray | None = None):
        print("Running function: export_results")
        self._require_real_runtime("export_results")
        self.save_masks(masks, f"{base_filename}_masks.tif", "Cellpose segmentation mask")
        colored_mask = self.color_masks(masks)

        color_path = Path(self.output_directory, f"{base_filename}_colored.png").expanduser().resolve()
        color_path.parent.mkdir(parents=True, exist_ok=True)
        color_path.touch(exist_ok=True)
        self._storagemanger.register_file(color_path.name, "Colored cellpose mask", "cellpose", "png")

        df = self.analyze_masks(masks)
        csv_path = self.save_csv(df, f"{base_filename}_analysis.csv")

        overlay_path: Path | None = None
        if image is not None:
            del image
            overlay_path = Path(self.output_directory, f"{base_filename}_overlay.png").expanduser().resolve()
            overlay_path.parent.mkdir(parents=True, exist_ok=True)
            overlay_path.touch(exist_ok=True)
            self._storagemanger.register_file(overlay_path.name, "Cellpose overlay image", "cellpose", "png")

        return {
            "mask_path": str(Path(self.output_directory, f"{base_filename}_masks.tif").expanduser().resolve()),
            "colored_mask": colored_mask,
            "colored_mask_path": str(color_path),
            "analysis_csv_path": str(csv_path),
            "overlay_path": str(overlay_path) if overlay_path is not None else None,
        }

    @staticmethod
    def _unique(p: Path) -> Path:
        return p
