from __future__ import annotations

from .common import *

class Frap(BaseTool):
    planning_hint = (
        "Use this tool for a compact FRAP workflow API: enable_frap, disable_frap, "
        "laser_position, and cell_contourextraction. When no single API exactly "
        "matches the requested outcome, first consider whether the task can be "
        "achieved by composing these atomic capabilities into a valid short plan."
    )
    execution_hint = (
        "Call enable_frap() before laser_position(). Use "
        "cell_contourextraction(image_path) to obtain centroid_px coordinates, then "
        "call disable_frap() when the sequence finishes."
    )

    def __init__(self, storage_manager=None, output_dir: str = "./output") -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir
        self._frap_enabled = False

    @tool_func
    def enable_frap(self, dry_run: bool = False) -> dict:
        """
        Open or switch the FRAP software into FRAP mode.

        Args:
            dry_run: When True, only report the planned software action.

        Returns:
            A dictionary describing the resulting FRAP software mode state.
        """
        print("Running function: enable_frap")
        if not dry_run:
            self._frap_enabled = True
        return {
            "status": "ok",
            "frap_enabled": True,
            "dry_run": bool(dry_run),
            "mode_action": "mock_enable",
            "action": None,
        }

    @tool_func
    def disable_frap(self, dry_run: bool = False) -> dict:
        """
        Close or exit FRAP mode in the FRAP software.

        Args:
            dry_run: When True, only report the planned software action.

        Returns:
            A dictionary describing the resulting FRAP software mode state.
        """
        print("Running function: disable_frap")
        if not dry_run:
            self._frap_enabled = False
        return {
            "status": "ok",
            "frap_enabled": False,
            "dry_run": bool(dry_run),
            "mode_action": "mock_disable",
            "action": None,
        }

    @tool_func
    def laser_position(self, x_px: float, y_px: float, dry_run: bool = False) -> dict:
        """
        Move to an image-relative target position and trigger the FRAP click.

        Args:
            x_px: Horizontal pixel offset relative to the image center, where
                the image center is always ``(0, 0)``. Positive x moves right
                from the center. Do not interpret this as a top-left-origin
                absolute image pixel coordinate.
            y_px: Vertical pixel offset relative to the image center, where
                the image center is always ``(0, 0)``. Positive y moves down
                from the center. Do not interpret this as a top-left-origin
                absolute image pixel coordinate.
            dry_run: When True, only report the planned movement and click.

        Returns:
            A dictionary describing the mapped ROI-panel target and click result.
        """
        print("Running function: laser_position")
        if not self._frap_enabled and not dry_run:
            raise RuntimeError("FRAP is not enabled. Call enable_frap() before laser_position().")
        return {
            "status": "ok",
            "frap_enabled": self._frap_enabled,
            "dry_run": bool(dry_run),
            "coordinate_system": "image_centered_pixels",
            "input_target_px": {
                "x_px": float(x_px),
                "y_px": float(y_px),
            },
            "image_pixel_target": {
                "x_px": float(x_px),
                "y_px": float(y_px),
            },
            "roi_panel_offset_px": {
                "x_px": int(round(float(x_px))),
                "y_px": int(round(float(y_px))),
            },
            "window": {
                "title": "mock_frap_window",
                "left": 0,
                "top": 0,
                "width": 512,
                "height": 512,
            },
            "activation": {
                "status": "planned" if dry_run else "activated",
                "dry_run": bool(dry_run),
            },
            "pre_actions": [],
            "click": {
                "status": "planned" if dry_run else "clicked",
                "screen_x": int(round(float(x_px))),
                "screen_y": int(round(float(y_px))),
                "dry_run": bool(dry_run),
                "move_duration_sec": 0.0,
            },
            "post_actions": [],
        }

    @tool_func
    def cell_contourextraction(self, image_path: str) -> dict:
        print("Running function: cell_contourextraction")
        resolved_path = self._resolve_input_path(image_path)
        if not resolved_path.exists():
            return {}

        image_width = 512
        image_height = 512
        half_size = 40.0
        contour_points = [
            (-half_size, -half_size),
            (half_size, -half_size),
            (half_size, half_size),
            (-half_size, half_size),
        ]
        area = float((2 * half_size) * (2 * half_size))
        perimeter = float(8 * half_size)
        return {
            "points": contour_points,
            "area": area,
            "perimeter": perimeter,
            "centroid_px": {
                "x_px": 0.0,
                "y_px": 0.0,
            },
            "coordinate_system": "image_centered_pixels",
            "image_width": image_width,
            "image_height": image_height,
            "source_image": str(resolved_path),
        }

    def _resolve_input_path(self, raw_path: str | Path) -> Path:
        candidate = Path(raw_path).expanduser()
        registered = self._resolve_registered_input_path(candidate)
        if registered is not None:
            return registered
        if candidate.is_absolute():
            return candidate.resolve()
        return Path(self.output_dir, candidate).expanduser().resolve()

    def _resolve_registered_input_path(self, candidate: Path) -> Path | None:
        if self.storage_manager is None or not hasattr(self.storage_manager, "read_log"):
            return None
        try:
            registered = self.storage_manager.read_log(True)
        except Exception:
            return None
        if not isinstance(registered, dict):
            return None

        lookup_keys = [item for item in {str(candidate).strip(), candidate.name.strip()} if item]
        matched_meta = None
        for key in lookup_keys:
            meta = registered.get(key)
            if isinstance(meta, dict):
                matched_meta = meta
                break

        if matched_meta is None:
            for meta in registered.values():
                if not isinstance(meta, dict):
                    continue
                filename = str(meta.get("filename", "") or "").strip()
                if filename and filename in lookup_keys:
                    matched_meta = meta
                    break

        if not isinstance(matched_meta, dict):
            return None
        filename = str(matched_meta.get("filename", "") or "").strip()
        if not filename:
            return None
        return Path(self.output_dir, filename).expanduser().resolve()
