from __future__ import annotations

from .common import *


class Frap(BaseTool):
    """Simulation implementation of the public FRAP API."""

    _active_instance: Frap | None = None

    planning_hint = (
        "Use laser_on before laser_position, then use laser_off when the FRAP "
        "sequence is complete. Coordinates are in microns relative to the field center."
    )
    execution_hint = (
        "This simulation validates the public FRAP call sequence without controlling cellSens."
    )

    def __init__(self, storage_manager=None, output_dir: str = "./output") -> None:
        self.storage_manager = storage_manager
        self.output_dir = output_dir
        self._laser_enabled = False
        type(self)._active_instance = self

    @tool_func
    def laser_on(self, power: float, duration: float) -> None:
        """
        Activate laser for precise cell manipulation.

        Args:
            power: Laser intensity as a percentage from 0.0 to 100.0.
            duration: Exposure time in milliseconds.

        Raises:
            ValueError: If power is outside 0.0-100.0 or duration is not positive.
        """
        if not (0.0 <= float(power) <= 100.0):
            raise ValueError("power must be between 0.0 and 100.0")
        if float(duration) <= 0:
            raise ValueError("duration must be positive")
        self._laser_enabled = True

    @tool_func
    def laser_off(self) -> None:
        """Immediately deactivate laser emission."""
        self._laser_enabled = False

    @tool_func
    @staticmethod
    def laser_position(x: int, y: int) -> None:
        """
        Set laser focal point coordinates.

        Args:
            x: X-axis position in microns relative to the field center.
            y: Y-axis position in microns relative to the field center.
        """
        del x, y
        instance = Frap._require_active_instance()
        if not instance._laser_enabled:
            raise RuntimeError("laser_position requires FRAP to be started first.")

    @tool_func
    @staticmethod
    def cell_detection() -> dict:
        """Detect and return the target position relative to the field center in microns."""
        Frap._require_active_instance()
        return {"x": 0.0, "y": 0.0}

    @tool_func
    @staticmethod
    def cell_contour_extraction() -> dict:
        """Extract field-centered contour points, area, and perimeter in microns."""
        Frap._require_active_instance()
        return {
            "points": [(-10.0, -10.0), (10.0, -10.0), (10.0, 10.0), (-10.0, 10.0)],
            "area": 400.0,
            "perimeter": 80.0,
        }

    @classmethod
    def _require_active_instance(cls) -> Frap:
        instance = cls._active_instance
        if instance is None:
            raise RuntimeError("Frap must be instantiated before using this static method.")
        return instance
